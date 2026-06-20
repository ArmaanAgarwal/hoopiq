import pandas as pd
import numpy as np

def load_games():
    df = pd.read_csv("data/games.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    return df

def add_opponent_points(df):
    df["OPP_PTS"] = df["PTS"] - df["PLUS_MINUS"]
    return df

def add_rolling_features(df, windows = [3,5,10,20]):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for w in windows:
        df[f"L{w}_PTS"] = (
            df.groupby("TEAM_ID")["PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f"L{w}_OPP_PTS"] = (
            df.groupby("TEAM_ID")["OPP_PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods = 1).mean())
        )
        df[f"L{w}_MARGIN"] = df[f"L{w}_PTS"] - df[f"L{w}_OPP_PTS"]
        df[f"L{w}_WIN_PCT"] = (
            df.groupby("TEAM_ID")["WIN"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods = 1).mean())
        )
    return df

def add_opponent_adjustment(df):
    opp = df[["GAME_ID", "TEAM_ID", "L10_MARGIN"]].copy()
    opp = opp.rename(columns={
        "TEAM_ID": "OPP_TEAM_ID",
        "L10_MARGIN": "OPP_L10_MARGIN"
    })

    merged = df.merge(opp, on="GAME_ID")
    merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()
    merged["SCHEDULE_STRENGTH"] = merged["OPP_L10_MARGIN"]

    '''dupes = merged[merged.duplicated(subset=["GAME_ID", "TEAM_ID"], keep=False)]
    print(f"Duplicate rows found: {len(dupes)}")'''

    return merged

def add_rest_features(df):
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)
    
    df["PREV_GAME_DATE"] = df.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    df["REST_DAYS"] = (df["GAME_DATE"] - df["PREV_GAME_DATE"]).dt.days
    
    # First game of season has no previous game - fill with a typical rest value
    df["REST_DAYS"] = df["REST_DAYS"].fillna(3)
    # Cap extreme values (offseason gaps) at 7 days
    df["REST_DAYS"] = df["REST_DAYS"].clip(upper=7)
    
    df["BACK_TO_BACK"] = (df["REST_DAYS"] <= 1).astype(int)
    
    # Games played in last 7 days (fatigue measure)
    df = df.set_index("GAME_DATE")
    df["GAMES_LAST_7D"] = (
        df.groupby("TEAM_ID")["WIN"]
        .transform(lambda x: x.rolling("7D").count().shift(1))
    )
    df = df.reset_index()
    df["GAMES_LAST_7D"] = df["GAMES_LAST_7D"].fillna(0)
    
    return df

def build_matchup_rows(df):
    # The feature columns we want for each team
    feature_cols = [
        "L3_MARGIN", "L5_MARGIN", "L10_MARGIN", "L20_MARGIN",
        "L3_WIN_PCT", "L5_WIN_PCT", "L10_WIN_PCT", "L20_WIN_PCT",
        "L10_PTS", "L10_OPP_PTS",
        "SCHEDULE_STRENGTH", "REST_DAYS", "BACK_TO_BACK", "GAMES_LAST_7D"
    ]

    # Split into home and away rows
    home = df[df["HOME"] == 1].copy()
    away = df[df["HOME"] == 0].copy()

    # Rename feature columns with HOME_ / AWAY_ prefixes
    home = home.rename(columns={c: f"HOME_{c}" for c in feature_cols})
    away = away.rename(columns={c: f"AWAY_{c}" for c in feature_cols})

    # Keep only what we need from each
    home_keep = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "WIN"] + [f"HOME_{c}" for c in feature_cols]
    away_keep = ["GAME_ID", "TEAM_ABBREVIATION"] + [f"AWAY_{c}" for c in feature_cols]

    home = home[home_keep].rename(columns={"TEAM_ABBREVIATION": "HOME_TEAM", "WIN": "HOME_WIN"})
    away = away[away_keep].rename(columns={"TEAM_ABBREVIATION": "AWAY_TEAM"})

    # Merge the two halves on GAME_ID
    games = home.merge(away, on="GAME_ID")

    return games
    
if __name__ == "__main__": 
    df = load_games()
    print(f"Loaded {len(df)} rows")

    df = add_opponent_points(df)
    print("Added opponent points")

    df = add_rolling_features(df)
    print("Added rolling features")

    df = add_opponent_adjustment(df)
    print("Added opponent adjustment")
    
    df = add_rest_features(df)
    print("Added rest features")

    df = build_matchup_rows(df)
    print(f"Built {len(df)} matchup rows")

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    print(f"After dropping rows with missing data: {len(df)} games")

    df.to_csv("data/games_model_ready.csv", index=False)
    print("Saved to data/games_model_ready.csv")

    print(df[["GAME_DATE", "HOME_TEAM", "AWAY_TEAM", "HOME_L10_MARGIN", "AWAY_L10_MARGIN", "HOME_WIN"]].tail(5).to_string())

