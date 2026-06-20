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

if __name__ == "__main__":
    df = load_games()
    print(f"Loaded {len(df)} rows")

    df = add_opponent_points(df)
    print("Added opponent points")

    df = add_rolling_features(df)
    print("Added rolling features")

    sample = df[df["TEAM_ABBREVIATION"] == "BOS"].tail(5)
    cols = ["GAME_DATE", "TEAM_ABBREVIATION", "PTS", "OPP_PTS", 
            "L10_PTS", "L10_OPP_PTS", "L10_MARGIN", "L10_WIN_PCT", "WIN"]
    print(sample[cols].to_string())
    
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df.to_csv("data/games_features.csv", index=False)
    print(f"\nSaved {len(df)} rows with features to data/games_features.csv")