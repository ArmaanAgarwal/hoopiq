import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import os

SEASONS = [
    "2000-01", "2001-02", "2002-03", "2003-04", "2004-05",
    "2005-06", "2006-07", "2007-08", "2008-09", "2009-10",
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15",
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"
]

COLS = [
    "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", 
    "MATCHUP", "WL", "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
    "REB", "AST", "STL", "BLK", "TOV", "PLUS_MINUS"
]

def fetch_season(season):
    print(f"    Fetching {season}...")
    try: 
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable = season,
            league_id_nullable = "00"
        )
        df = finder.get_data_frames()[0]
        time.sleep(0.6)
        return df
    except Exception as e:
        print(f"    Error on {season}: {e}")
        return None
    
def fetch_today():
    today = datetime.now().strftime("%m/%d/%Y")
    print(f"    Fetching today's games ({today})...")
    try:
        board = scoreboardv2.ScoreboardV2(
            game_date = today,
            league_id = "00"
        )
        games = board.get_data_frames()[0]
        time.sleep(0.6)
        return games
    except Exception as e:
        print(f"    Error fetching today: {e}")
        return None

def clean_games(df):
    df = df[df["WL"].notna()].copy()
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["HOME"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df = df.sort_values("GAME_DATE").reset_index(drop = True)
    available = [c for c in COLS + ["WIN", "HOME"] if c in df.columns]
    return df[available]

def fetch_all_historical():
    all_frames = []
    for season in SEASONS:
        df = fetch_season(season)
        if df is not None:
            all_frames.append(df)
    combined = pd.concat(all_frames, ignore_index = True)
    combined = combined.drop_duplicates(subset = ["GAME_ID", "TEAM_ID"])
    return combined

if __name__ == "__main__":
    os.makedirs("data", exist_ok = True)

    print("=== Pulling historical game data ===")
    raw = fetch_all_historical()
    print(f"Total rows before cleaning: {len(raw)}")

    clean = clean_games(raw)
    print(f"Total rows after cleaning: {len(clean)}")
    print(f"Date range: {clean.GAME_DATE.min()} to {clean.GAME_DATE.max()}")
    print(f"Total unique games: {clean.GAME_ID.nunique()}")

    clean.to_csv("data/games.csv", index = False)
    print("Saved to data/games.csv")

    print("\n=== Today's games ===")
    today = fetch_today()
    if today is not None and len(today) > 0:
        print(today[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].to_string())
        today.to_csv("data/today.csv", index=False)
        print("Saved to data/today.csv")
    else:
        print("No games today or season is off.")

    print("\nFirst 5 rows of historical data:")
    print(clean.head())