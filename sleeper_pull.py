import requests
import pandas as pd

def get_json(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_league_data(league_id: str) -> pd.DataFrame:
    # Users (to get display_name for owners)
    users = pd.DataFrame(get_json(f"https://api.sleeper.app/v1/league/{league_id}/users"))

    # Rosters for the league
    rosters = pd.DataFrame(get_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters"))

    # Player metadata dictionary keyed by player_id
    players = get_json("https://api.sleeper.app/v1/players/nfl")

    # Some rosters may have None for "players" — make it an empty list so explode is safe
    rosters["players"] = rosters["players"].apply(lambda x: x if isinstance(x, list) else [])

    # One row per player on a roster
    rosters = rosters.explode("players").rename(columns={"players": "player_id"})

    # Join owner display_name
    rosters = rosters.merge(
        users[["user_id", "display_name"]],
        left_on="owner_id",
        right_on="user_id",
        how="left"
    )

    # Lookups from the players dict
    def lookup(pid, field):
        try:
            return players.get(str(pid), {}).get(field)
        except Exception:
            return None

    rosters["name"] = rosters["player_id"].apply(lambda x: lookup(x, "full_name"))
    rosters["pos"]  = rosters["player_id"].apply(lambda x: lookup(x, "position"))
    rosters["age"]  = rosters["player_id"].apply(lambda x: lookup(x, "age"))
    rosters["team"] = rosters["player_id"].apply(lambda x: lookup(x, "team"))

    # Keep only useful columns
    keep = ["display_name", "player_id", "name", "pos", "age", "team"]
    return rosters[keep].dropna(subset=["name", "pos"]).reset_index(drop=True)

if __name__ == "__main__":
    test_id = "1195252934627844096"  # replace with any league id to quick test
    df = fetch_league_data(test_id)
    print(df.head(12))
    df.to_csv("league_rosters.csv", index=False)
    print("Saved league_rosters.csv")
