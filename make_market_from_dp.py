# make_market_from_dp.py  (short-line, wrap-safe)
import pandas as pd
import numpy as np
import re
import sys

SRC = "dp_values_players.csv"
OUT = "market.csv"
URL = ("https://raw.githubusercontent.com/dynastyprocess/data/master/files/"
       "values-players.csv")

def load_dp_values():
    try:
        return pd.read_csv(SRC, low_memory=False)
    except FileNotFoundError:
        # auto-download if missing
        try:
            import urllib.request
            print("Downloading DynastyProcess values ...")
            urllib.request.urlretrieve(URL, SRC)
            return pd.read_csv(SRC, low_memory=False)
        except Exception as e:
            msg = (
                "Could not load or download dp_values_players.csv.\n"
                "Try manually:\n"
                "  curl -L -o dp_values_players.csv {}\n"
                "Error: {}"
            ).format(URL, e)
            sys.exit(msg)

def pick_col(df, options):
    # return first column that exists (case-insensitive)
    lower = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt in lower:
            return lower[opt]
    return None

def find_value_cols(df):
    cols = []
    pat = r"(value_1qb|ktc_1qb|fc_1qb|ecr_1qb|dlf_1qb|fp_1qb)"
    for c in df.columns:
        if re.search(pat, c, re.IGNORECASE):
            cols.append(c)
    if not cols:
        # last resort: anything ending with _1qb
        for c in df.columns:
            if c.lower().endswith("_1qb"):
                cols.append(c)
    return cols

def main():
    df = load_dp_values()

    # find columns
    name = pick_col(df, ["player_name", "name", "full_name", "player"])
    pos  = pick_col(df, ["position", "pos", "position_text", "position_group"])

    if not name or not pos:
        prev = ", ".join(list(df.columns)[:25])
        sys.exit("Missing name/pos columns. Saw: {}".format(prev))

    val_cols = find_value_cols(df)
    if not val_cols:
        sys.exit(
            "No 1QB value columns found. Look for value_1qb, ktc_1qb, fc_1qb, "
            "ecr_1qb, dlf_1qb, fp_1qb, or any *_1qb column."
        )

    # build market frame
    vals = df[val_cols].apply(pd.to_numeric, errors="coerce")
    val = vals.mean(axis=1)
    mask = val.notna()

    m = pd.DataFrame({
        "name": df.loc[mask, name].astype(str)
                    .str.replace(r"\s+", " ", regex=True).str.strip(),
        "pos":  df.loc[mask, pos].astype(str).str.upper().str.strip(),
        "market_value": val[mask]
    })

    # map/limit positions
    m["pos"] = m["pos"].replace({"PK": "K", "DEF": "DST", "DST": "DST"})
    m = m[m["pos"].isin(["QB", "RB", "WR", "TE", "K"])]

    # scale to ~60..100 to match app scale
    if len(m) == 0 or m["market_value"].max() == m["market_value"].min():
        m["market_value"] = 80.0
    else:
        vmin = float(m["market_value"].min())
        vmax = float(m["market_value"].max())
        m["market_value"] = 60 + (m["market_value"] - vmin) * (40.0 / (vmax - vmin))
        m["market_value"] = m["market_value"].round(1)

    m.to_csv(OUT, index=False)
    print("Saved {} with {} players. Using {} source cols."
          .format(OUT, len(m), len(val_cols)))

if __name__ == "__main__":
    main()

