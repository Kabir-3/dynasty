# team_tools.py — trade helpers aligned with app.py
import pandas as pd
import numpy as np

POS_ORDER = ["QB", "RB", "WR", "TE", "K"]

def _score_col(df: pd.DataFrame) -> str:
    """Pick the best available score column from the app."""
    if "WinNowScore" in df.columns:
        return "WinNowScore"
    if "edge_z_adj" in df.columns:
        return "edge_z_adj"
    if "edge" in df.columns:
        return "edge"
    return "__zero__"  # sentinel -> treat as 0

def _score_series(df: pd.DataFrame) -> pd.Series:
    col = _score_col(df)
    if col == "__zero__":
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

def pos_strength_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team positional strength using sum of true_value (fallback if no VORP).
    Returns a wide table of z-scores for easy 'need vs surplus' reading.
    """
    metric = "vorp" if "vorp" in df.columns else "true_value"
    x = df.copy()
    if metric not in x.columns:
        # last resort: use market_value so table still renders
        metric = "market_value" if "market_value" in x.columns else None
    if metric is None:
        # create a neutral metric if absolutely nothing is available
        x["__metric__"] = 0.0
        metric = "__metric__"

    grp = (
        x.groupby(["display_name", "pos"], dropna=False)[metric]
          .sum()
          .reset_index()
    )
    pivot = grp.pivot(index="display_name", columns="pos", values=metric).fillna(0.0)

    # z-score per position column
    z = (pivot - pivot.mean()) / (pivot.std(ddof=0).replace(0, np.nan))
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z = z.reindex(columns=POS_ORDER, fill_value=0.0)
    z["TOTAL"] = z.sum(axis=1)
    return z.sort_values("TOTAL", ascending=False)

def team_needs(
    df: pd.DataFrame,
    team: str,
    need_thresh: float = -0.10,
    surplus_thresh: float = 0.10,
):
    """
    Identify 'need' positions (<= need_thresh) and 'surplus' (>= surplus_thresh)
    using z-scores from pos_strength_table.
    """
    z = pos_strength_table(df)
    if team not in z.index:
        return [], []
    row = z.loc[team]
    needs = [p for p in POS_ORDER if row.get(p, 0.0) <= need_thresh]
    surplus = [p for p in POS_ORDER if row.get(p, 0.0) >= surplus_thresh]
    return needs, surplus

def trade_targets_for_team(
    df: pd.DataFrame,
    my_team: str,
    top_n: int = 25,
    need_thresh: float = -0.10,
    surplus_thresh: float = 0.10,
):
    """
    Buy targets for my_team:
      - positive score (undervalued per chosen score column)
      - on teams with surplus at that position
      - at positions my_team needs
    Falls back to league-wide positive score if strict filter is empty.
    """
    score = _score_series(df)
    needs, _ = team_needs(df, my_team, need_thresh, surplus_thresh)
    z = pos_strength_table(df)
    surplus_teams = {pos: set(z.index[z[pos] >= surplus_thresh].tolist()) for pos in POS_ORDER}

    base = df[(df["display_name"] != my_team)].copy()
    base["__score__"] = score.reindex(base.index).fillna(0.0)
    base = base[base["__score__"] > 0]

    if needs:
        base = base[base["pos"].isin(needs)]
        base = base[
            base.apply(lambda r: r["display_name"] in surplus_teams.get(r["pos"], set()), axis=1)
        ]

    if base.empty:
        base = df[(df["display_name"] != my_team)].copy()
        base["__score__"] = score.reindex(base.index).fillna(0.0)
        base = base[base["__score__"] > 0]

    return base.sort_values(["__score__", "true_value"], ascending=[False, False]).head(top_n).drop(columns="__score__", errors="ignore")

def give_list_for_partner(
    df: pd.DataFrame,
    my_team: str,
    partner: str,
    top_n: int = 20,
    need_thresh: float = -0.10,
    surplus_thresh: float = 0.10,
):
    """
    Players you could send:
      - negative score (sell-high)
      - from positions where you have surplus and partner has need
    Falls back to any negative-score players on your team if strict
    filter is empty.
    """
    score = _score_series(df)
    _, my_surplus = team_needs(df, my_team, need_thresh, surplus_thresh)
    partner_needs, _ = team_needs(df, partner, need_thresh, surplus_thresh)
    give_positions = [p for p in my_surplus if p in partner_needs]

    base = df[(df["display_name"] == my_team)].copy()
    base["__score__"] = score.reindex(base.index).fillna(0.0)
    base = base[base["__score__"] < 0]

    if give_positions:
        base = base[base["pos"].isin(give_positions)]

    return base.sort_values(["__score__", "true_value"], ascending=[True, False]).head(top_n).drop(columns="__score__", errors="ignore")

def receive_list_from_partner(
    df: pd.DataFrame,
    my_team: str,
    partner: str,
    top_n: int = 20,
    need_thresh: float = -0.10,
    surplus_thresh: float = 0.10,
):
    """
    Players you could receive:
      - positive score (buy-low)
      - from positions where partner has surplus and you have need
    Falls back to any positive-score players on partner if strict
    filter is empty.
    """
    score = _score_series(df)
    my_needs, _ = team_needs(df, my_team, need_thresh, surplus_thresh)
    _, partner_surplus = team_needs(df, partner, need_thresh, surplus_thresh)
    get_positions = [p for p in my_needs if p in partner_surplus]

    base = df[(df["display_name"] == partner)].copy()
    base["__score__"] = score.reindex(base.index).fillna(0.0)
    base = base[base["__score__"] > 0]

    if get_positions:
        base = base[base["pos"].isin(get_positions)]

    return base.sort_values(["__score__", "true_value"], ascending=[False, False]).head(top_n).drop(columns="__score__", errors="ignore")

def quick_balance_score(send_df: pd.DataFrame, recv_df: pd.DataFrame):
    """
    Rough fairness using market_value totals.
    Returns (send_total, recv_total, diff, score in [0,1]).
    """
    s_total = float(send_df["market_value"].sum()) if len(send_df) else 0.0
    r_total = float(recv_df["market_value"].sum()) if len(recv_df) else 0.0
    diff = r_total - s_total
    denom = max(1.0, s_total + r_total)
    score = 1.0 - min(1.0, abs(diff) / denom)
    return s_total, r_total, diff, score
