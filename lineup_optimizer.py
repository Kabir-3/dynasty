# lineup_optimizer.py
# V1: momentum-based weekly projections + optimal lineup + bench/FA deltas
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---------------- Config / Eligibility ----------------
ELIGIBLE_FLEX = {"RB", "WR", "TE"}
ELIGIBLE_SFLEX = {"QB", "RB", "WR", "TE"}


@dataclass
class LineupConfig:
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1          # RB/WR/TE
    superflex: int = 0     # QB/RB/WR/TE
    k: int = 0
    te_premium: bool = False


def default_config(superflex: bool, te_premium: bool) -> LineupConfig:
    # Basic defaults; you’ll usually pass an explicit config from Sleeper meta
    if superflex:
        return LineupConfig(qb=1, rb=2, wr=2, te=1, flex=1, superflex=1, k=0, te_premium=te_premium)
    return LineupConfig(qb=1, rb=2, wr=2, te=1, flex=2, superflex=0, k=0, te_premium=te_premium)


def make_config(qb:int, rb:int, wr:int, te:int, flex:int, superflex:int, k:int, te_premium:bool) -> LineupConfig:
    return LineupConfig(qb=qb, rb=rb, wr=wr, te=te, flex=flex, superflex=superflex, k=k, te_premium=te_premium)


def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------- Projections (with offseason fallback) ----------------
def project_points_this_week(
    df: pd.DataFrame,
    w_ewma: float = 0.6,
    w_ppg: float = 0.4,
    trend_boost: float = 0.10,   # add 10% of positive trend
    floor_min_games: int = 2,
    pos_prior: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Momentum projection with an offseason fallback:
      proj = shrink(0.6*EWMA + 0.4*PPG) + max(0, trend)*0.10 + small prior
    If ewma/ppg are absent/zero (offseason), fallback to market_value percentile
    within position scaled by a light position baseline.
    """
    x = df.copy()
    for col in ["ewma", "ppg", "trend", "games_played", "market_value", "pos"]:
        if col not in x.columns:
            x[col] = np.nan

    ewma = _safe_num(x["ewma"])
    ppg  = _safe_num(x["ppg"])
    gp   = _safe_num(x["games_played"]).fillna(0)

    # momentum part
    shrink = np.clip((gp - floor_min_games) / max(1, floor_min_games), 0.0, 1.0)
    base_momentum = (w_ewma * ewma.fillna(0.0) + w_ppg * ppg.fillna(0.0)) * (0.5 + 0.5 * shrink)

    trend = _safe_num(x["trend"]).fillna(0.0)
    boost = np.maximum(0.0, trend) * trend_boost

    prior = 0.0
    if pos_prior:
        prior = x["pos"].map(pos_prior).fillna(0.0)

    proj = base_momentum + boost + prior

    # ---- OFFSEASON / NO-DATA FALLBACK ----
    no_signal = (ewma.fillna(0.0) == 0.0) & (ppg.fillna(0.0) == 0.0)
    if no_signal.any():
        mv = _safe_num(x["market_value"])
        pct = mv.groupby(x["pos"]).rank(pct=True, method="average")
        base_pts = x["pos"].map({"QB": 16.0, "RB": 11.5, "WR": 11.0, "TE": 9.0, "K": 8.0}).fillna(10.0)
        proj_fallback = (pct.fillna(0.0) * base_pts).astype(float)
        proj = proj.where(~no_signal, proj_fallback)

    # TE premium nudge if desired (very light; keep logic simple here)
    if "pos" in x.columns and x["pos"].isin(["TE"]).any():
        # If you want to bump TEs in TE premium leagues at projection time,
        # do it by passing a positive prior for "TE" in pos_prior from app.py.
        pass

    x["proj_week"] = proj.fillna(0.0)
    return x


# ---------------- Slotting / Optimizer ----------------
def _fill_slots(candidates: pd.DataFrame, cfg: LineupConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy selection. Adds a 'slot' column so you can see fixed and FLEX/SFLEX distinctly.
    Assumes candidates contain at least ["name","pos","proj_week"].
    """
    pool = candidates.copy()
    starters_parts: List[pd.DataFrame] = []

    def take_pos(pos: str, n: int):
        nonlocal pool, starters_parts
        if n <= 0:
            return
        sub = pool[pool["pos"] == pos]
        picked = sub.nlargest(n, "proj_week")
        if not picked.empty:
            tmp = picked.copy()
            tmp["slot"] = pos
            starters_parts.append(tmp)
            pool = pool.drop(picked.index, errors="ignore")

    # fixed slots
    take_pos("QB", cfg.qb)
    take_pos("RB", cfg.rb)
    take_pos("WR", cfg.wr)
    take_pos("TE", cfg.te)
    take_pos("K",  cfg.k)

    starters = pd.concat(starters_parts, ignore_index=True) if starters_parts else pool.iloc[0:0]

    # FLEX (RB/WR/TE)
    if cfg.flex > 0:
        flex_pool = pool[pool["pos"].isin(ELIGIBLE_FLEX)]
        flex_pick = flex_pool.nlargest(cfg.flex, "proj_week")
        if not flex_pick.empty:
            tmp = flex_pick.copy()
            tmp["slot"] = "FLEX"
            starters = pd.concat([starters, tmp], ignore_index=True)
            pool = pool.drop(flex_pick.index, errors="ignore")

    # SUPERFLEX (QB/RB/WR/TE)
    if cfg.superflex > 0:
        sf_pool = pool[pool["pos"].isin(ELIGIBLE_SFLEX)]
        sf_pick = sf_pool.nlargest(cfg.superflex, "proj_week")
        if not sf_pick.empty:
            tmp = sf_pick.copy()
            tmp["slot"] = "SFLEX"
            starters = pd.concat([starters, tmp], ignore_index=True)
            pool = pool.drop(sf_pick.index, errors="ignore")

    starters_df = starters.copy()
    bench_df = pool.copy()
    return starters_df, bench_df


def recommend_lineup(
    roster_df: pd.DataFrame,
    superflex: bool,
    te_premium: bool,
    pos_prior: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Convenience API using default_config(superflex, te_premium).
    Prefer recommend_lineup_with_cfg when you can pass an explicit config.
    """
    cfg = default_config(superflex=superflex, te_premium=te_premium)
    return recommend_lineup_with_cfg(roster_df, cfg, pos_prior=pos_prior)


def recommend_lineup_with_cfg(
    roster_df: pd.DataFrame,
    cfg: LineupConfig,
    pos_prior: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Input: roster_df with at least ["name","pos"] and ideally ["ewma","ppg","trend","games_played","market_value"]
    Output: (starters_df with 'slot', bench_df, total_projected_points)
    """
    scored = project_points_this_week(roster_df, pos_prior=pos_prior)
    eligible = scored[scored["pos"].isin({"QB", "RB", "WR", "TE", "K"})].copy()
    if "proj_week" not in eligible.columns:
        eligible["proj_week"] = 0.0
    starters_df, bench_df = _fill_slots(eligible, cfg)
    total = float(pd.to_numeric(starters_df["proj_week"], errors="coerce").fillna(0.0).sum())
    return starters_df, bench_df, total


# ---------------- Bench Swap Suggestions ----------------
def bench_swap_suggestions(
    starters_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    cfg: LineupConfig
) -> pd.DataFrame:
    """
    Suggest best one-for-one bench upgrades for each slot type.
    Returns columns: ["slot","starter","candidate","delta_pts"] (empty-safe).
    """
    cols = ["slot", "starter", "candidate", "delta_pts"]
    if starters_df is None or starters_df.empty or bench_df is None or bench_df.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, object]] = []
    starters = starters_df.copy()
    bench = bench_df.copy()

    def best_swap_for_pos(pos: str) -> Optional[Tuple[pd.Series, pd.Series, float]]:
        s_pool = starters[starters["pos"] == pos]
        if s_pool.empty:
            return None
        worst = s_pool.nsmallest(1, "proj_week").iloc[0]
        b_pool = bench[bench["pos"] == pos]
        if b_pool.empty:
            return None
        best_bench = b_pool.nlargest(1, "proj_week").iloc[0]
        delta = float(best_bench["proj_week"] - worst["proj_week"])
        if delta > 0.01:
            return worst, best_bench, delta
        return None

    # fixed slots
    for pos, n in [("QB", cfg.qb), ("RB", cfg.rb), ("WR", cfg.wr), ("TE", cfg.te), ("K", cfg.k)]:
        for _ in range(n):
            swap = best_swap_for_pos(pos)
            if swap:
                worst, cand, delta = swap
                rows.append({"slot": pos, "starter": worst["name"], "candidate": cand["name"], "delta_pts": round(delta, 2)})

    # FLEX (RB/WR/TE)
    for _ in range(cfg.flex):
        s_pool = starters[starters["slot"] == "FLEX"] if "slot" in starters.columns else starters[starters["pos"].isin(ELIGIBLE_FLEX)]
        if s_pool.empty:
            break
        worst = s_pool.nsmallest(1, "proj_week").iloc[0]
        b_pool = bench[bench["pos"].isin(ELIGIBLE_FLEX)]
        if b_pool.empty:
            break
        best_bench = b_pool.nlargest(1, "proj_week").iloc[0]
        delta = float(best_bench["proj_week"] - worst["proj_week"])
        if delta > 0.01:
            rows.append({"slot": "FLEX", "starter": worst["name"], "candidate": best_bench["name"], "delta_pts": round(delta, 2)})

    # SUPERFLEX (QB/RB/WR/TE)
    for _ in range(cfg.superflex):
        s_pool = starters[starters["slot"] == "SFLEX"] if "slot" in starters.columns else starters[starters["pos"].isin(ELIGIBLE_SFLEX)]
        if s_pool.empty:
            break
        worst = s_pool.nsmallest(1, "proj_week").iloc[0]
        b_pool = bench[bench["pos"].isin(ELIGIBLE_SFLEX)]
        if b_pool.empty:
            break
        best_bench = b_pool.nlargest(1, "proj_week").iloc[0]
        delta = float(best_bench["proj_week"] - worst["proj_week"])
        if delta > 0.01:
            rows.append({"slot": "SFLEX", "starter": worst["name"], "candidate": best_bench["name"], "delta_pts": round(delta, 2)})

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("delta_pts", ascending=False).reset_index(drop=True)


# ---------------- FA Prep & Upgrades ----------------
def prepare_fa_pool(dp_market: pd.DataFrame, league_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DP market rows that are NOT rostered anywhere in this league,
    using exact name_key, last+first-initial, and fuzzy tokens (aligned with app.py).
    Output: ['name','pos','market_value']
    """
    if dp_market is None or dp_market.empty or league_df is None or league_df.empty:
        return pd.DataFrame(columns=["name","pos","market_value"])

    def norm_key(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

    def tok(s: str) -> set:
        s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in (s or "").lower())
        toks = [t for t in s.split() if t and t not in {"jr","sr","ii","iii","iv","v"}]
        return set(toks)

    def lf_key(name: str) -> str:
        parts = [p for p in (name or "").split() if p]
        if not parts: return ""
        last = parts[-1].lower()
        first_init = parts[0][0].lower() if parts[0] else ""
        return f"{last}_{first_init}"

    m = dp_market.copy()
    m = m[m["pos"].isin({"QB","RB","WR","TE","K"})].copy()
    m["name_key"] = norm_key(m["name"])
    m["lfk"] = m["name"].astype(str).apply(lf_key)

    roster = league_df[["name","pos"]].dropna().copy()
    roster["name_key"] = norm_key(roster["name"])
    roster["lfk"] = roster["name"].astype(str).apply(lf_key)
    r_exact = roster[["pos","name_key"]].drop_duplicates()
    r_lfk   = set(roster["lfk"].unique().tolist())

    # exact anti-join by (pos, name_key)
    left = m.merge(r_exact, on=["pos","name_key"], how="left", indicator=True)
    left = left[left["_merge"] == "left_only"].drop(columns=["_merge"])

    # fast lfk filter
    left = left[~left["lfk"].isin(r_lfk)]

    # fuzzy token guard (nickname/diacritics)
    rost_pos_tokens = {p: [tok(nk) for nk in grp["name_key"].unique()]
                       for p, grp in roster.groupby("pos")}

    def fuzzy_rostered(row, thresh=0.85):
        t = tok(row["name_key"]); pos = row["pos"]
        for tr in rost_pos_tokens.get(pos, []):
            inter = len(t & tr); uni = len(t | tr)
            if uni and (inter / uni) >= thresh:
                return True
        return False

    if not left.empty:
        left["_fuzzy"] = left.apply(fuzzy_rostered, axis=1)
        left = left[~left["_fuzzy"]].drop(columns=["_fuzzy"])

    return left[["name","pos","market_value"]].copy()




def fa_upgrade_suggestions(
    starters_df: pd.DataFrame,
    fa_df: pd.DataFrame,
    cfg: LineupConfig,
    min_delta: float = 0.50,
    max_results: int = 8
) -> pd.DataFrame:
    """
    For each lineup slot type, find FA(s) who beat your current worst starter by >= min_delta.
    Returns a concise table (empty-safe) sorted by delta_pts desc (up to max_results).
    Columns: ["slot","replace","add","proj_add","proj_replace","delta_pts","pos"]
    """
    cols = ["slot", "replace", "add", "proj_add", "proj_replace", "delta_pts", "pos"]
    if starters_df is None or starters_df.empty or fa_df is None or fa_df.empty:
        return pd.DataFrame(columns=cols)

    fa = fa_df.copy()
    if "proj_week" not in fa.columns:
        fa["proj_week"] = 0.0

    recs: List[Dict[str, object]] = []

    def add_reco(slot_label: str, elig: set):
        if slot_label in {"FLEX", "SFLEX"}:
            # find the worst starter among that slot’s eligibility
            s_pool = starters_df[starters_df["pos"].isin(elig)]
        else:
            # exact fixed slot when available; otherwise fallback to pos match
            if "slot" in starters_df.columns:
                s_pool = starters_df[starters_df["slot"] == slot_label]
                if s_pool.empty:
                    s_pool = starters_df[starters_df["pos"].isin(elig)]
            else:
                s_pool = starters_df[starters_df["pos"].isin(elig)]
        if s_pool.empty:
            return
        worst = s_pool.nsmallest(1, "proj_week").iloc[0]

        f_pool = fa[fa["pos"].isin(elig)]
        if f_pool.empty:
            return
        best = f_pool.nlargest(1, "proj_week").iloc[0]
        delta = float(best["proj_week"] - worst["proj_week"])
        if delta >= min_delta:
            recs.append({
                "slot": slot_label,
                "replace": worst.get("name"),
                "add": best.get("name"),
                "proj_add": round(float(best["proj_week"]), 2),
                "proj_replace": round(float(worst["proj_week"]), 2),
                "delta_pts": round(delta, 2),
                "pos": best.get("pos"),
            })

    # fixed positions
    for pos, n in [("QB", cfg.qb), ("RB", cfg.rb), ("WR", cfg.wr), ("TE", cfg.te), ("K", cfg.k)]:
        if n > 0:
            add_reco(pos, {pos})

    # FLEX / SUPERFLEX
    if cfg.flex > 0:
        add_reco("FLEX", ELIGIBLE_FLEX)
    if cfg.superflex > 0:
        add_reco("SFLEX", ELIGIBLE_SFLEX)

    if not recs:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(recs, columns=cols).sort_values("delta_pts", ascending=False)
    return out.head(max_results).reset_index(drop=True)
