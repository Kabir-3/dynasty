# app.py — Dynasty Buy/Sell Radar (Sleeper)
# Win-Now w/ weekly fallback + star-protection; Market Edge prime-age buy lows;
# Player lookup chart; FA Finder (roster-safe); saved league IDs.

import io, requests, datetime as dt
from typing import Optional, Iterable, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from math import exp
from collections import defaultdict
from lineup_optimizer import (
    LineupConfig,
    default_config,
    make_config,
    project_points_this_week,
    recommend_lineup_with_cfg,
    bench_swap_suggestions,
    prepare_fa_pool,
    fa_upgrade_suggestions,
)


from sleeper_pull import fetch_league_data
from value_engine import compute_true_value, attach_markets
from team_tools import quick_balance_score

# ---------- Action Plan helpers ----------

def _proj_variance_defaults():
    # conservative weekly SDs by position, if we can't learn from data
    return {"QB": 5.0, "RB": 4.5, "WR": 4.0, "TE": 3.5, "K": 3.0}

def learn_position_sd(weekly_df: pd.DataFrame) -> Dict[str, float]:
    if weekly_df is None or weekly_df.empty or "pos" not in weekly_df.columns:
        return _proj_variance_defaults()
    w = weekly_df.copy()
    w["points"] = pd.to_numeric(w["points"], errors="coerce").fillna(0.0)
    sd = w.groupby("pos")["points"].std(ddof=0).to_dict()
    base = _proj_variance_defaults()
    # blend learned with defaults
    return {p: float(np.nan_to_num(sd.get(p, base[p]))*0.7 + base[p]*0.3) for p in base}

def weakest_starters(starters_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    if starters_df is None or starters_df.empty:
        return starters_df.iloc[0:0]
    cols = [c for c in ["slot","name","pos","proj_week"] if c in starters_df.columns]
    return starters_df.sort_values("proj_week", ascending=True)[cols].head(k)

def top_fa_upgrades(starters_df, fa_mkt, lineup_cfg, pos_prior, min_delta=0.5, max_results=5):
    # score FA pool using same projection function (offseason-safe)
    fa_scored = project_points_this_week(
        fa_mkt.assign(ewma=np.nan, ppg=np.nan, trend=np.nan, games_played=np.nan),
        pos_prior=pos_prior
    )
    # cap crazy off-season fallbacks a bit
    no_signal = fa_scored[["ewma","ppg"]].fillna(0).sum(axis=1).eq(0)
    pos_cap = {"QB": 8.0, "RB": 8.0, "WR": 7.0, "TE": 6.0, "K": 8.0}
    def _cap_row(r):
        cap = pos_cap.get(str(r.get("pos")), 6.0)
        return min(float(r.get("proj_week", 0.0)) * 0.65, cap)
    fa_scored.loc[no_signal, "proj_week"] = fa_scored.loc[no_signal].apply(_cap_row, axis=1)

    return fa_upgrade_suggestions(starters_df, fa_scored, lineup_cfg, min_delta=min_delta, max_results=max_results)

def breakout_stashes(my_df: pd.DataFrame, weekly_df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """
    Simple stash heuristic: positive trend, EWMA > PPG, price percentile <= 0.60 (cheap),
    not already in top starters.
    """
    x = my_df.copy()
    if x.empty:
        return x.iloc[0:0]
    # ensure these exist
    for c in ["trend","ewma","ppg","price_pct"]:
        if c not in x.columns:
            x[c] = np.nan
    # fill safety
    x["price_pct"] = pd.to_numeric(x["price_pct"], errors="coerce").fillna(0.5)
    x["trend"] = pd.to_numeric(x["trend"], errors="coerce").fillna(0.0)
    x["ewma"] = pd.to_numeric(x["ewma"], errors="coerce").fillna(0.0)
    x["ppg"]  = pd.to_numeric(x["ppg"],  errors="coerce").fillna(0.0)

    cand = x[(x["trend"] > 0) & (x["ewma"] > x["ppg"]) & (x["price_pct"] <= 0.60)].copy()
    # upside score: trend + ewma surplus, break ties by lower price
    cand["stash_score"] = (cand["trend"]*0.6 + (cand["ewma"]-cand["ppg"])*0.4) - (cand["price_pct"]*0.2)
    keep = [c for c in ["name","pos","team","ppg","ewma","trend","price_pct","stash_score"] if c in cand.columns]
    return cand.sort_values("stash_score", ascending=False)[keep].head(limit)



# ---------------- App setup ----------------
st.set_page_config(page_title="Dynasty Buy/Sell Radar", layout="wide")
st.title("Dynasty Buy/Sell Radar (Sleeper)")

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _clear_prod_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

# Small helper to be compatible with older/newer pandas groupby.apply behavior
def group_apply(gb, func):
    """
    Call GroupBy.apply(func) in a way that works across pandas versions
    where include_groups was added (FutureWarning otherwise).
    """
    try:
        return gb.apply(func, include_groups=False)
    except TypeError:
        # pandas <= 2.2 (no include_groups kwarg)
        return gb.apply(func)

# ---------------- Session defaults ----------------
if "league_id" not in st.session_state:
    st.session_state.league_id = "1195252934627844096"
if "league_history" not in st.session_state:
    st.session_state.league_history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("League")
    league_id_input = st.text_input("Sleeper League ID", value=st.session_state["league_id"])
    c_loadR, c_rem = st.columns([1, 1])
    with c_loadR:
        if st.button("Load league", type="primary", use_container_width=True):
            st.session_state["league_id"] = league_id_input.strip()
            st.cache_data.clear()
            _safe_rerun()
    with c_rem:
        if st.button("Remember current", use_container_width=True):
            lid = league_id_input.strip()
            if lid and lid not in st.session_state.league_history:
                st.session_state.league_history = [lid] + st.session_state.league_history[:9]
                st.success(f"Saved {lid}")

    st.markdown("---")
    st.caption("Saved old leagues")
    row = st.columns([3, 1])
    with row[0]:
        old_id_to_add = st.text_input(
            "Add old Sleeper League ID",
            key="add_old_id",
            placeholder="e.g. 987654321012345678",
            label_visibility="collapsed",
        )
    with row[1]:
        if st.button("Add", key="add_old_btn", use_container_width=True):
            lid = (old_id_to_add or "").strip()
            if lid and lid not in st.session_state.league_history:
                st.session_state.league_history = [lid] + st.session_state.league_history[:9]
                _safe_rerun()

    if st.session_state.league_history:
        saved_sel = st.selectbox("Saved IDs", st.session_state.league_history, key="saved_ids_select")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Load saved", key="load_saved_btn", use_container_width=True):
                st.session_state["league_id"] = saved_sel
                st.cache_data.clear()
                _safe_rerun()
        with c2:
            if st.button("Forget", key="forget_saved_btn", use_container_width=True):
                st.session_state.league_history = [x for x in st.session_state.league_history if x != saved_sel]
                _safe_rerun()
        with c3:
            if st.button("Clear all", key="clear_all_saved_btn", use_container_width=True):
                st.session_state.league_history = []
                _safe_rerun()
    else:
        st.caption("No saved IDs yet — add one above.")

    st.subheader("Data refresh")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Refresh market"):
            st.cache_data.clear()
            _safe_rerun()
    with c2:
        if st.button("Refresh production"):
            st.cache_data.clear()
            _safe_rerun()

    st.subheader("Model knobs (market)")
    if "age_weight" not in st.session_state:
        st.session_state.age_weight = 1.0
    if "youth_bonus" not in st.session_state:
        st.session_state.youth_bonus = 3.0
    if "age_cap" not in st.session_state:
        st.session_state.age_cap = 8.0
    age_weight = st.slider("Age weight", 0.0, 1.5, st.session_state.age_weight, 0.05, key="age_weight")
    youth_bonus = st.slider("Youth bonus (≤24 RB/WR/TE)", 0.0, 6.0, st.session_state.youth_bonus, 0.5, key="youth_bonus")
    age_cap = st.slider("Age impact cap (abs)", 2.0, 12.0, st.session_state.age_cap, 0.5, key="age_cap")

# ---------------- Top controls ----------------
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    scoring_choice = st.selectbox("Scoring", ["PPR", "Half", "Standard"], index=0)
with c2:
    superflex = st.checkbox("Superflex", False)
    te_premium = st.checkbox("TE Premium (1.5)", False)
with c3:
    w_dp = st.slider("Weight: DynastyProcess", 0.0, 1.0, 0.7, 0.05)
    w_fp = st.slider("Weight: ADP fallback", 0.0, 1.0, 0.3, 0.05)
ppr = scoring_choice == "PPR"

mode = st.radio(
    "Evaluation mode",
    ["Market Edge", "Win-Now (auto from Sleeper)", "Lineup Optimizer"],
    horizontal=True,
)



# ---------------- External sources ----------------
DP_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values-players.csv"
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_STATE = "https://api.sleeper.app/v1/state/nfl"
SLEEPER_STATS_WEEK = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"

def normalize_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()

def tokens_for(s: str) -> List[str]:
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in (s or "").lower())
    toks = [t for t in s.split() if t and t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return [t for t in toks if not (len(t) == 1 and t.isalpha())]

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return 0.0 if not sa or not sb else len(sa & sb) / len(sa | sb)

def build_fuzzy_map(prod_keys: pd.DataFrame, roster_keys: pd.DataFrame, threshold: float = 0.85) -> Dict[str, Dict[str, str]]:
    mapping_by_pos: Dict[str, Dict[str, str]] = {}
    roster_by_pos: Dict[str, List[Tuple[str, List[str]]]] = {}
    for pos, grp in roster_keys.groupby("pos"):
        roster_by_pos[pos] = [(nk, tokens_for(nk)) for nk in grp["name_key"].unique()]
    for pos, grp in prod_keys.groupby("pos"):
        target = roster_by_pos.get(pos, [])
        mp: Dict[str, str] = {}
        for nk in grp["name_key"].unique():
            src = tokens_for(nk)
            best_key, best_score = None, 0.0
            for nk_tgt, tgt in target:
                s = jaccard(src, tgt)
                if s > best_score:
                    best_key, best_score = nk_tgt, s
            if best_key and best_score >= threshold:
                mp[nk] = best_key
        if mp:
            mapping_by_pos[pos] = mp
    return mapping_by_pos

def apply_mapping(name_key: str, pos: str, mapping_by_pos: Dict[str, Dict[str, str]]) -> str:
    return mapping_by_pos.get(pos, {}).get(name_key, name_key)

@st.cache_data(ttl=24 * 60 * 60)
def fetch_dp_values() -> pd.DataFrame:
    r = requests.get(DP_URL, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=24 * 60 * 60)
def fetch_sleeper_adp() -> Optional[pd.DataFrame]:
    try:
        players = requests.get(SLEEPER_PLAYERS, timeout=30).json()
        adp = requests.get("https://api.sleeper.app/v1/adp/nfl/2024?type=ppr", timeout=30).json()
        rows = []
        for row in adp:
            pid = row.get("player_id")
            p = players.get(pid) if isinstance(players, dict) else None
            if not p:
                continue
            name = p.get("full_name") or (p.get("first_name", "") + " " + p.get("last_name", "")).strip()
            rows.append({"name": name, "pos": p.get("position"), "Rank": row.get("adp")})
        df = pd.DataFrame(rows)
        return None if df.empty else df
    except Exception:
        return None

def dp_to_market(dp: pd.DataFrame) -> pd.DataFrame:
    df = dp.copy()
    name_candidates = ["name", "player_name", "player", "Player", "full_name", "Player Name"]
    pos_candidates = ["pos", "position", "Position"]

    def pick(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    name_col = pick(name_candidates)
    pos_col = pick(pos_candidates)
    if name_col is None or pos_col is None:
        raise RuntimeError(f"Missing name/pos in DP CSV. Saw: {list(df.columns)}")
    value_cols = [c for c in df.columns if ("_1qb" in c.lower()) or ("value" in c.lower())]
    if not value_cols:
        raise RuntimeError(f"No value columns in DP CSV. Saw: {list(df.columns)}")
    mv = df[value_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return pd.DataFrame({"name": df[name_col].astype(str), "pos": df[pos_col].astype(str), "market_value": mv})

@st.cache_data(ttl=2 * 60 * 60)
def get_sleeper_state() -> dict:
    try:
        return requests.get(SLEEPER_STATE, timeout=15).json()
    except Exception:
        return {}

def default_season_and_week() -> Tuple[int, int]:
    # timezone-aware to avoid deprecation warning
    now = dt.datetime.now(dt.timezone.utc)
    state = get_sleeper_state()
    season = int(state.get("season") or now.year)
    week = int(state.get("week") or 1)
    return season, max(1, week)

@st.cache_data(ttl=24 * 60 * 60)
def get_players_map() -> Dict[str, dict]:
    try:
        return requests.get(SLEEPER_PLAYERS, timeout=30).json()
    except Exception:
        return {}

@st.cache_data(ttl=24 * 60 * 60)
def get_league_meta(league_id: str) -> dict:
    try:
        r = requests.get(f"https://api.sleeper.app/v1/league/{league_id}", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# ---------- Sleeper matchups-based production (robust to deprecated /stats) ----------
MATCHUPS = "https://api.sleeper.app/v1/league/{league_id}/matchups/{week}"

# Map past seasons to their league_id (fill these if you want to browse old seasons)
PAST_LEAGUES: Dict[int, str] = {
    # 2024: "YOUR_2024_LEAGUE_ID",
    # 2023: "YOUR_2023_LEAGUE_ID",
}

@st.cache_data(ttl=6 * 60 * 60)
def fetch_matchups_week(league_id: str, week: int) -> pd.DataFrame:
    try:
        resp = requests.get(MATCHUPS.format(league_id=str(league_id), week=int(week)), timeout=30)
        data = resp.json()
    except Exception:
        data = None

    if not data or (isinstance(data, list) and len(data) == 0):
        return pd.DataFrame(columns=["player_id", "name", "pos", "week", "points", "name_key"])

    players = get_players_map()
    rows = []
    for m in data if isinstance(data, list) else []:
        pp = m.get("players_points") or {}
        for pid, pts in pp.items():
            pid = str(pid)
            p = players.get(pid) if isinstance(players, dict) else None
            name = (p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()) if p else ""
            pos = p.get("position") if p else None
            if not name or not pos:
                continue
            rows.append(
                {"player_id": pid, "name": name, "pos": pos, "week": int(week), "points": float(pts or 0.0)}
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["player_id", "name", "pos", "week", "points", "name_key"])
    out["name_key"] = normalize_name(out["name"])
    return out

@st.cache_data(ttl=6 * 60 * 60)
def fetch_points_season_to_date_via_matchups(league_id: str, through_week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weeks = list(range(1, max(1, int(through_week)) + 1))
    frames = [fetch_matchups_week(league_id, w) for w in weeks]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    weekly = pd.concat(frames, ignore_index=True)
    if "name_key" not in weekly.columns:
        weekly["name_key"] = normalize_name(weekly["name"])
    totals = (
        weekly.groupby(["player_id", "name", "pos", "name_key"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "points"})
    )
    return totals, weekly

# -------- Weekly fetch + fallback (kept for legacy; Win-Now now uses matchups) --------
@st.cache_data(ttl=6 * 60 * 60)
def fetch_week_df(season: int, week: int) -> pd.DataFrame:
    players = get_players_map()
    try:
        url = SLEEPER_STATS_WEEK.format(season=season, week=week)
        wkstats = requests.get(url, timeout=30).json()
    except Exception:
        wkstats = None

    def gv(row: dict, key, default=0.0):
        v = row.get(key)
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    def calc_ppr(row: dict) -> float:
        if row is None:
            return 0.0
        if "pts_ppr" in row and row["pts_ppr"] is not None:
            try:
                return float(row["pts_ppr"])
            except Exception:
                pass
        return (
            0.04 * gv(row, "pass_yd")
            + 4 * gv(row, "pass_td")
            - 2 * gv(row, "pass_int")
            + 0.1 * gv(row, "rush_yd")
            + 6 * gv(row, "rush_td")
            + 0.1 * gv(row, "rec_yd")
            + 6 * gv(row, "rec_td")
            + 1 * gv(row, "rec")
            - 2 * (gv(row, "fum_lost", 0.0) or gv(row, "fumbles_lost", 0.0))
        )

    rows = []
    iterable = wkstats if isinstance(wkstats, list) else (wkstats.values() if isinstance(wkstats, dict) else [])
    for row in iterable:
        pid = str(row.get("player_id") or row.get("player") or "")
        if not pid:
            continue
        p = players.get(pid) if isinstance(players, dict) else None
        name = (
            (p.get("full_name") or (p.get("first_name", "") + " " + p.get("last_name", "")).strip())
            if p
            else (row.get("full_name") or row.get("player_name") or "").strip()
        )
        pos = p.get("position") if p else row.get("position")
        if not name or not pos:
            continue
        rows.append({"player_id": pid, "name": name, "pos": pos, "week": int(week), "points": float(calc_ppr(row))})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["player_id", "name", "pos", "week", "points", "name_key"])
    out["name_key"] = normalize_name(out["name"])
    return out

@st.cache_data(ttl=6 * 60 * 60)
def fetch_sleeper_points_ppr(season: int, through_week: int) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (totals, weekly, meta), with fallback to previous season if empty."""
    def _one(sea: int, wk: int):
        weeks = list(range(1, max(1, int(wk)) + 1))
        wk_frames = [fetch_week_df(int(sea), w) for w in weeks]
        valid = [w for w in wk_frames if w is not None and not w.empty]
        if not valid:
            return pd.DataFrame(), pd.DataFrame()
        weekly = pd.concat(valid, ignore_index=True)
        totals = weekly.groupby(["player_id", "name", "pos", "name_key"], as_index=False)["points"].sum()
        totals.rename(columns={"points": "points"}, inplace=True)
        return totals, weekly

    t, w = _one(season, through_week)
    meta = {"season_used": season, "weeks_used": list(range(1, through_week + 1)), "source": "direct"}
    if w.empty and season >= 2019:
        t, w = _one(season - 1, 18)
        meta = {"season_used": season - 1, "weeks_used": list(range(1, 19)), "source": "fallback"}
    return t, w, meta

# ---------------- Win-Now controls ----------------
weekly_df: Optional[pd.DataFrame] = None
meta_fetch: Dict[str, object] = {}
if mode.startswith("Win"):
    st.subheader("Win-Now production (auto from Sleeper)")
    league_meta = get_league_meta(st.session_state["league_id"])
    league_season_default = int(league_meta.get("season") or default_season_and_week()[0])
    now_season, now_week = default_season_and_week()

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        season = st.number_input(
            "Season",
            2018,
            league_season_default + 1,
            league_season_default,
            1,
            key="wn_season",
            on_change=_clear_prod_cache,
        )
    with cB:
        wk_def = 18 if season < now_season else now_week
        through_week = st.number_input(
            "Through week",
            1,
            23,
            int(wk_def),
            1,
            key="wn_through",
            on_change=_clear_prod_cache,
        )
    with cC:
        st.caption("Production cache auto-refreshes every 6 hours.")

    # >>>>>>>>>> MATCHUPS-BASED FETCH (replaces old stats fetch) <<<<<<<<<<
    with st.spinner(f"Fetching production from league matchups, weeks ≤ {through_week}…"):
        league_for = PAST_LEAGUES.get(int(season), st.session_state["league_id"])
        prod_df, weekly_df = fetch_points_season_to_date_via_matchups(league_for, int(through_week))
        meta_fetch = {
            "season_used": int(season),
            "weeks_used": list(range(1, int(through_week) + 1)),
            "source": "matchups",
        }
        st.caption(
            f"Source: matchups | league_id={league_for} | through_week={through_week} | "
            f"weekly_rows={0 if weekly_df is None else len(weekly_df)}"
        )

    if weekly_df is None or weekly_df.empty:
        st.warning("No weekly stats for that season/week yet. Showing last season as a fallback.")
    else:
        st.info(f"Using season **{meta_fetch['season_used']}** ({meta_fetch['source']}).")

# ---------------- Mode-specific weekly production fetch ----------------
# Win-Now already fetched weekly_df above using the user’s season/week controls.
# Only fetch here for Lineup Optimizer; leave Market Edge with no weekly_df.
if mode == "Lineup Optimizer":
    weekly_df = None
    meta_fetch = {}
    league_meta = get_league_meta(st.session_state["league_id"])
    now_season, now_week = default_season_and_week()
    season = int(league_meta.get("season") or now_season)
    through_week = int(now_week)
    with st.spinner(f"Fetching production (matchups) for lineup optimizer…"):
        league_for = st.session_state["league_id"]
        _, weekly_df = fetch_points_season_to_date_via_matchups(league_for, through_week)
    meta_fetch = {
        "season_used": season,
        "weeks_used": list(range(1, through_week + 1)),
        "source": "matchups",
    }
elif mode == "Market Edge":
    weekly_df = None
    meta_fetch = {}
# (when mode starts with "Win", DO NOT touch weekly_df/meta_fetch here — the earlier block controls it)


# ---------------- Load league + markets ----------------
@st.cache_data(ttl=24 * 60 * 60)
def load_all(
    league_id,
    superflex,
    te_premium,
    ppr,
    w_dp,
    w_fp,
    age_weight,
    youth_bonus,
    age_cap,
    weekly_df=None,
):
    roster = fetch_league_data(league_id)  # includes player_id
    roster = compute_true_value(
        roster,
        superflex=superflex,
        te_premium=te_premium,
        ppr=ppr,
        age_weight=age_weight,
        youth_bonus=youth_bonus,
        age_cap=age_cap,
        weekly_df=weekly_df,
    )
    dp_mkt = dp_to_market(fetch_dp_values())
    fp_like = fetch_sleeper_adp()
    roster = attach_markets(roster, dp_df=dp_mkt, fp_df=fp_like, w_dp=w_dp, w_fp=w_fp)
    return roster, dp_mkt, fp_like is not None

try:
    df, dp_mkt, has_fp_like = load_all(
        st.session_state["league_id"],
        superflex,
        te_premium,
        ppr,
        w_dp,
        w_fp,
        age_weight,
        youth_bonus,
        age_cap,
        weekly_df=weekly_df,
    )
    if df is None or df.empty:
        st.error("Couldn’t load that League ID (404/private/empty).")
        st.stop()
    lid = st.session_state["league_id"].strip()
    if lid and lid not in st.session_state.league_history:
        st.session_state.league_history = [lid] + st.session_state.league_history[:9]
except Exception:
    st.error("Couldn’t load that League ID (network or invalid). Try again.")
    st.stop()

# ---------------- Merge weekly -> roster (player_id first, then fuzzy) ----------------
if "name_key" not in df.columns:
    df["name_key"] = normalize_name(df["name"])

debug_direct, debug_after = 0, 0
weekly_rows = 0
if weekly_df is not None and not weekly_df.empty:
    wk = weekly_df.copy()
    weekly_rows = len(wk)
    # exact via player_id
    if "player_id" in df.columns and "player_id" in wk.columns:
        by_pid = wk.groupby("player_id", as_index=False)["points"].sum().rename(columns={"points": "points_total"})
        df = df.merge(by_pid, on="player_id", how="left")
    if "points_total" in df.columns:
        debug_direct = int(df["points_total"].notna().sum())
    # fuzzy fill for holes
    need_map_mask = df["points_total"].isna() if "points_total" in df.columns else pd.Series(True, index=df.index)
    if need_map_mask.any():
        prod_keys = wk[["name_key", "pos"]].drop_duplicates()
        rost_keys = df.loc[need_map_mask, ["name_key", "pos"]].drop_duplicates()
        mapping_by_pos = build_fuzzy_map(prod_keys, rost_keys, threshold=0.85)
        wk_mapped = wk.copy()
        wk_mapped["name_key"] = wk_mapped.apply(
            lambda r: apply_mapping(r["name_key"], r["pos"], mapping_by_pos), axis=1
        )
        by_key = (
            wk_mapped.groupby(["name_key", "pos"], as_index=False)["points"]
            .sum()
            .rename(columns={"points": "points_total_fuzzy"})
        )
        df = df.merge(by_key, on=["name_key", "pos"], how="left")
        if "points_total" in df.columns:
            df["points_total"] = df["points_total"].fillna(df["points_total_fuzzy"])
        else:
            df["points_total"] = df["points_total_fuzzy"]
        df.drop(columns=["points_total_fuzzy"], inplace=True)
        weekly_df = wk_mapped
    else:
        weekly_df = wk
    debug_after = int(df["points_total"].notna().sum())
    df["z_prod"] = df.groupby("pos")["points_total"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
else:
    if "points_total" not in df.columns:
        df["points_total"] = np.nan
    if "z_prod" not in df.columns:
        df["z_prod"] = np.nan

st.caption(
    f"Sources: DP=Yes, ADP={'Yes' if has_fp_like else 'No'} | "
    f"Blend: DP {w_dp:.2f} / ADP {w_fp:.2f} | "
    f"Age wt {age_weight:.2f}, Youth {youth_bonus:.1f}, Cap ±{age_cap:.1f}"
)

# ---------------- Helpers ----------------
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.0, index=s.index, dtype=float)
    mu, sd = s.mean(), s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s - mu) / sd

# ---------- Playoff Odds helpers ----------
@st.cache_data(ttl=2*60*60)
def _fetch_owner_maps(league_id: str):
    """Map roster_id -> display_name and back. Used to translate Sleeper matchups to team names."""
    try:
        users = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users", timeout=20).json()
        rosters = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters", timeout=20).json()
    except Exception:
        return {}, {}, {}

    uid_to_name = {u.get("user_id"): (u.get("display_name") or f"user_{u.get('user_id')}") for u in users or []}
    rid_to_uid  = {r.get("roster_id"): r.get("owner_id") for r in rosters or []}
    rid_to_name = {rid: uid_to_name.get(uid, f"owner_{uid}") for rid, uid in rid_to_uid.items()}

    # name -> roster_id (useful for resolving)
    name_to_rid = {}
    for rid, uid in rid_to_uid.items():
        nm = uid_to_name.get(uid)
        if nm:
            name_to_rid[nm] = rid
    return rid_to_name, name_to_rid, rosters

@st.cache_data(ttl=2*60*60)
def _fetch_schedule_pairs(league_id: str, start_week: int, end_week: int):
    """
    Returns: {week: [(nameA, nameB), ...]}, using current season league schedule via /matchups/{week}.
    """
    rid_to_name, _, _ = _fetch_owner_maps(league_id)
    schedule = {}
    for wk in range(int(start_week), int(end_week) + 1):
        try:
            data = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{wk}", timeout=30).json()
        except Exception:
            data = None
        if not data:
            schedule[wk] = []
            continue
        # group by matchup_id then pair the two rosters
        buckets = defaultdict(list)
        for row in data:
            buckets[row.get("matchup_id")].append(row)
        pairs = []
        for mid, rows in buckets.items():
            names = []
            for r in rows:
                rid = r.get("roster_id")
                nm = rid_to_name.get(rid)
                if nm:
                    names.append(nm)
            if len(names) == 2:
                pairs.append((names[0], names[1]))
        schedule[wk] = pairs
    return schedule

def _team_ratings_from_df(df: pd.DataFrame, metric: str = "z_prod") -> Dict[str, float]:
    """Use your existing strength_table totals as a rating baseline."""
    tbl = strength_table(df, metric)
    # Normalize to ~N(0,1) just in case
    r = tbl["TOTAL"]
    mu, sd = r.mean(), r.std(ddof=0) or 1.0
    return {team: float((val - mu) / sd) for team, val in r.items()}

def _win_prob(r_a: float, r_b: float, k: float = 0.9) -> float:
    """Logistic on rating diff → win probability for A vs B."""
    return 1.0 / (1.0 + exp(-k * (r_a - r_b)))

def simulate_playoff_odds(
    df: pd.DataFrame,
    league_id: str,
    start_week: int,
    end_week: int,
    playoff_teams: int,
    metric_for_ratings: str = "z_prod",
    sims: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo using schedule + rating-based win probs.
    Returns: {display_name: playoff_odds_float_between_0_and_1}
    """
    import random
    random.seed(int(seed))

    teams = sorted(df["display_name"].dropna().unique().tolist())
    if not teams:
        return {}
    ratings = _team_ratings_from_df(df, metric_for_ratings)
    schedule = _fetch_schedule_pairs(league_id, start_week, end_week)

    counts = {t: 0 for t in teams}     # # of simulations where team makes playoffs
    for _ in range(int(sims)):
        wins = {t: 0 for t in teams}
        for wk in range(int(start_week), int(end_week) + 1):
            for (a, b) in schedule.get(wk, []):
                if (a not in ratings) or (b not in ratings):
                    continue
                p = _win_prob(ratings[a], ratings[b])
                if random.random() < p:
                    wins[a] += 1
                else:
                    wins[b] += 1
        # top-N by wins make playoffs (random tiebreaker)
        ordered = sorted(teams, key=lambda t: (wins[t], random.random()), reverse=True)
        qualifiers = set(ordered[:int(playoff_teams)])
        for t in qualifiers:
            counts[t] += 1

    return {t: counts[t] / float(sims) for t in teams}

def compute_playoff_window(meta: dict):
    """Figure out remaining regular-season weeks from league meta."""
    now_season, now_week = default_season_and_week()
    settings = (meta or {}).get("settings") or {}
    start = max(int(now_week), 1)
    # Sleeper commonly has 'playoff_week_start'
    end = int(settings.get("playoff_week_start") or (start + 3))
    playoff_teams = int(settings.get("playoff_teams") or 6)
    # clamp
    end = max(end - 1, start)  # simulate regular season up to the week before playoffs
    return start, end, playoff_teams



def build_weekly_action_plan(swaps: pd.DataFrame, fa_up: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []

    # ---- Bench swaps (strict schema expected) ----
    if isinstance(swaps, pd.DataFrame) and not swaps.empty:
        need = {"slot", "starter", "candidate", "delta_pts"}
        if need.issubset(set(swaps.columns)):
            for _, r in swaps.sort_values("delta_pts", ascending=False).iterrows():
                rows.append({
                    "Priority": "Start/Sit",
                    "Action": f"Swap in {r['candidate']} for {r['starter']} at {r['slot']}",
                    "Delta (pts)": round(float(r["delta_pts"]), 2),
                })

    # ---- FA upgrades (be flexible on column names) ----
    if isinstance(fa_up, pd.DataFrame) and not fa_up.empty:
        # Try multiple possible column names to be robust across versions
        cand_col  = next((c for c in ["candidate", "add", "player", "name"]        if c in fa_up.columns), None)
        repl_col  = next((c for c in ["replaced", "starter", "over", "drop"]       if c in fa_up.columns), None)
        slot_col  = next((c for c in ["slot", "pos", "position"]                   if c in fa_up.columns), None)
        delta_col = next((c for c in ["delta_pts", "delta", "gain", "improvement"] if c in fa_up.columns), None)

        if all([cand_col, repl_col, slot_col, delta_col]):
            for _, r in fa_up.sort_values(delta_col, ascending=False).iterrows():
                rows.append({
                    "Priority": "FA Upgrade",
                    "Action": f"Add {r[cand_col]} and start over {r[repl_col]} at {r[slot_col]}",
                    "Delta (pts)": round(float(r[delta_col]), 2),
                })

    if not rows:
        return pd.DataFrame([{"Priority": "Info", "Action": "No actionable moves this week", "Delta (pts)": 0.0}])

    out = pd.DataFrame(rows).sort_values(["Delta (pts)", "Priority"], ascending=[False, True], ignore_index=True)
    return out


def strength_table(df_all: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    x = df_all.copy()
    if metric_col == "z_mkt":
        x["z_col"] = x.groupby("pos")["market_value"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    elif metric_col == "z_prod":
        x["z_col"] = x["z_prod"].fillna(0.0)
    elif metric_col == "z_true":
        x["z_col"] = x.groupby("pos")["true_value"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    else:
        x["z_col"] = x.groupby("pos")["market_value"].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    pivot = x.pivot_table(index="display_name", columns="pos", values="z_col", aggfunc="mean", fill_value=0.0)
    for col in ["QB", "RB", "WR", "TE", "K"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot = pivot[["QB", "RB", "WR", "TE", "K"]]
    pivot["TOTAL"] = pivot[["QB", "RB", "WR", "TE", "K"]].sum(axis=1)
    return pivot.sort_values("TOTAL", ascending=False).round(4)


from collections import Counter

# -------- Robust league roster index + FA filter (shared) --------
def _tokens_loose(s: str) -> set:
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in (s or "").lower())
    toks = [t for t in s.split() if t and t not in {"jr","sr","ii","iii","iv","v"}]
    return set(toks)

def _lf_key(name: str) -> str:
    # last-name + first-initial, e.g. "brown_m" for Marquise Brown
    parts = [p for p in (name or "").strip().split() if p]
    if not parts: 
        return ""
    last = parts[-1].lower()
    first_init = parts[0][0].lower() if parts[0] else ""
    return f"{last}_{first_init}"

def build_league_roster_index(df_all: pd.DataFrame):
    """Return a dict by pos with exact name_keys, fuzzy token sets, and last+firstInitial keys."""
    roster = df_all[["name","pos"]].dropna().copy()
    roster["name_key"] = normalize_name(roster["name"])
    roster["lfk"] = roster["name"].astype(str).apply(_lf_key)
    out = {}
    for p, grp in roster.groupby("pos"):
        exact = set(grp["name_key"].unique().tolist())
        lfk = set(grp["lfk"].unique().tolist())
        fuzz = [_tokens_loose(nk) for nk in grp["name_key"].unique()]
        out[p] = {"exact": exact, "lfk": lfk, "fuzz": fuzz}
    return out

def filter_unrostered(dp_market: pd.DataFrame, roster_index: dict, fuzzy_thresh: float = 0.80) -> pd.DataFrame:
    """dp_market: columns name,pos,market_value. Removes anyone who looks rostered in this league."""
    if dp_market is None or dp_market.empty:
        return pd.DataFrame(columns=["name","pos","market_value"])
    m = dp_market.copy()
    m = m[m["pos"].isin(["QB","RB","WR","TE","K"])].copy()
    m["name_key"] = normalize_name(m["name"])
    m["lfk"] = m["name"].astype(str).apply(_lf_key)

    def is_rostered(row):
        pos = row["pos"]
        nk = row["name_key"]
        lfk = row["lfk"]
        idx = roster_index.get(pos, {})
        # exact name_key or last+firstInitial
        if nk in idx.get("exact", set()) or lfk in idx.get("lfk", set()):
            return True
        # fuzzy tokens (to catch nicknames/accents)
        toks = _tokens_loose(nk)
        for tset in idx.get("fuzz", []):
            if not toks or not tset:
                continue
            inter = len(toks & tset); uni = len(toks | tset)
            if uni and (inter / uni) >= fuzzy_thresh:
                return True
        return False

    mask = m.apply(is_rostered, axis=1)
    return m[~mask].drop(columns=["name_key","lfk"])



def detect_lineup_config_from_sleeper(league_id: str) -> LineupConfig:
    meta = get_league_meta(league_id) or {}
    rpos = meta.get("roster_positions") or []   # e.g., ["QB","RB","RB","WR","WR","TE","FLEX","SUPER_FLEX","K","BN",...]
    c = Counter(rpos)

    qb = c.get("QB", 1)
    rb = c.get("RB", 2)
    wr = c.get("WR", 2)
    te = c.get("TE", 1)
    k  = c.get("K", 0)
    flex = c.get("FLEX", 0)          # RB/WR/TE
    sflex = c.get("SUPER_FLEX", 0)   # QB/RB/WR/TE

    ss = meta.get("scoring_settings") or {}
    rec = float(ss.get("rec", 0.0) or 0.0)
    rec_te = float(ss.get("rec_te", rec) or rec)
    te_prem = rec_te > rec + 1e-9

    return make_config(qb, rb, wr, te, flex, sflex, k, te_prem)

def canonicalize_alias(nk: str) -> str:
    if not nk:
        return nk
    toks = nk.split()
    s = set(toks)
    if "brown" in s and "hollywood" in s:
        toks = [t for t in toks if t != "hollywood"]
        if "marquise" not in toks:
            toks = ["marquise"] + toks
    return " ".join(toks)


def build_roster_safe_fa_pool(dp_market: pd.DataFrame, league_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DP market rows that are NOT rostered anywhere in this league,
    using exact + alias-canonicalized keys and a fuzzy guard.
    Output: ['name','pos','market_value']  (empty-safe)
    """
    if dp_market is None or dp_market.empty or league_df is None or league_df.empty:
        return pd.DataFrame(columns=["name","pos","market_value"])

    # 1) DP subset + normalized+canonicalized key
    m = dp_market.copy()
    m = m[m["pos"].isin(["QB","RB","WR","TE","K"])].copy()
    m["name_key"] = normalize_name(m["name"]).apply(canonicalize_alias)

    # 2) league-wide roster keys (normalized+canonicalized)
    roster_keys = league_df[["name","pos"]].dropna().copy()
    roster_keys["name_key"] = normalize_name(roster_keys["name"]).apply(canonicalize_alias)
    roster_keys = roster_keys[["pos","name_key"]].drop_duplicates()

    # 3) exact anti-join by (pos, name_key)
    exact = m.merge(roster_keys, on=["pos","name_key"], how="left", indicator=True)
    exact = exact[exact["_merge"] == "left_only"].drop(columns=["_merge"])

    # 3.5) unique last-name + position guard to block alias leaks (e.g., Hollywood/Marquise)
    def _last_from_key(nk: str) -> str:
        parts = [p for p in (nk or "").split() if p]
        return parts[-1] if parts else ""

    _roster_last_counts = (
        league_df[["name","pos"]].dropna().assign(
            name_key=lambda d: normalize_name(d["name"]).apply(canonicalize_alias),
            last=lambda d: d["name_key"].apply(_last_from_key)
        ).groupby(["pos","last"]).size().rename("ct").reset_index()
    )
    _unique_last = set(_roster_last_counts.loc[_roster_last_counts["ct"] == 1, ["pos","last"]]
                       .itertuples(index=False, name=None))

    if not exact.empty:
        exact = exact.assign(__last=exact["name_key"].apply(_last_from_key))
        mask_conflict = exact.apply(lambda r: (r["pos"], r["__last"]) in _unique_last, axis=1)
        exact = exact.loc[~mask_conflict].drop(columns=["__last"])

    # 4) fuzzy guard — same tokens/jaccard as Market Edge
    def jaccard_sets(a: set, b: set) -> float:
        return 0.0 if not a or not b else len(a & b) / len(a | b)

    rost_pos_map: Dict[str, List[Tuple[str, set]]] = {}
    for p, grp in roster_keys.groupby("pos"):
        rost_pos_map[p] = [(nk, set(tokens_for(nk))) for nk in grp["name_key"].unique()]

    def looks_rostered_fuzzy(row, thresh=0.85):
        p = row["pos"]; nk = row["name_key"]
        toks = set(tokens_for(nk))
        for _, toks_r in rost_pos_map.get(p, []):
            if jaccard_sets(toks, toks_r) >= thresh:
                return True
        return False

    if not exact.empty:
        exact["_is_rostered_fuzzy"] = exact.apply(looks_rostered_fuzzy, axis=1)
        fa_mkt = exact[~exact["_is_rostered_fuzzy"]].drop(columns=["_is_rostered_fuzzy"])
    else:
        fa_mkt = exact.copy()

    # Keep only the columns we actually need downstream
    return fa_mkt[["name","pos","market_value"]].copy()



def why_player(row: pd.Series) -> str:
    bits = []
    if "edge_z_adj" in row and pd.notna(row["edge_z_adj"]):
        bits.append(f"Model vs Market: {row['edge_z_adj']:+.2f} z")
    if "edge" in row and pd.notna(row["edge"]):
        bits.append(f"Raw gap: {row['edge']:+.1f}")
    if "points_total" in row and pd.notna(row["points_total"]):
        bits.append(f"Season PPR: {row['points_total']:.1f}")
    if "ppg" in row and pd.notna(row["ppg"]):
        bits.append(f"PPG: {row['ppg']:.2f}")
    if "ewma" in row and pd.notna(row["ewma"]):
        bits.append(f"Recent form (EWMA): {row['ewma']:.1f}")
    if "trend" in row and pd.notna(row["trend"]):
        bits.append(f"Trend vs avg: {row['trend']:+.1f}")
    if "z_consis" in row and pd.notna(row["z_consis"]):
        bits.append(f"Consistency: {row['z_consis']:+.2f} z")
    return " • ".join(bits) if bits else "No extra notes yet."

def friendly_df(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    rename_map = {
        "display_name": "Team",
        "name": "Player",
        "pos": "Pos",
        "age": "Age",
        "team": "NFL",
        "true_value": "Model Value",
        "market_value": "Market Value",
        "edge": "Raw Diff",
        "edge_z_adj": "Model vs Market (z)",
        "WinNowScore": "Win-Now Score",
        "z_prod": "Prod z",
        "points_total": "Season PPR",
        "ppg": "PPG",
        "ewma": "EWMA",
        "trend": "Trend",
    }
    cols = [c for c in rename_map if c in out.columns] + ([score_col] if score_col not in rename_map else [])
    out = out[cols].rename(columns=rename_map)
    num_cols = [
        c
        for c in [
            "Age",
            "Model Value",
            "Market Value",
            "Raw Diff",
            "Model vs Market (z)",
            "Win-Now Score",
            "Prod z",
            "Season PPR",
            "PPG",
            "EWMA",
            "Trend",
        ]
        if c in out.columns
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out

def is_prime_age(row) -> bool:
    a = row.get("age", np.nan)
    p = row.get("pos")
    if pd.isna(a):
        return False
    a = float(a)
    if p == "RB":
        return 23 <= a <= 26
    if p == "WR":
        return 23 <= a <= 27
    if p == "TE":
        return 24 <= a <= 28
    if p == "QB":
        return 24 <= a <= 31
    return 23 <= a <= 28

# ---- Win-Now feature engineering ----

# Games Played (robust merge by player_id else name_key)
if weekly_df is not None and not weekly_df.empty:
    w = weekly_df.copy()
    if "name_key" not in w.columns and "name" in w.columns:
        w["name_key"] = normalize_name(w["name"])
    if "name_key" not in df.columns and "name" in df.columns:
        df["name_key"] = normalize_name(df["name"])

    use_pid = ("player_id" in w.columns) and ("player_id" in df.columns)

    if use_pid:
        g = w.groupby("player_id")["week"].nunique().rename("games").reset_index()
        df = df.merge(g.rename(columns={"games": "_gp_tmp"}), on="player_id", how="left")
    else:
        g = w.groupby(["name_key", "pos"])["week"].nunique().rename("games").reset_index()
        df = df.merge(g.rename(columns={"games": "_gp_tmp"}), on=["name_key", "pos"], how="left")

    df["games_played"] = df["_gp_tmp"].fillna(0)
    df.drop(columns=["_gp_tmp"], inplace=True, errors="ignore")
else:
    df["games_played"] = np.nan

# PPG
df["ppg"] = df.apply(
    lambda r: (r["points_total"] / r["games_played"])
    if pd.notna(r.get("points_total")) and r.get("games_played", 0) > 0
    else np.nan,
    axis=1,
)

# EWMA (recent form)
if weekly_df is not None and not weekly_df.empty:
    w = weekly_df.copy()
    if "name_key" not in w.columns and "name" in w.columns:
        w["name_key"] = normalize_name(w["name"])
    key = "player_id" if ("player_id" in w.columns and "player_id" in df.columns) else "name_key"

    def ewma_grp(g, alpha=0.6):
        x = pd.to_numeric(g.sort_values("week")["points"], errors="coerce").fillna(0.0).values
        if len(x) == 0:
            return 0.0
        wts = np.array([(1 - alpha) ** i for i in range(len(x) - 1, -1, -1)], dtype=float)
        wts /= wts.sum()
        return float(np.dot(x, wts))

    cewma = group_apply(w.sort_values("week").groupby([key, "pos"]), ewma_grp).rename("ewma").reset_index()
    if key == "player_id":
        df = df.merge(cewma, on=["player_id", "pos"], how="left")
    else:
        df = df.merge(cewma, on=["name_key", "pos"], how="left")
else:
    df["ewma"] = np.nan

# Trend
df["trend"] = df["ewma"] - df["ppg"]

# Consistency (std over last 4)
if weekly_df is not None and not weekly_df.empty:
    w = weekly_df.copy()
    if "name_key" not in w.columns and "name" in w.columns:
        w["name_key"] = normalize_name(w["name"])
    key = "player_id" if ("player_id" in w.columns and "player_id" in df.columns) else "name_key"

    def last4_std(g):
        x = pd.to_numeric(g.sort_values("week")["points"].tail(4), errors="coerce")
        return float(x.std(ddof=0)) if len(x) > 1 else np.nan

    cstd = group_apply(w.groupby([key, "pos"]), last4_std).rename("std4").reset_index()
    if key == "player_id":
        df = df.merge(cstd, on=["player_id", "pos"], how="left")
    else:
        df = df.merge(cstd, on=["name_key", "pos"], how="left")
else:
    df["std4"] = np.nan

df["price_pct"] = df["market_value"].rank(pct=True).fillna(0.0)
df["z_ppg"] = zscore(df["ppg"].fillna(0.0))
df["z_ewma"] = zscore(df["ewma"].fillna(0.0))
df["z_trend"] = zscore(df["trend"].fillna(0.0))
df["z_consis"] = -zscore(df["std4"].fillna(df["std4"].median() if pd.notna(df["std4"].median()) else 0.0))

# ---------------- Scoring selection ----------------
if mode.startswith("Win"):
    st.subheader("Win-Now blend")
    st.caption("Win-Now prefers hot, consistent producers who are affordable right now.")
    w_form = st.slider("Weight: recent form (EWMA)", 0.0, 1.0, 0.45, 0.05)
    w_trend = st.slider("Weight: trend vs avg", 0.0, 1.0, 0.25, 0.05)
    w_model = st.slider("Weight: model vs market", 0.0, 1.0, 0.20, 0.05)
    w_consis = st.slider("Weight: consistency", 0.0, 1.0, 0.10, 0.05)
    price_pen = st.slider("Price penalty (cost pct)", 0.0, 1.5, 0.85, 0.05)
    cap_bonus = st.slider("Crushing bonus cap", 0.0, 1.5, 0.30, 0.05)
    raw = (
        w_form * df["z_ewma"].fillna(0.0)
        + w_trend * df["z_trend"].fillna(0.0)
        + w_model * df["edge_z_adj"].fillna(0.0)
        + w_consis * df["z_consis"].fillna(0.0)
    )
    df["WinNowScore"] = raw - np.maximum(0.0, price_pen * df["price_pct"].fillna(0.0) - cap_bonus)
    score_col = "WinNowScore"
    score_name = "Win-Now Score (price-adjusted)"
else:
    score_col = "edge_z_adj"
    score_name = "Model vs Market (z)"

# ---------------- Debug strip ----------------
with st.expander("Debug (merge status)", expanded=False):
    st.write(
        f"Weekly rows fetched: **{weekly_rows}** | merged direct by player_id: **{debug_direct}** | "
        f"after fuzzy: **{debug_after}** | season_used: **{meta_fetch.get('season_used','—')}**"
    )


# ---------------- Lineup Optimizer (3rd mode) ----------------
if mode == "Lineup Optimizer":
    st.header("Lineup Optimizer")

    # --- Choose which team to optimize (this was missing) ---
    teams = sorted(df["display_name"].dropna().unique().tolist())
    if not teams:
        st.warning("No teams found in this league.")
        st.stop()
    my_team = st.selectbox("Team to optimize", teams, index=0)
    my_df = df[df["display_name"] == my_team].copy()
    if my_df.empty:
        st.warning("That team has no players loaded yet.")
        st.stop()

    # --- Use Sleeper league settings (slot counts + TE premium) ---
    lineup_cfg = detect_lineup_config_from_sleeper(st.session_state["league_id"])

    # Optional tiny TE prior if league has TE premium
    te_prior = 0.5 if lineup_cfg.te_premium else 0.0
    pos_prior = {"TE": te_prior}

    # --- Recommend starters/bench using momentum projections (weekly_df-backed) ---
    starters_df, bench_df, total_pts = recommend_lineup_with_cfg(my_df, lineup_cfg, pos_prior=pos_prior)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"**Recommended Starters for {my_team}** (sum: {total_pts:.2f})")
        st.dataframe(starters_df[["slot","name","pos","proj_week"]], use_container_width=True)
    with c2:
        st.markdown("**Bench**")
        st.dataframe(bench_df[["name","pos","proj_week"]], use_container_width=True)

    # --- Bench swap ideas ---
    swaps = bench_swap_suggestions(starters_df, bench_df, lineup_cfg)
    st.subheader("Bench Swap Suggestions")
    st.dataframe(swaps, use_container_width=True)

        # --- Roster-safe FA pool (league-wide) ---
    st.subheader("Free-Agent Upgrades")
    fa_mkt = build_roster_safe_fa_pool(dp_mkt, df)  # robust anti-dup filter using full league df
    if fa_mkt is None or fa_mkt.empty:
        st.caption("No free agents (per DP) after roster-safe filtering.")
    else:
        # Score FAs with the same projection function (offseason fallback will kick in if needed)
        fa_scored = project_points_this_week(
            fa_mkt.assign(ewma=np.nan, ppg=np.nan, trend=np.nan, games_played=np.nan),
            pos_prior=pos_prior
        )

        # Cap “offseason” fallbacks by position so randos don’t project like studs
        fa_scored = fa_scored.copy()
        no_signal = fa_scored[["ewma","ppg"]].fillna(0).sum(axis=1).eq(0)
        pos_cap = {"QB": 8.0, "RB": 8.0, "WR": 7.0, "TE": 6.0, "K": 8.0}

        def _cap_row(r):
            cap = pos_cap.get(str(r.get("pos")), 6.0)
            return min(float(r.get("proj_week", 0.0)) * 0.65, cap)

        fa_scored.loc[no_signal, "proj_week"] = fa_scored.loc[no_signal].apply(_cap_row, axis=1)

        # Upgrades relative to your current starters
        # NOTE: pass lineup_cfg (avoid naming it 'cfg' to prevent accidental shadowing)
        fa_up = fa_upgrade_suggestions(starters_df, fa_scored, lineup_cfg, min_delta=0.50, max_results=8)
        st.dataframe(fa_up, use_container_width=True)


    st.markdown("---")
    st.subheader("Weekly Action Plan")

    with st.expander("My Gameplan This Week", expanded=True):
        # Current optimal lineup summary
        st.markdown(f"**Optimal lineup projection for {my_team}:** {total_pts:.2f} pts")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Weakest starters")
            st.dataframe(weakest_starters(starters_df, k=3), use_container_width=True)

        with c2:
            st.caption("Breakout stashes (bench)")
            st.dataframe(breakout_stashes(my_df, weekly_df, limit=5), use_container_width=True)

        with c3:
            st.caption("FA upgrades (this week)")
            # Reuse fa_up from above if available, else try to compute quickly
            if "fa_up" in locals() and fa_up is not None and not fa_up.empty:
                st.dataframe(fa_up, use_container_width=True)
            else:
                fa_mkt_plan = build_roster_safe_fa_pool(dp_mkt, df)
                if fa_mkt_plan is None or fa_mkt_plan.empty:
                    st.write("—")
                else:
                    fa_up_plan = top_fa_upgrades(
                        starters_df, fa_mkt_plan, lineup_cfg, pos_prior,
                        min_delta=0.5, max_results=6
                    )
                    st.dataframe(fa_up_plan, use_container_width=True)

        # Prioritized checklist (bench swaps + FA adds)
        safe_swaps = swaps if isinstance(swaps, pd.DataFrame) else pd.DataFrame()
        safe_fa_up = fa_up if ("fa_up" in locals() and isinstance(fa_up, pd.DataFrame)) else pd.DataFrame()

        plan_df = build_weekly_action_plan(safe_swaps, safe_fa_up)
        st.dataframe(plan_df, use_container_width=True)

        
        st.caption("Prioritized actions")
        st.dataframe(plan_df, use_container_width=True)

    # --- Playoff Odds (Lineup Optimizer tab) ---
    st.markdown("---")
    st.subheader("Playoff Odds (this season)")
    meta = get_league_meta(st.session_state["league_id"])
    start_wk, end_wk, n_playoff = compute_playoff_window(meta)

    if start_wk >= end_wk:
        st.caption("Schedule window too short right now — odds will appear once there are multiple weeks left.")
    else:
        # Use production-based team ratings (z_prod) by default
        odds = simulate_playoff_odds(
            df,
            st.session_state["league_id"],
            start_week=start_wk,
            end_week=end_wk,
            playoff_teams=n_playoff,
            metric_for_ratings="z_prod",
            sims=1500,
        )
        if odds:
            mine = float(odds.get(my_team, 0.0))
            st.write(f"**{my_team}** playoff odds: **{mine*100:.1f}%**  "
                     f"(window: weeks {start_wk}–{end_wk}, {n_playoff} playoff spots)")
            # quick table
            show = (
                pd.Series(odds)
                .rename("Playoff Odds")
                .mul(100.0)
                .round(1)
                .sort_values(ascending=False)
                .rename_axis("Team")
                .to_frame()
            )
            st.dataframe(show, use_container_width=True)
        else:
            st.caption("Couldn’t compute odds — missing schedule or teams.")




    # Done with this mode — prevent the rest of the page from running
    st.stop()





# ---------------- Filters ----------------
st.header("Dashboards")
f1, f2, f3 = st.columns([1, 1, 2])
with f1:
    pos_sel = st.multiselect("Positions", ["QB", "RB", "WR", "TE", "K"], default=["RB", "WR", "TE", "QB"])
with f2:
    max_age = st.slider("Max age", 20, 40, 30)
with f3:
    team_sel = st.selectbox("Filter by team (optional)", ["All"] + sorted(df["display_name"].dropna().unique().tolist()))

df_view = df[df["pos"].isin(pos_sel)].copy()
q20 = df_view.groupby("pos")["market_value"].transform(lambda s: s.quantile(0.20))
df_view = df_view[df_view["market_value"] >= q20]
df_view = df_view[df_view["age"].fillna(99) <= max_age]
if team_sel != "All":
    df_view = df_view[df_view["display_name"] == team_sel]




# ---------------- Buy / Sell (with star protection in both modes) ----------------
L, R = st.columns(2)
base_cols = [
    "display_name",
    "name",
    "pos",
    "age",
    "team",
    "true_value",
    "market_value",
    "edge",
    "edge_z_adj",
    "points_total",
    "z_prod",
    "ppg",
    "ewma",
    "trend",
]
cols_show = [c for c in base_cols if c in df_view.columns]
if score_col not in cols_show:
    cols_show.append(score_col)

have_weekly = weekly_df is not None and not weekly_df.empty

# SELL protection
elite_cut = 0.90 if have_weekly else 0.80
elite = df_view["price_pct"] >= elite_cut
elite_bad = (df_view["edge_z_adj"] <= -0.6) & (df_view["z_ewma"].fillna(0.0) <= 0.0) if have_weekly else (
    df_view["edge_z_adj"] <= -1.10
)
sell_pool = df_view[(~elite) | (elite & elite_bad)]

# BUY lists
if mode.startswith("Win"):
    # Affordable vs Premium (already price-adjusted)
    aff = df_view[df_view["price_pct"] <= 0.60]
    prem = df_view[df_view["price_pct"] > 0.60]
    with L:
        st.subheader("Win-Now BUY: Affordable Targets")
        buy_aff = aff.sort_values(score_col, ascending=False)[cols_show].head(25)
        st.dataframe(friendly_df(buy_aff, score_col), use_container_width=True)
        p1 = st.selectbox("Explain", ["—"] + buy_aff["name"].tolist(), key="why_buy_aff")
        if p1 != "—":
            st.info(why_player(buy_aff[buy_aff["name"] == p1].iloc[0]))

        st.subheader("Win-Now BUY: Premium Upgrades")
        buy_prem = prem.sort_values(score_col, ascending=False)[cols_show].head(15)
        st.dataframe(friendly_df(buy_prem, score_col), use_container_width=True)
else:
    # Market Edge: Prime-Age Buy Lows + Affordable Upside
    df_view["is_prime"] = df_view.apply(is_prime_age, axis=1)
    prime_pool = df_view[(df_view["is_prime"]) & (df_view["price_pct"] <= 0.85)]
    afford_pool = df_view[df_view["price_pct"] <= 0.65]

    with L:
        st.subheader("Market Edge BUY: Prime-Age Buy Lows")
        buy_prime = prime_pool.sort_values("edge_z_adj", ascending=False)[cols_show].head(25)
        st.dataframe(friendly_df(buy_prime, "edge_z_adj"), use_container_width=True)
        p1 = st.selectbox("Explain", ["—"] + buy_prime["name"].tolist(), key="why_buy_prime")
        if p1 != "—":
            st.info(why_player(buy_prime[buy_prime["name"] == p1].iloc[0]))

        st.subheader("Market Edge BUY: Affordable Upside")
        buy_aff = afford_pool.sort_values("edge_z_adj", ascending=False)[cols_show].head(20)
        st.dataframe(friendly_df(buy_aff, "edge_z_adj"), use_container_width=True)

with R:
    st.subheader(f"{'Win-Now' if mode.startswith('Win') else 'Market Edge'} SELL candidates")
    sell_tbl = sell_pool.sort_values(score_col, ascending=True)[cols_show].head(25)
    st.dataframe(friendly_df(sell_tbl, score_col), use_container_width=True)
    p2 = st.selectbox("Explain ", ["—"] + sell_tbl["name"].tolist(), key="why_sell")
    if p2 != "—":
        st.info(why_player(sell_tbl[sell_tbl["name"] == p2].iloc[0]))

# ---------------- Player Lookup (inline weekly chart) ----------------
st.header("Player Lookup")
q = st.text_input("Search by name (type 2+ letters)", value="")
if len(q.strip()) >= 2:
    hits = df[df["name"].str.contains(q, case=False, na=False)].copy()
    if not hits.empty:
        nice_cols = [
            "display_name",
            "name",
            "pos",
            "age",
            "team",
            "true_value",
            "market_value",
            "edge",
            "edge_z_adj",
            "points_total",
            "z_prod",
            "WinNowScore",
            "ppg",
            "ewma",
            "trend",
        ]
        nice_cols = [c for c in nice_cols if c in hits.columns]
        st.dataframe(friendly_df(hits[nice_cols], score_col), use_container_width=True)

        psel = st.selectbox("Chart a player", ["—"] + hits["name"].tolist(), key="lookup_chart")
        if psel != "—":
            if weekly_df is not None and not weekly_df.empty:
                sub = hits.loc[hits["name"] == psel, ["player_id", "name_key"]].iloc[0]
                w = weekly_df.copy()
                if "player_id" in w.columns and pd.notna(sub.get("player_id", np.nan)):
                    w = w[w["player_id"] == str(sub["player_id"])]
                else:
                    w = w[w["name_key"] == sub["name_key"]]
                if not w.empty:
                    line = alt.Chart(w).mark_line(point=True).encode(
                        x=alt.X("week:O", title="Week"),
                        y=alt.Y("points:Q", title="PPR points"),
                        tooltip=["week", "points"],
                    ).properties(height=240)
                    st.altair_chart(line, use_container_width=True)
                else:
                    st.caption("No weekly points found for the chosen season/weeks.")
            else:
                st.caption("Switch to **Win-Now** and pick a season/week range to see weekly points.")
    else:
        st.caption("No matches.")

# ---------------- Team Strength ----------------
st.header("Team Strength Tables")
cA, cB = st.columns(2)
with cA:
    st.markdown("**Asset Strength (Market z)**")
    st.dataframe(strength_table(df, "z_mkt"), use_container_width=True)
with cB:
    st.markdown("**Win-Now Strength (Production z)**")
    st.caption("Computed from this season’s weekly totals if available.")
    st.dataframe(strength_table(df, "z_prod"), use_container_width=True)

# ---------------- Partner Trade Builder ----------------
st.header("Partner Trade Builder")
t1, t2 = st.columns(2)
with t1:
    my_team = st.selectbox("Your team", sorted(df["display_name"].dropna().unique().tolist()))
with t2:
    partner = st.selectbox(
        "Trade partner", [t for t in sorted(df["display_name"].dropna().unique().tolist()) if t != my_team]
    )

tc1, tc2 = st.columns(2)
with tc1:
    buy_thresh = st.slider(f"Tag as BUY when {score_name} ≥", 0.0, 2.0, 0.5, 0.1)
with tc2:
    avoid_thresh = st.slider(f"Tag as AVOID when {score_name} ≤", -2.0, 0.0, -0.5, 0.1)

def tag_from_score(v: float) -> str:
    if pd.isna(v):
        return "FAIR"
    if v >= buy_thresh:
        return "BUY"
    if v <= avoid_thresh:
        return "AVOID"
    return "FAIR"

my_roster = df[df["display_name"] == my_team].copy()
pt_roster = df[df["display_name"] == partner].copy()
my_roster["tag"] = my_roster[score_col].apply(tag_from_score)
pt_roster["tag"] = pt_roster[score_col].apply(tag_from_score)

def labelify(row):
    v = row[score_col]
    vv = "—" if pd.isna(v) else f"{v:.2f}"
    return f"{row['name']} ({row['pos']}, {score_name} {vv}, {row['tag']})"

my_roster["label"] = my_roster.apply(labelify, axis=1)
pt_roster["label"] = pt_roster.apply(labelify, axis=1)

cL2, cR2 = st.columns(2)
with cL2:
    st.markdown("**Send (your players):**")
    give_sel = st.multiselect("Pick players to send", my_roster["label"].tolist(), key="send_any_sel")
    give_pick = my_roster[my_roster["label"].isin(give_sel)]
with cR2:
    st.markdown("**Receive (partner players):**")
    recv_sel = st.multiselect("Pick players to receive", pt_roster["label"].tolist(), key="recv_any_sel")
    recv_pick = pt_roster[pt_roster["label"].isin(recv_sel)]

st.markdown("**Selected Package**")
s_total, r_total, diff, score = quick_balance_score(give_pick, recv_pick)
st.write(
    f"Your send (market): **{s_total:.1f}** | Their send: **{r_total:.1f}** | Diff: **{diff:.1f}** | Fairness: **{score:.2f}**"
)

cols_pkg = ["display_name", "name", "pos", "age", "market_value", "edge", "edge_z_adj"]
if score_col not in cols_pkg:
    cols_pkg.append(score_col)
st.dataframe(friendly_df(give_pick[cols_pkg], score_col), use_container_width=True)
st.dataframe(friendly_df(recv_pick[cols_pkg], score_col), use_container_width=True)

st.subheader("Package Impact (asset z-scores before vs after)")

def strength_table_market(df_all: pd.DataFrame) -> pd.DataFrame:
    return strength_table(df_all, "z_mkt")

def simulate_trade_and_strengths(df_all, my_team, partner, give_df, recv_df):
    z_before = strength_table_market(df_all)
    df_after = df_all.copy()

    def keyify(x):
        return (str(x["display_name"]), str(x["name"]), str(x["pos"]))

    give_keys = set(give_df.apply(keyify, axis=1).tolist())
    recv_keys = set(recv_df.apply(keyify, axis=1).tolist())
    mask_give = df_after.apply(keyify, axis=1).isin(give_keys)
    mask_recv = df_after.apply(keyify, axis=1).isin(recv_keys)
    df_after.loc[mask_give, "display_name"] = partner
    df_after.loc[mask_recv, "display_name"] = my_team
    z_after = strength_table_market(df_after)

    def row_or_zeros(z, t):
        return z.loc[t] if t in z.index else pd.Series({c: 0.0 for c in z.columns})

    show = ["QB", "RB", "WR", "TE", "K", "TOTAL"]
    my_b = row_or_zeros(z_before, my_team)[show]
    my_a = row_or_zeros(z_after, my_team)[show]
    pt_b = row_or_zeros(z_before, partner)[show]
    pt_a = row_or_zeros(z_after, partner)[show]
    my_tbl = pd.DataFrame({"Before": my_b.values, "After": my_a.values, "Δ": (my_a - my_b).values}, index=show)
    pt_tbl = pd.DataFrame({"Before": pt_b.values, "After": pt_a.values, "Δ": (pt_a - pt_b).values}, index=show)
    return my_tbl, pt_tbl

my_tbl, pt_tbl = simulate_trade_and_strengths(df, my_team, partner, give_pick, recv_pick)
cA2, cB2 = st.columns(2)
with cA2:
    st.markdown(f"**Your team: {my_team}**")
    st.dataframe(
        my_tbl.style.format({"Before": "{:.2f}", "After": "{:.2f}", "Δ": "{:+.2f}"}),
        use_container_width=True,
    )
with cB2:
    st.markdown(f"**Partner: {partner}**")
    st.dataframe(
        pt_tbl.style.format({"Before": "{:.2f}", "After": "{:.2f}", "Δ": "{:+.2f}"}),
        use_container_width=True,
    )


# ---------------- Free-Agent Finder (roster-safe) ----------------
st.header("Free-Agent Finder")
try:
    dp_market = dp_to_market(fetch_dp_values())
except Exception:
    dp_market = None

if dp_market is None or dp_market.empty:
    st.info("DynastyProcess upstream unavailable.")
else:
    roster_keys = df[["name", "pos"]].dropna().copy()
    roster_keys["name_key"] = normalize_name(roster_keys["name"])
    m = dp_market.copy()
    m["name_key"] = normalize_name(m["name"])
    m = m[m["pos"].isin(["QB", "RB", "WR", "TE", "K"])]

    exact = m.merge(roster_keys[["pos", "name_key"]].drop_duplicates(), on=["pos", "name_key"], how="left", indicator=True)
    exact["_is_rostered_exact"] = exact["_merge"] == "both"
    exact.drop(columns=["_merge"], inplace=True)

    def jaccard_sets(a: set, b: set) -> float:
        return 0.0 if not a or not b else len(a & b) / len(a | b)

    rost_pos_map: Dict[str, List[Tuple[str, set]]] = {}
    for p, grp in roster_keys.groupby("pos"):
        rost_pos_map[p] = [(nk, set(tokens_for(nk))) for nk in grp["name_key"].unique()]

    def looks_rostered_fuzzy(row, thresh=0.85):
        p = row["pos"]
        nk = row["name_key"]
        toks = set(tokens_for(nk))
        for _, toks_r in rost_pos_map.get(p, []):
            if jaccard_sets(toks, toks_r) >= thresh:
                return True
        return False

    exact["_is_rostered_fuzzy"] = exact.apply(looks_rostered_fuzzy, axis=1)
    fa_mkt = exact[~(exact["_is_rostered_exact"] | exact["_is_rostered_fuzzy"])].copy()
    fa_mkt.drop(columns=["_is_rostered_exact", "_is_rostered_fuzzy"], inplace=True)

    fa = pd.DataFrame(
        {
            "display_name": ["Free Agent"] * len(fa_mkt),
            "name": fa_mkt["name"].astype(str),
            "pos": fa_mkt["pos"].astype(str),
            "age": np.nan,
            "team": "",
        }
    )
    fa = compute_true_value(
        fa, superflex=superflex, te_premium=te_premium, ppr=ppr, age_weight=age_weight, youth_bonus=youth_bonus, age_cap=age_cap
    )
    fa = attach_markets(fa, dp_df=fa_mkt[["name", "pos", "market_value"]], fp_df=None, w_dp=1.0, w_fp=0.0)

    comb = pd.concat(
        [
            df[
                [
                    "name",
                    "pos",
                    "true_value",
                    "market_value",
                    "edge",
                    "edge_z_adj",
                    "display_name",
                    "age",
                    "team",
                    "z_prod",
                    "WinNowScore",
                ]
            ]
            if "WinNowScore" in df.columns
            else df[["name", "pos", "true_value", "market_value", "edge", "edge_z_adj", "display_name", "age", "team"]],
            fa[["name", "pos", "true_value", "market_value", "edge", "edge_z_adj", "display_name", "age", "team"]],
        ],
        ignore_index=True,
    )

    fa_view = comb[(comb["display_name"] == "Free Agent")].copy()
    fa_c1, fa_c2 = st.columns(2)
    with fa_c1:
        fa_pos = st.multiselect("Positions (FA)", ["QB", "RB", "WR", "TE", "K"], default=["RB", "WR", "TE"])
    with fa_c2:
        fa_min_edge = st.slider("Minimum Model vs Market (z)", -0.5, 2.0, 0.2, 0.1)
    fa_view = fa_view[(fa_view["pos"].isin(fa_pos)) & (fa_view["edge_z_adj"] >= fa_min_edge)]
    fa_view = fa_view.sort_values(["edge_z_adj", "market_value"], ascending=[False, False])
    st.caption("Unrostered per DP; your model rates them above consensus.")
    st.dataframe(fa_view[["name", "pos", "true_value", "market_value", "edge", "edge_z_adj"]], use_container_width=True)

st.caption("Tip: in **Win-Now**, pick a past season + week range, then use Player Lookup to chart points.")
