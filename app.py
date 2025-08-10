# app.py — Dynasty Buy/Sell Radar (Sleeper)
# Win-Now w/ weekly fallback + star-protection; Market Edge prime-age buy lows;
# Player lookup chart; FA Finder (roster-safe); saved league IDs.

import io, requests, datetime as dt
from typing import Optional, Iterable, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sleeper_pull import fetch_league_data
from value_engine import compute_true_value, attach_markets
from team_tools import quick_balance_score

# ---------------- App setup ----------------
st.set_page_config(page_title="Dynasty Buy/Sell Radar", layout="wide")
st.title("Dynasty Buy/Sell Radar (Sleeper)")

def _safe_rerun():
    try: st.rerun()
    except Exception:
        try: st.experimental_rerun()
        except Exception: pass

def _clear_prod_cache():
    try: st.cache_data.clear()
    except Exception: pass

# ---------------- Session defaults ----------------
if "league_id" not in st.session_state:
    st.session_state.league_id = "1195252934627844096"
if "league_history" not in st.session_state:
    st.session_state.league_history = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("League")
    league_id_input = st.text_input("Sleeper League ID", value=st.session_state["league_id"])
    c_loadR, c_rem = st.columns([1,1])
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
    row = st.columns([3,1])
    with row[0]:
        old_id_to_add = st.text_input("Add old Sleeper League ID", key="add_old_id",
                                      placeholder="e.g. 987654321012345678", label_visibility="collapsed")
    with row[1]:
        if st.button("Add", key="add_old_btn", use_container_width=True):
            lid = (old_id_to_add or "").strip()
            if lid and lid not in st.session_state.league_history:
                st.session_state.league_history = [lid] + st.session_state.league_history[:9]
                _safe_rerun()

    if st.session_state.league_history:
        saved_sel = st.selectbox("Saved IDs", st.session_state.league_history, key="saved_ids_select")
        c1, c2, c3 = st.columns([1,1,1])
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
        if st.button("Refresh market"): st.cache_data.clear(); _safe_rerun()
    with c2:
        if st.button("Refresh production"): st.cache_data.clear(); _safe_rerun()

    st.subheader("Model knobs (market)")
    if "age_weight" not in st.session_state: st.session_state.age_weight = 1.0
    if "youth_bonus" not in st.session_state: st.session_state.youth_bonus = 3.0
    if "age_cap"    not in st.session_state: st.session_state.age_cap    = 8.0
    age_weight  = st.slider("Age weight", 0.0, 1.5, st.session_state.age_weight, 0.05, key="age_weight")
    youth_bonus = st.slider("Youth bonus (≤24 RB/WR/TE)", 0.0, 6.0, st.session_state.youth_bonus, 0.5, key="youth_bonus")
    age_cap     = st.slider("Age impact cap (abs)", 2.0, 12.0, st.session_state.age_cap, 0.5, key="age_cap")

# ---------------- Top controls ----------------
c1, c2, c3 = st.columns([1,1,2])
with c1:
    scoring_choice = st.selectbox("Scoring", ["PPR","Half","Standard"], index=0)
with c2:
    superflex = st.checkbox("Superflex", False)
    te_premium = st.checkbox("TE Premium (1.5)", False)
with c3:
    w_dp = st.slider("Weight: DynastyProcess", 0.0, 1.0, 0.7, 0.05)
    w_fp = st.slider("Weight: ADP fallback", 0.0, 1.0, 0.3, 0.05)
ppr = (scoring_choice == "PPR")

mode = st.radio("Evaluation mode", ["Market Edge", "Win-Now (auto from Sleeper)"], horizontal=True)

# ---------------- External sources ----------------
DP_URL = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/values-players.csv"
SLEEPER_PLAYERS = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_STATE   = "https://api.sleeper.app/v1/state/nfl"
SLEEPER_STATS_WEEK = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"

def normalize_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9 ]","",regex=True).str.strip()

def tokens_for(s: str) -> List[str]:
    s = "".join(ch if ch.isalnum() or ch == " " else " " for ch in (s or "").lower())
    toks = [t for t in s.split() if t and t not in {"jr","sr","ii","iii","iv","v"}]
    return [t for t in toks if not (len(t) == 1 and t.isalpha())]

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b);  return 0.0 if not sa or not sb else len(sa & sb) / len(sa | sb)

def build_fuzzy_map(prod_keys: pd.DataFrame, roster_keys: pd.DataFrame, threshold: float = 0.85) -> Dict[str, Dict[str, str]]:
    mapping_by_pos: Dict[str, Dict[str, str]] = {}
    roster_by_pos: Dict[str, List[Tuple[str, List[str]]]] = {}
    for pos, grp in roster_keys.groupby("pos"):
        roster_by_pos[pos] = [(nk, tokens_for(nk)) for nk in grp["name_key"].unique()]
    for pos, grp in prod_keys.groupby("pos"):
        target = roster_by_pos.get(pos, [])
        mp: Dict[str, str] = {}
        for nk in grp["name_key"].unique():
            src = tokens_for(nk); best_key, best_score = None, 0.0
            for nk_tgt, tgt in target:
                s = jaccard(src, tgt)
                if s > best_score: best_key, best_score = nk_tgt, s
            if best_key and best_score >= threshold: mp[nk] = best_key
        if mp: mapping_by_pos[pos] = mp
    return mapping_by_pos

def apply_mapping(name_key: str, pos: str, mapping_by_pos: Dict[str, Dict[str, str]]) -> str:
    return mapping_by_pos.get(pos, {}).get(name_key, name_key)

@st.cache_data(ttl=24*60*60)
def fetch_dp_values() -> pd.DataFrame:
    r = requests.get(DP_URL, timeout=30); r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))

@st.cache_data(ttl=24*60*60)
def fetch_sleeper_adp() -> Optional[pd.DataFrame]:
    try:
        players = requests.get(SLEEPER_PLAYERS, timeout=30).json()
        adp = requests.get("https://api.sleeper.app/v1/adp/nfl/2024?type=ppr", timeout=30).json()
        rows = []
        for row in adp:
            pid = row.get("player_id")
            p = players.get(pid) if isinstance(players, dict) else None
            if not p: continue
            name = (p.get("full_name") or (p.get("first_name","")+" "+p.get("last_name","")).strip())
            rows.append({"name": name, "pos": p.get("position"), "Rank": row.get("adp")})
        df = pd.DataFrame(rows)
        return None if df.empty else df
    except Exception:
        return None

def dp_to_market(dp: pd.DataFrame) -> pd.DataFrame:
    df = dp.copy()
    name_candidates = ["name","player_name","player","Player","full_name","Player Name"]
    pos_candidates  = ["pos","position","Position"]
    def pick(cands: Iterable[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns: return c
        return None
    name_col = pick(name_candidates); pos_col = pick(pos_candidates)
    if name_col is None or pos_col is None:
        raise RuntimeError(f"Missing name/pos in DP CSV. Saw: {list(df.columns)}")
    value_cols = [c for c in df.columns if ("_1qb" in c.lower()) or ("value" in c.lower())]
    if not value_cols:
        raise RuntimeError(f"No value columns in DP CSV. Saw: {list(df.columns)}")
    mv = df[value_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return pd.DataFrame({"name": df[name_col].astype(str), "pos": df[pos_col].astype(str), "market_value": mv})

@st.cache_data(ttl=2*60*60)
def get_sleeper_state() -> dict:
    try: return requests.get(SLEEPER_STATE, timeout=15).json()
    except Exception: return {}

def default_season_and_week() -> Tuple[int,int]:
    now = dt.datetime.utcnow()
    state = get_sleeper_state()
    season = int(state.get("season") or now.year)
    week = int(state.get("week") or 1)
    return season, max(1, week)

@st.cache_data(ttl=24*60*60)
def get_players_map() -> Dict[str, dict]:
    try: return requests.get(SLEEPER_PLAYERS, timeout=30).json()
    except Exception: return {}

@st.cache_data(ttl=24*60*60)
def get_league_meta(league_id: str) -> dict:
    try:
        r = requests.get(f"https://api.sleeper.app/v1/league/{league_id}", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

# -------- Weekly fetch + fallback --------
@st.cache_data(ttl=6*60*60)
def fetch_week_df(season: int, week: int) -> pd.DataFrame:
    players = get_players_map()
    try:
        url = SLEEPER_STATS_WEEK.format(season=season, week=week)
        wkstats = requests.get(url, timeout=30).json()
    except Exception:
        wkstats = None

    def gv(row: dict, key, default=0.0):
        v = row.get(key)
        try: return float(v) if v is not None else float(default)
        except Exception: return float(default)

    def calc_ppr(row: dict) -> float:
        if row is None: return 0.0
        if "pts_ppr" in row and row["pts_ppr"] is not None:
            try: return float(row["pts_ppr"])
            except Exception: pass
        return (0.04*gv(row,"pass_yd") + 4*gv(row,"pass_td") - 2*gv(row,"pass_int")
                + 0.1*gv(row,"rush_yd") + 6*gv(row,"rush_td")
                + 0.1*gv(row,"rec_yd")  + 6*gv(row,"rec_td") + 1*gv(row,"rec")
                - 2*(gv(row,"fum_lost", 0.0) or gv(row,"fumbles_lost", 0.0)))

    rows = []
    iterable = wkstats if isinstance(wkstats, list) else (wkstats.values() if isinstance(wkstats, dict) else [])
    for row in iterable:
        pid = str(row.get("player_id") or row.get("player") or "")
        if not pid: continue
        p = players.get(pid) if isinstance(players, dict) else None
        name = (p.get("full_name") or (p.get("first_name","")+" "+p.get("last_name","")).strip()) if p else (row.get("full_name") or row.get("player_name") or "").strip()
        pos  = p.get("position") if p else row.get("position")
        if not name or not pos: continue
        rows.append({"player_id": pid,"name": name,"pos": pos,"week": int(week),"points": float(calc_ppr(row))})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["player_id","name","pos","week","points","name_key"])
    out["name_key"] = normalize_name(out["name"])
    return out

@st.cache_data(ttl=6*60*60)
def fetch_sleeper_points_ppr(season: int, through_week: int) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (totals, weekly, meta), with fallback to previous season if empty."""
    def _one(sea: int, wk: int):
        weeks = list(range(1, max(1, int(wk)) + 1))
        wk_frames = [fetch_week_df(int(sea), w) for w in weeks]
        valid = [w for w in wk_frames if w is not None and not w.empty]
        if not valid: return pd.DataFrame(), pd.DataFrame()
        weekly = pd.concat(valid, ignore_index=True)
        totals = weekly.groupby(["player_id","name","pos","name_key"], as_index=False)["points"].sum()
        totals.rename(columns={"points":"points"}, inplace=True)
        return totals, weekly

    t, w = _one(season, through_week)
    meta = {"season_used": season, "weeks_used": list(range(1, through_week+1)), "source": "direct"}
    if w.empty and season >= 2019:
        t, w = _one(season-1, 18)
        meta = {"season_used": season-1, "weeks_used": list(range(1, 19)), "source": "fallback"}
    return t, w, meta

# ---------------- Win-Now controls ----------------
weekly_df = None
meta_fetch = {}
if mode.startswith("Win"):
    st.subheader("Win-Now production (auto from Sleeper)")
    league_meta = get_league_meta(st.session_state["league_id"])
    league_season_default = int(league_meta.get("season") or default_season_and_week()[0])
    now_season, now_week = default_season_and_week()

    cA, cB, cC = st.columns([1,1,2])
    with cA:
        season = st.number_input("Season", 2018, league_season_default + 1, league_season_default, 1,
                                 key="wn_season", on_change=_clear_prod_cache)
    with cB:
        wk_def = 18 if season < now_season else now_week
        through_week = st.number_input("Through week", 1, 23, int(wk_def), 1,
                                       key="wn_through", on_change=_clear_prod_cache)
    with cC:
        st.caption("Production cache auto-refreshes every 6 hours.")

    with st.spinner(f"Fetching Sleeper production for {season}, week ≤ {through_week}…"):
        _, weekly_df, meta_fetch = fetch_sleeper_points_ppr(int(season), int(through_week))

    if weekly_df is None or weekly_df.empty:
        st.warning("No weekly stats for that season/week yet. Showing last season as a fallback.")
    else:
        st.info(f"Using season **{meta_fetch['season_used']}** ({meta_fetch['source']}).")

# ---------------- Load league + markets ----------------
@st.cache_data(ttl=24*60*60)
def load_all(league_id, superflex, te_premium, ppr, w_dp, w_fp, age_weight, youth_bonus, age_cap, weekly_df=None):
    roster = fetch_league_data(league_id)  # includes player_id
    roster = compute_true_value(
        roster, superflex=superflex, te_premium=te_premium, ppr=ppr,
        age_weight=age_weight, youth_bonus=youth_bonus, age_cap=age_cap,
        weekly_df=weekly_df
    )
    dp_mkt = dp_to_market(fetch_dp_values())
    fp_like = fetch_sleeper_adp()
    roster = attach_markets(roster, dp_df=dp_mkt, fp_df=fp_like, w_dp=w_dp, w_fp=w_fp)
    return roster, dp_mkt, fp_like is not None

try:
    df, dp_mkt, has_fp_like = load_all(
        st.session_state["league_id"], superflex, te_premium, ppr,
        w_dp, w_fp, age_weight, youth_bonus, age_cap, weekly_df=weekly_df
    )
    if df is None or df.empty:
        st.error("Couldn’t load that League ID (404/private/empty)."); st.stop()
    lid = st.session_state["league_id"].strip()
    if lid and lid not in st.session_state.league_history:
        st.session_state.league_history = [lid] + st.session_state.league_history[:9]
except Exception:
    st.error("Couldn’t load that League ID (network or invalid). Try again."); st.stop()

# ---------------- Merge weekly -> roster (player_id first, then fuzzy) ----------------
if "name_key" not in df.columns:
    df["name_key"] = normalize_name(df["name"])

debug_direct, debug_after = 0, 0
weekly_rows = 0
if weekly_df is not None and not weekly_df.empty:
    wk = weekly_df.copy(); weekly_rows = len(wk)
    # exact via player_id
    if "player_id" in df.columns and "player_id" in wk.columns:
        by_pid = wk.groupby("player_id", as_index=False)["points"].sum().rename(columns={"points":"points_total"})
        df = df.merge(by_pid, on="player_id", how="left")
    if "points_total" in df.columns: debug_direct = int(df["points_total"].notna().sum())
    # fuzzy fill for holes
    need_map_mask = df["points_total"].isna() if "points_total" in df.columns else pd.Series(True, index=df.index)
    if need_map_mask.any():
        prod_keys = wk[["name_key","pos"]].drop_duplicates()
        rost_keys = df.loc[need_map_mask, ["name_key","pos"]].drop_duplicates()
        mapping_by_pos = build_fuzzy_map(prod_keys, rost_keys, threshold=0.85)
        wk_mapped = wk.copy()
        wk_mapped["name_key"] = wk_mapped.apply(lambda r: apply_mapping(r["name_key"], r["pos"], mapping_by_pos), axis=1)
        by_key = wk_mapped.groupby(["name_key","pos"], as_index=False)["points"].sum().rename(columns={"points":"points_total_fuzzy"})
        df = df.merge(by_key, on=["name_key","pos"], how="left")
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
    if "points_total" not in df.columns: df["points_total"] = np.nan
    if "z_prod" not in df.columns: df["z_prod"] = np.nan

st.caption(
    f"Sources: DP=Yes, ADP={'Yes' if has_fp_like else 'No'} | "
    f"Blend: DP {w_dp:.2f} / ADP {w_fp:.2f} | "
    f"Age wt {age_weight:.2f}, Youth {youth_bonus:.1f}, Cap ±{age_cap:.1f}"
)

# ---------------- Helpers ----------------
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2: return pd.Series(0.0, index=s.index, dtype=float)
    mu, sd = s.mean(), s.std(ddof=0)
    if not sd or np.isnan(sd): return pd.Series(0.0, index=s.index, dtype=float)
    return (s - mu) / sd

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
    for col in ["QB","RB","WR","TE","K"]:
        if col not in pivot.columns: pivot[col] = 0.0
    pivot = pivot[["QB","RB","WR","TE","K"]]; pivot["TOTAL"] = pivot[["QB","RB","WR","TE","K"]].sum(axis=1)
    return pivot.sort_values("TOTAL", ascending=False).round(4)

def why_player(row: pd.Series) -> str:
    bits = []
    if "edge_z_adj" in row and pd.notna(row["edge_z_adj"]): bits.append(f"Model vs Market: {row['edge_z_adj']:+.2f} z")
    if "edge" in row and pd.notna(row["edge"]): bits.append(f"Raw gap: {row['edge']:+.1f}")
    if "points_total" in row and pd.notna(row["points_total"]): bits.append(f"Season PPR: {row['points_total']:.1f}")
    if "ppg" in row and pd.notna(row["ppg"]): bits.append(f"PPG: {row['ppg']:.2f}")
    if "ewma" in row and pd.notna(row["ewma"]): bits.append(f"Recent form (EWMA): {row['ewma']:.1f}")
    if "trend" in row and pd.notna(row["trend"]): bits.append(f"Trend vs avg: {row['trend']:+.1f}")
    if "z_consis" in row and pd.notna(row["z_consis"]): bits.append(f"Consistency: {row['z_consis']:+.2f} z")
    return " • ".join(bits) if bits else "No extra notes yet."

def friendly_df(df_in: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df_in.copy()
    rename_map = {
        "display_name":"Team","name":"Player","pos":"Pos","age":"Age","team":"NFL",
        "true_value":"Model Value","market_value":"Market Value","edge":"Raw Diff",
        "edge_z_adj":"Model vs Market (z)","WinNowScore":"Win-Now Score","z_prod":"Prod z",
        "points_total":"Season PPR","ppg":"PPG","ewma":"EWMA","trend":"Trend"
    }
    cols = [c for c in rename_map if c in out.columns] + ([score_col] if score_col not in rename_map else [])
    out = out[cols].rename(columns=rename_map)
    num_cols = [c for c in ["Age","Model Value","Market Value","Raw Diff","Model vs Market (z)","Win-Now Score","Prod z","Season PPR","PPG","EWMA","Trend"] if c in out.columns]
    for c in num_cols: out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out

def is_prime_age(row) -> bool:
    a = row.get("age", np.nan); p = row.get("pos")
    if pd.isna(a): return False
    a = float(a)
    if p == "RB": return 23 <= a <= 26
    if p == "WR": return 23 <= a <= 27
    if p == "TE": return 24 <= a <= 28
    if p == "QB": return 24 <= a <= 31
    return 23 <= a <= 28

# ---- Win-Now feature engineering ----
if weekly_df is not None and not weekly_df.empty:
    games = weekly_df.groupby("player_id" if "player_id" in weekly_df.columns else ["name_key","pos"])["week"].nunique()
    if "player_id" in weekly_df.columns and "player_id" in df.columns:
        df["games_played"] = df["player_id"].map(games).fillna(0)
    else:
        g2 = games.reset_index().rename(columns={"week":"games"})
        df = df.merge(g2.rename(columns={"games":"_gp_tmp"}), left_on=["name_key","pos"], right_on=["name_key","pos"], how="left")
        df["games_played"] = df["_gp_tmp"].fillna(0); df.drop(columns=["_gp_tmp"], inplace=True, errors="ignore")
else:
    df["games_played"] = np.nan

df["ppg"] = df.apply(lambda r: (r["points_total"]/r["games_played"]) if pd.notna(r.get("points_total")) and r.get("games_played",0)>0 else np.nan, axis=1)

if weekly_df is not None and not weekly_df.empty:
    w = weekly_df.copy()
    key = "player_id" if ("player_id" in w.columns and "player_id" in df.columns) else "name_key"
    def ewma_grp(g, alpha=0.6):
        x = pd.to_numeric(g["points"], errors="coerce").fillna(0.0).values
        if len(x)==0: return 0.0
        wts = np.array([(1-alpha)**i for i in range(len(x)-1,-1,-1)], dtype=float); wts /= wts.sum()
        return float(np.dot(x, wts))
    ew = w.sort_values(["week"]).groupby([key,"pos"], as_index=False).apply(lambda g: ewma_grp(g)).rename(columns={None:"ewma"})
    if key == "player_id":
        df = df.merge(ew[[key,"pos","ewma"]], on=[key,"pos"], how="left")
    else:
        df = df.merge(ew[[key,"pos","ewma"]], left_on=["name_key","pos"], right_on=[key,"pos"], how="left")
        df.drop(columns=["name_key_y"], inplace=True, errors="ignore"); df.rename(columns={"name_key_x":"name_key"}, inplace=True, errors="ignore")
else:
    df["ewma"] = np.nan

df["trend"] = df["ewma"] - df["ppg"]
if weekly_df is not None and not weekly_df.empty:
    w = weekly_df.copy()
    def last4_std(g):
        x = pd.to_numeric(g.sort_values("week")["points"].tail(4), errors="coerce")
        return float(x.std(ddof=0)) if len(x)>1 else np.nan
    if "player_id" in w.columns and "player_id" in df.columns:
        cstd = w.groupby(["player_id","pos"]).apply(last4_std).rename("std4").reset_index()
        df = df.merge(cstd, on=["player_id","pos"], how="left")
    else:
        cstd = w.groupby(["name_key","pos"]).apply(last4_std).rename("std4").reset_index()
        df = df.merge(cstd, on=["name_key","pos"], how="left")
else:
    df["std4"] = np.nan

df["price_pct"] = df["market_value"].rank(pct=True).fillna(0.0)
df["z_ppg"]    = zscore(df["ppg"].fillna(0.0))
df["z_ewma"]   = zscore(df["ewma"].fillna(0.0))
df["z_trend"]  = zscore(df["trend"].fillna(0.0))
df["z_consis"] = -zscore(df["std4"].fillna(df["std4"].median() if pd.notna(df["std4"].median()) else 0.0))

# ---------------- Scoring selection ----------------
if mode.startswith("Win"):
    st.subheader("Win-Now blend")
    st.caption("Win-Now prefers hot, consistent producers who are affordable right now.")
    w_form   = st.slider("Weight: recent form (EWMA)", 0.0, 1.0, 0.45, 0.05)
    w_trend  = st.slider("Weight: trend vs avg",       0.0, 1.0, 0.25, 0.05)
    w_model  = st.slider("Weight: model vs market",    0.0, 1.0, 0.20, 0.05)
    w_consis = st.slider("Weight: consistency",        0.0, 1.0, 0.10, 0.05)
    price_pen= st.slider("Price penalty (cost pct)",   0.0, 1.5, 0.85, 0.05)
    cap_bonus= st.slider("Crushing bonus cap",         0.0, 1.5, 0.30, 0.05)
    raw = (w_form*df["z_ewma"].fillna(0.0) + w_trend*df["z_trend"].fillna(0.0) +
           w_model*df["edge_z_adj"].fillna(0.0) + w_consis*df["z_consis"].fillna(0.0))
    df["WinNowScore"] = raw - np.maximum(0.0, price_pen * df["price_pct"].fillna(0.0) - cap_bonus)
    score_col = "WinNowScore"; score_name = "Win-Now Score (price-adjusted)"
else:
    score_col = "edge_z_adj"; score_name = "Model vs Market (z)"

# ---------------- Debug strip ----------------
with st.expander("Debug (merge status)", expanded=False):
    st.write(
        f"Weekly rows fetched: **{weekly_rows}** | merged direct by player_id: **{debug_direct}** | "
        f"after fuzzy: **{debug_after}** | season_used: **{meta_fetch.get('season_used','—')}**"
    )

# ---------------- Filters ----------------
st.header("Dashboards")
f1, f2, f3 = st.columns([1,1,2])
with f1:
    pos_sel = st.multiselect("Positions", ["QB","RB","WR","TE","K"], default=["RB","WR","TE","QB"])
with f2:
    max_age = st.slider("Max age", 20, 40, 30)
with f3:
    team_sel = st.selectbox("Filter by team (optional)", ["All"] + sorted(df["display_name"].dropna().unique().tolist()))

df_view = df[df["pos"].isin(pos_sel)].copy()
q20 = df_view.groupby("pos")["market_value"].transform(lambda s: s.quantile(0.20))
df_view = df_view[df_view["market_value"] >= q20]
df_view = df_view[df_view["age"].fillna(99) <= max_age]
if team_sel != "All": df_view = df_view[df_view["display_name"] == team_sel]

# ---------------- Buy / Sell (with star protection in both modes) ----------------
L, R = st.columns(2)
base_cols = ["display_name","name","pos","age","team","true_value","market_value","edge","edge_z_adj",
             "points_total","z_prod","ppg","ewma","trend"]
cols_show = [c for c in base_cols if c in df_view.columns]
if score_col not in cols_show: cols_show.append(score_col)

have_weekly = weekly_df is not None and not weekly_df.empty

# SELL protection: if weekly data → protect top 10% unless bad model and bad form;
# if no weekly → protect top 20% outright (only allow if edge_z_adj extremely poor).
elite_cut = 0.90 if have_weekly else 0.80
elite = df_view["price_pct"] >= elite_cut
elite_bad = (df_view["edge_z_adj"] <= -0.6) & (df_view["z_ewma"].fillna(0.0) <= 0.0) if have_weekly else (df_view["edge_z_adj"] <= -1.10)
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
        if p1 != "—": st.info(why_player(buy_aff[buy_aff["name"]==p1].iloc[0]))

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
        if p1 != "—": st.info(why_player(buy_prime[buy_prime["name"]==p1].iloc[0]))

        st.subheader("Market Edge BUY: Affordable Upside")
        buy_aff = afford_pool.sort_values("edge_z_adj", ascending=False)[cols_show].head(20)
        st.dataframe(friendly_df(buy_aff, "edge_z_adj"), use_container_width=True)

with R:
    st.subheader(f"{'Win-Now' if mode.startswith('Win') else 'Market Edge'} SELL candidates")
    sell_tbl = sell_pool.sort_values(score_col, ascending=True)[cols_show].head(25)
    st.dataframe(friendly_df(sell_tbl, score_col), use_container_width=True)
    p2 = st.selectbox("Explain ", ["—"] + sell_tbl["name"].tolist(), key="why_sell")
    if p2 != "—": st.info(why_player(sell_tbl[sell_tbl["name"]==p2].iloc[0]))

# ---------------- Player Lookup (inline weekly chart) ----------------
st.header("Player Lookup")
q = st.text_input("Search by name (type 2+ letters)", value="")
if len(q.strip()) >= 2:
    hits = df[df["name"].str.contains(q, case=False, na=False)].copy()
    if not hits.empty:
        nice_cols = ["display_name","name","pos","age","team","true_value","market_value","edge","edge_z_adj","points_total","z_prod","WinNowScore","ppg","ewma","trend"]
        nice_cols = [c for c in nice_cols if c in hits.columns]
        st.dataframe(friendly_df(hits[nice_cols], score_col), use_container_width=True)

        psel = st.selectbox("Chart a player", ["—"] + hits["name"].tolist(), key="lookup_chart")
        if psel != "—":
            if weekly_df is not None and not weekly_df.empty:
                sub = hits.loc[hits["name"] == psel, ["player_id","name_key"]].iloc[0]
                w = weekly_df.copy()
                if "player_id" in w.columns and pd.notna(sub.get("player_id", np.nan)):
                    w = w[w["player_id"] == str(sub["player_id"])]
                else:
                    w = w[w["name_key"] == sub["name_key"]]
                if not w.empty:
                    line = alt.Chart(w).mark_line(point=True).encode(
                        x=alt.X("week:O", title="Week"),
                        y=alt.Y("points:Q", title="PPR points"),
                        tooltip=["week","points"]
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
    partner = st.selectbox("Trade partner", [t for t in sorted(df["display_name"].dropna().unique().tolist()) if t != my_team])

tc1, tc2 = st.columns(2)
with tc1:
    buy_thresh = st.slider(f"Tag as BUY when {score_name} ≥", 0.0, 2.0, 0.5, 0.1)
with tc2:
    avoid_thresh = st.slider(f"Tag as AVOID when {score_name} ≤", -2.0, 0.0, -0.5, 0.1)

def tag_from_score(v: float) -> str:
    if pd.isna(v): return "FAIR"
    if v >= buy_thresh: return "BUY"
    if v <= avoid_thresh: return "AVOID"
    return "FAIR"

my_roster = df[df["display_name"] == my_team].copy()
pt_roster = df[df["display_name"] == partner].copy()
my_roster["tag"] = my_roster[score_col].apply(tag_from_score)
pt_roster["tag"] = pt_roster[score_col].apply(tag_from_score)

def labelify(row):
    v = row[score_col]; vv = "—" if pd.isna(v) else f"{v:.2f}"
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
st.write(f"Your send (market): **{s_total:.1f}** | Their send: **{r_total:.1f}** | Diff: **{diff:.1f}** | Fairness: **{score:.2f}**")

cols_pkg = ["display_name","name","pos","age","market_value","edge","edge_z_adj"]
if score_col not in cols_pkg: cols_pkg.append(score_col)
st.dataframe(friendly_df(give_pick[cols_pkg], score_col), use_container_width=True)
st.dataframe(friendly_df(recv_pick[cols_pkg], score_col), use_container_width=True)

st.subheader("Package Impact (asset z-scores before vs after)")
def strength_table_market(df_all: pd.DataFrame) -> pd.DataFrame: return strength_table(df_all, "z_mkt")
def simulate_trade_and_strengths(df_all, my_team, partner, give_df, recv_df):
    z_before = strength_table_market(df_all); df_after = df_all.copy()
    def keyify(x): return (str(x["display_name"]), str(x["name"]), str(x["pos"]))
    give_keys = set(give_df.apply(keyify, axis=1).tolist()); recv_keys = set(recv_df.apply(keyify, axis=1).tolist())
    mask_give = df_after.apply(keyify, axis=1).isin(give_keys); mask_recv = df_after.apply(keyify, axis=1).isin(recv_keys)
    df_after.loc[mask_give, "display_name"] = partner; df_after.loc[mask_recv, "display_name"] = my_team
    z_after = strength_table_market(df_after)
    def row_or_zeros(z, t): return z.loc[t] if t in z.index else pd.Series({c:0.0 for c in z.columns})
    show = ["QB","RB","WR","TE","K","TOTAL"]
    my_b = row_or_zeros(z_before, my_team)[show]; my_a = row_or_zeros(z_after, my_team)[show]
    pt_b = row_or_zeros(z_before, partner)[show]; pt_a = row_or_zeros(z_after, partner)[show]
    my_tbl = pd.DataFrame({"Before": my_b.values, "After": my_a.values, "Δ": (my_a-my_b).values}, index=show)
    pt_tbl = pd.DataFrame({"Before": pt_b.values, "After": pt_a.values, "Δ": (pt_a-pt_b).values}, index=show)
    return my_tbl, pt_tbl
my_tbl, pt_tbl = simulate_trade_and_strengths(df, my_team, partner, give_pick, recv_pick)
cA2, cB2 = st.columns(2)
with cA2: st.markdown(f"**Your team: {my_team}**"); st.dataframe(my_tbl.style.format({"Before":"{:.2f}","After":"{:.2f}","Δ":"{:+.2f}"}), use_container_width=True)
with cB2: st.markdown(f"**Partner: {partner}**"); st.dataframe(pt_tbl.style.format({"Before":"{:.2f}","After":"{:.2f}","Δ":"{:+.2f}"}), use_container_width=True)

# ---------------- Free-Agent Finder (roster-safe) ----------------
st.header("Free-Agent Finder")
try:
    dp_market = dp_to_market(fetch_dp_values())
except Exception:
    dp_market = None

if dp_market is None or dp_market.empty:
    st.info("DynastyProcess upstream unavailable.")
else:
    roster_keys = df[["name","pos"]].dropna().copy(); roster_keys["name_key"] = normalize_name(roster_keys["name"])
    m = dp_market.copy(); m["name_key"] = normalize_name(m["name"]); m = m[m["pos"].isin(["QB","RB","WR","TE","K"])]

    exact = m.merge(roster_keys[["pos","name_key"]].drop_duplicates(), on=["pos","name_key"], how="left", indicator=True)
    exact["_is_rostered_exact"] = (exact["_merge"] == "both"); exact.drop(columns=["_merge"], inplace=True)
    def jaccard_sets(a: set, b: set) -> float: return 0.0 if not a or not b else len(a & b) / len(a | b)
    rost_pos_map: Dict[str, List[Tuple[str, set]]] = {}
    for p, grp in roster_keys.groupby("pos"):
        rost_pos_map[p] = [(nk, set(tokens_for(nk))) for nk in grp["name_key"].unique()]
    def looks_rostered_fuzzy(row, thresh=0.85):
        p = row["pos"]; nk = row["name_key"]; toks = set(tokens_for(nk))
        for _, toks_r in rost_pos_map.get(p, []):
            if jaccard_sets(toks, toks_r) >= thresh: return True
        return False
    exact["_is_rostered_fuzzy"] = exact.apply(looks_rostered_fuzzy, axis=1)
    fa_mkt = exact[~(exact["_is_rostered_exact"] | exact["_is_rostered_fuzzy"])].copy()
    fa_mkt.drop(columns=["_is_rostered_exact","_is_rostered_fuzzy"], inplace=True)

    fa = pd.DataFrame({
        "display_name": ["Free Agent"] * len(fa_mkt),
        "name": fa_mkt["name"].astype(str),
        "pos": fa_mkt["pos"].astype(str),
        "age": np.nan, "team": "",
    })
    fa = compute_true_value(fa, superflex=superflex, te_premium=te_premium, ppr=ppr,
                            age_weight=age_weight, youth_bonus=youth_bonus, age_cap=age_cap)
    fa = attach_markets(fa, dp_df=fa_mkt[["name","pos","market_value"]], fp_df=None, w_dp=1.0, w_fp=0.0)

    comb = pd.concat([
        df[["name","pos","true_value","market_value","edge","edge_z_adj","display_name","age","team","z_prod","WinNowScore"]]
        if "WinNowScore" in df.columns else
        df[["name","pos","true_value","market_value","edge","edge_z_adj","display_name","age","team"]],
        fa[["name","pos","true_value","market_value","edge","edge_z_adj","display_name","age","team"]],
    ], ignore_index=True)

    fa_view = comb[(comb["display_name"] == "Free Agent")].copy()
    fa_c1, fa_c2 = st.columns(2)
    with fa_c1:
        fa_pos = st.multiselect("Positions (FA)", ["QB","RB","WR","TE","K"], default=["RB","WR","TE"])
    with fa_c2:
        fa_min_edge = st.slider("Minimum Model vs Market (z)", -0.5, 2.0, 0.2, 0.1)
    fa_view = fa_view[(fa_view["pos"].isin(fa_pos)) & (fa_view["edge_z_adj"] >= fa_min_edge)]
    fa_view = fa_view.sort_values(["edge_z_adj","market_value"], ascending=[False, False])
    st.caption("Unrostered per DP; your model rates them above consensus.")
    st.dataframe(fa_view[["name","pos","true_value","market_value","edge","edge_z_adj"]], use_container_width=True)

st.caption("Tip: in **Win-Now**, pick a past season + week range, then use Player Lookup to chart points.")

