# value_engine.py — V4.1
# VAR + role momentum + 2-year decay + Bayesian blend + elite shrink in market attach.

from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd

# ---------- utilities ----------


def _canonicalize_alias_key(nk: str) -> str:
    # nicknames → canonical tokens
    if not nk:
        return nk
    toks = nk.split()
    s = set(toks)
    # Marquise “Hollywood” Brown
    if "brown" in s and "hollywood" in s:
        toks = [t for t in toks if t != "hollywood"]
        if "marquise" not in toks:
            toks = ["marquise"] + toks
    return " ".join(toks)


def _series_or_default(df_like, col: str, default: float = 0.0) -> pd.Series:
    import pandas as pd
    import numpy as np
    if hasattr(df_like, "columns"):  # DataFrame
        if col in df_like.columns:
            return pd.to_numeric(df_like[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df_like.index, dtype=float)
    if hasattr(df_like, "index"):    # Series
        return pd.to_numeric(df_like, errors="coerce").fillna(default)
    # scalar or None
    return pd.Series(default, index=None, dtype=float)



def _z(s: pd.Series) -> pd.Series:
    mu = s.mean(); sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd): return s * 0
    return (s - mu) / sd

def _clean_name_key(series: pd.Series) -> pd.Series:
    # Normalize names for robust joins across Sleeper/DP/ADP
    s = series.astype(str).str.lower()
    s = s.str.replace(r"[^a-z0-9 ]", "", regex=True)                # drop punctuation
    s = s.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True)   # drop suffixes
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()          # squeeze spaces
    return s


def _ewma_np(values: np.ndarray, alpha: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0: return 0.0
    w = np.array([(1 - alpha) ** i for i in range(v.size - 1, -1, -1)], dtype=float)
    w /= w.sum()
    return float((v * w).sum())

def _safe_series_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(default, index=df.index)), errors="coerce").fillna(default)

# ---------- priors ----------
def _pos_baseline(pos: str, superflex: bool, te_premium: bool) -> float:
    if pos == "QB": return 1.10 if superflex else 0.85
    if pos == "RB": return 1.00
    if pos == "WR": return 0.95
    if pos == "TE": return 0.95 if te_premium else 0.85
    if pos == "K":  return 0.30
    return 0.75

def _age_adj(age: float, pos: str, age_weight: float, youth_bonus: float, age_cap: float) -> float:
    if pd.isna(age): return 0.0
    sweet = {"QB": 27, "RB": 24, "WR": 25, "TE": 26, "K": 28}
    base = sweet.get(pos, 25)
    delta = (base - float(age)) / 10.0
    if pos in ("RB","WR","TE") and age <= 24:
        delta += (youth_bonus / 10.0)
    delta *= age_weight
    return float(np.clip(delta, -age_cap/10.0, age_cap/10.0))

# ---------- role momentum ----------
_ROLE_COLS = {
    "WR": ["targets", "routes_run", "snap_share"],
    "RB": ["rush_att", "targets", "snap_share"],
    "TE": ["targets", "routes_run", "snap_share"],
    "QB": ["snap_share"],
    "K" : ["snap_share"],
}

def _compute_role_momentum(weekly: pd.DataFrame, alpha_recent: float = 0.6) -> pd.DataFrame:
    w = weekly.copy()
    if "name_key" not in w or "pos" not in w or "week" not in w:
        return pd.DataFrame(columns=["name_key","pos","role_momentum","role_stability"])
    w["week"] = pd.to_numeric(w["week"], errors="coerce").fillna(0).astype(int)

    parts = []
    for pos, cols in _ROLE_COLS.items():
        cols_avail = [c for c in cols if c in w.columns]
        if not cols_avail: continue
        ww = w[w["pos"] == pos].copy()
        for c in cols_avail:
            ww[c] = pd.to_numeric(ww[c], errors="coerce").fillna(0.0)
        agg = ww.groupby(["name_key","pos","week"], as_index=False)[cols_avail].sum().sort_values("week")

        def _comp(g: pd.DataFrame) -> pd.Series:
            vals = g[cols_avail].sum(axis=1).to_numpy(dtype=float)
            full = _ewma_np(vals, alpha=0.35)
            recent = _ewma_np(vals[-3:], alpha=alpha_recent) if len(vals) else 0.0
            ratio = (recent / (full + 1e-6))
            momentum = float(np.clip(0.5 + 0.5*ratio, 0.8, 1.2))
            last6 = vals[-6:] if len(vals) >= 1 else np.array([0.0])
            st = float((np.mean(last6) / (np.std(last6, ddof=0) + 1.0)))
            stability = float(np.clip(st / 3.0, 0.0, 1.0))
            return pd.Series({"role_momentum": momentum, "role_stability": stability})

            parts.append(
                agg.groupby(["name_key","pos"], as_index=False)
                 .apply(_comp)
        )


    if not parts:
        return pd.DataFrame(columns=["name_key","pos","role_momentum","role_stability"])
    out = pd.concat(parts, ignore_index=True)
    return out.groupby(["name_key","pos"], as_index=False).mean(numeric_only=True)

# ---------- future decay ----------
def _future_decay_multiplier(age: float, pos: str) -> float:
    if pd.isna(age): return 1.0
    age = float(age)
    peaks = {"QB": 29, "RB": 25, "WR": 27, "TE": 28, "K": 30}
    slopes = {"QB": 0.010, "RB": 0.040, "WR": 0.025, "TE": 0.020, "K": 0.010}
    grow   = {"QB": 0.015, "RB": 0.020, "WR": 0.018, "TE": 0.015, "K": 0.010}
    peak = peaks.get(pos, 27)
    if age < peak:
        years = min(2.0, peak - age)
        mult = 1.0 + grow.get(pos, 0.02) * years
    else:
        years = min(2.0, age - peak)
        mult = 1.0 - slopes.get(pos, 0.02) * years
    return float(np.clip(mult, 0.70, 1.15))

# ---------- main ----------
def compute_true_value(
    roster_df: pd.DataFrame,
    superflex: bool = False,
    te_premium: bool = False,
    ppr: bool = True,
    age_weight: float = 1.0,
    youth_bonus: float = 3.0,
    age_cap: float = 8.0,
    weekly_df: Optional[pd.DataFrame] = None,
    ewma_alpha: float = 0.75,
    discount_gamma: float = 0.97,
    horizon_weeks: int = 16,
) -> pd.DataFrame:
    df = roster_df.copy()
    for c in ("name","pos"):
        if c not in df.columns:
            raise ValueError(f"compute_true_value missing column: {c}")
    passthru = [c for c in ["display_name","team","player_id","sleeper_id","id"] if c in df.columns]

    df["pos"] = df["pos"].astype(str)
    df["age"] = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    if "name_key" not in df.columns:
        df["name_key"] = _clean_name_key(df["name"])

    base = df["pos"].map(lambda p: _pos_baseline(p, superflex, te_premium))
    ppr_boost = 1.0 + (0.05 if ppr else 0.0)
    if te_premium:
        base = base.mask(df["pos"].eq("TE"), base * 1.05)
    ageadj = df.apply(lambda r: _age_adj(r["age"], r["pos"], age_weight, youth_bonus, age_cap), axis=1)
    prior_value = (base * ppr_boost) * (1.0 + ageadj) * 100.0

    # no weekly: prior + decay glimpse
    if weekly_df is None or weekly_df.empty:
        out = df[["name","pos","age","name_key"] + passthru].copy()
        out["true_value"] = prior_value
        out["confidence"] = 0.55
        out["future_mult"] = df.apply(lambda r: _future_decay_multiplier(r["age"], r["pos"]), axis=1)
        out["true_value"] = out["true_value"] * (0.7 + 0.3*out["future_mult"])
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.55).clip(0,1)
        return out

    # weekly cleanup
    w = weekly_df.copy()
    need = {"name_key","pos","week","points"}
    if any(c not in w.columns for c in need):
        out = df[["name","pos","age","name_key"] + passthru].copy()
        out["true_value"] = prior_value
        out["confidence"] = 0.55
        out["future_mult"] = df.apply(lambda r: _future_decay_multiplier(r["age"], r["pos"]), axis=1)
        out["true_value"] = out["true_value"] * (0.7 + 0.3*out["future_mult"])
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.55).clip(0,1)
        return out

    w["points"] = pd.to_numeric(w["points"], errors="coerce").fillna(0.0)
    w["week"]   = pd.to_numeric(w["week"], errors="coerce").fillna(0).astype(int)

    # replacement level
    repl_week = (w[w["points"] > 0].groupby(["pos","week"], as_index=False)["points"].median().rename(columns={"points":"rep_week"}))
    repl_pos  = repl_week.groupby("pos", as_index=False)["rep_week"].mean().rename(columns={"rep_week":"rep"})
    w = w.merge(repl_pos, on="pos", how="left")
    w["var"] = (w["points"] - w["rep"]).clip(lower=0.0)

    role = _compute_role_momentum(w)

    def _agg_player(g: pd.DataFrame) -> pd.Series:
        pts = g["var"].to_numpy(dtype=float)
        ew  = _ewma_np(pts, alpha=ewma_alpha)
        mean = float(np.mean(pts) if pts.size else 0.0)
        std  = float(np.std(pts, ddof=0) if pts.size else 0.0)
        last_week = int(g["week"].max()) if len(g) else 0
        return pd.Series({"games": int((g["points"] > 0).sum()),
                          "ewma_var": ew, "mean_var": mean, "std_var": std, "last_week": last_week})

    stats = w.groupby(["name_key","pos"], as_index=False).apply(_agg_player)
    stats = stats.merge(role, on=["name_key","pos"], how="left")

    current_like = int(w["week"].max()) if len(w["week"].unique()) else 1
    coverage  = (1.0 - np.exp(-stats["games"] / 6.0)).clip(0,1)
    stability = (stats["mean_var"] / (stats["std_var"] + 1.0)).clip(0,3) / 3.0
    recency   = (discount_gamma ** (current_like - stats["last_week"]).clip(lower=0)).clip(0,1)
    role_stab = stats.get("role_stability", pd.Series(0.5, index=stats.index)).clip(0,1)
    stats["confidence"] = (0.20 + 0.45*coverage + 0.25*stability + 0.05*recency + 0.05*role_stab).clip(0,1)

    pos_scale = {"QB": 20.0, "RB": 24.0, "WR": 21.0, "TE": 16.5, "K": 4.0}
    stats["uplift"] = stats.apply(lambda r: float(r["ewma_var"]) * pos_scale.get(str(r["pos"]), 20.0), axis=1)

    tau_prior = {"QB": 0.8, "RB": 0.9, "WR": 0.9, "TE": 0.85, "K": 0.6}
    tau_data = (0.15 + 0.12 * stats["games"].clip(0, 12) / 12.0 + 0.08 * ((stats["mean_var"] / (stats["std_var"] + 1.0)).clip(0,3) / 3.0)).clip(0.15, 0.5)
    stats["w_data"] = (tau_data / (tau_data + stats["pos"].map(tau_prior).fillna(0.8))).clip(0.05, 0.8)

    out = df.merge(stats[["name_key","pos","uplift","w_data","confidence","role_momentum"]], on=["name_key","pos"], how="left")
    out["true_value"] = prior_value

    has_prod = out["uplift"].notna()
    out.loc[has_prod, "true_value"] = ((1.0 - out.loc[has_prod, "w_data"]) * prior_value[has_prod].values
                                       + out.loc[has_prod, "w_data"] * out.loc[has_prod, "uplift"].values)

    rm = pd.to_numeric(out.get("role_momentum", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0).clip(0.8, 1.2)
    out["true_value"] = out["true_value"] * rm

    out["future_mult"] = out.apply(lambda r: _future_decay_multiplier(r.get("age", np.nan), r["pos"]), axis=1)
    age = pd.to_numeric(out.get("age"), errors="coerce")
    older_mask = (age.notna()) & (((out["pos"] == "RB") & (age > 26)) | ((out["pos"] == "WR") & (age > 28)) | ((out["pos"] == "TE") & (age > 29)) | ((out["pos"] == "QB") & (age > 32)))
    blend = pd.Series(0.30, index=out.index); blend[older_mask] = 0.45
    out["true_value"] = (1.0 - blend) * out["true_value"] + blend * (out["true_value"] * out["future_mult"])

    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.55).clip(0,1)

    ret = out[["name","pos","age","name_key","true_value","confidence","future_mult"]].copy()
    for c in passthru: ret[c] = df[c].values
    return ret

# ---------- market attach ----------
def attach_markets(
    df: pd.DataFrame,
    dp_df: Optional[pd.DataFrame] = None,
    fp_df: Optional[pd.DataFrame] = None,
    w_dp: float = 0.7,
    w_fp: float = 0.3,
    guardrail_lambda: float = 0.6,
    guardrail_floor: float = 0.4,
    guardrail_penalty: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    if "name_key" not in out.columns:
        out["name_key"] = _clean_name_key(out["name"])
    # NEW: canonicalize roster keys (e.g., "hollywood brown" -> "marquise brown")
    out["name_key"] = out["name_key"].apply(_canonicalize_alias_key)

    if dp_df is not None and not dp_df.empty:
        dpm = dp_df.copy()
        dpm["name_key"] = _clean_name_key(dpm["name"])
        # NEW: canonicalize DP keys too
        dpm["name_key"] = dpm["name_key"].apply(_canonicalize_alias_key)

        dpm["pos"] = dpm["pos"].astype(str)
        dpm["market_value"] = pd.to_numeric(dpm["market_value"], errors="coerce")
        out = out.merge(dpm[["name_key","pos","market_value"]], on=["name_key","pos"], how="left")

    if fp_df is not None and not fp_df.empty:
        fpm = fp_df.copy()
        fpm["name_key"] = _clean_name_key(fpm["name"])
        # NEW: canonicalize fallback/ADP keys as well
        fpm["name_key"] = fpm["name_key"].apply(_canonicalize_alias_key)

        fpm["pos"] = fpm["pos"].astype(str)
        fpm["Rank"] = pd.to_numeric(fpm.get("Rank"), errors="coerce")
        fpm["fp_value"] = (1_000.0 / (1.0 + fpm["Rank"].clip(lower=1.0))).fillna(0.0)
        out = out.merge(fpm[["name_key","pos","fp_value"]], on=["name_key","pos"], how="left")

    mv  = _safe_series_default(out, "market_value", 0.0)
    fpv = _safe_series_default(out, "fp_value",   0.0)
    denom = max(w_dp + w_fp, 1e-9)
    out["market_value"] = ((w_dp * mv + w_fp * fpv) / denom).astype(float)

    out["z_true"] = out.groupby("pos")["true_value"].transform(_z)
    out["z_mkt"]  = out.groupby("pos")["market_value"].transform(_z)
    out["edge"]   = out["true_value"] - out["market_value"]
    out["edge_z"] = out["z_true"] - out["z_mkt"]

    out["mv_pct"] = out.groupby("pos")["market_value"].rank(pct=True).fillna(0.0)
    scale = guardrail_floor + guardrail_lambda * out["mv_pct"].clip(0.0, 1.0)
    if guardrail_penalty and guardrail_penalty > 0:
        scale = scale - guardrail_penalty * (1.0 - out["mv_pct"].clip(0.0, 1.0))
    out["edge_z_adj"] = out["edge_z"] * scale

    elite_mask = out["mv_pct"] >= 0.90
    out.loc[elite_mask, "edge_z_adj"] *= 0.75

    return out


# ===== Projection Upgrade (Buckets + Caps + Decay + Ramp) =====
from typing import Dict, Tuple
import numpy as np
import pandas as pd

DEFAULT_PROJ_PARAMS: Dict = {
    # enable/disable features
    "rookie_caps": True,
    "injury_decay": True,
    "return_ramp": True,
    "td_spike_guard": False,    # keep False until we wire usage cols
    "inj_dep_guard": False,     # keep False until we wire teammate status

    # NEW: simple blow-up game guard (works with just points)
    "spike_guard": True,
    "spike_ratio": 0.30,     # last game ≥ 35% of season points
    "spike_discount": 0.75,  # multiply projection by this if spike detected

    # blend weights
    "alpha_ewma": 0.65,
    "beta_ppg": 0.35,
    "gamma_prior": 0.25,  # used when blend falls back

    # sample-size weight: w = min(1, games_played / sample_target)
    "sample_target": 5,

    # positional priors (typical startable baseline)
    "pos_prior": {"RB": 12.0, "WR": 11.0, "TE": 8.0, "QB": 18.0, "FLEX": 10.0},

    # rookie/speculative caps when EWMA==PPG==0 and no NFL snaps
    "rookie_cap": {"RB": 6.0, "WR": 5.0, "TE": 4.0, "QB": 8.0},

    # inactive decay (per missed week) and floors
    "inactive_decay": 0.90,
    "pos_floor": {"RB": 7.0, "WR": 6.0, "TE": 5.0, "QB": 10.0},

    # return ramp multipliers for 1st / 2nd game back
    "return_ramp_mult": {1: 0.85, 2: 0.95},
}


def _get_pos(row) -> str:
    p = str(row.get("pos", "")).upper()
    if p in ("RB","WR","TE","QB"): return p
    return "FLEX"
    


def _market_prior(series_mv: pd.Series, series_pos: pd.Series, params: Dict) -> pd.Series:
    """Convert market_value to within-position percentile and map to prior points."""
        # Coerce to Series aligned to series_pos
    if isinstance(series_mv, pd.Series):
        mv = pd.to_numeric(series_mv, errors="coerce")
    else:
        mv = pd.Series(series_mv, index=series_pos.index, dtype=float)

    pos = series_pos.fillna("FLEX").astype(str).str.upper()

    priors = pos.map(lambda x: params["pos_prior"].get(x, params["pos_prior"]["FLEX"]))
    # rank within pos
    df = pd.DataFrame({"mv": mv, "pos": pos, "prior": priors})
    df["mv_pct"] = (
        df.groupby("pos")["mv"]
          .rank(pct=True, method="average")
          .fillna(0.50)  # neutral if missing
    )
    # linear mapping: percent * prior (you can make this nonlinear later)
    return (df["mv_pct"] * df["prior"]).astype(float)


def _compute_baseline(row, params: Dict) -> float:
    ewma = float(row.get("ewma", 0.0) or 0.0)
    ppg  = float(row.get("ppg", 0.0) or 0.0)
    ppg_last = float(row.get("ppg_last_year", 0.0) or 0.0)
    return max(ewma, 0.7*ppg, 0.5*ppg_last, 0.0)


def _cap_if_rookie(prior_points: float, pos: str, params: Dict) -> float:
    cap = params["rookie_cap"].get(pos, params["rookie_cap"]["RB"])
    return min(float(prior_points), float(cap))


def _apply_inactive_decay(baseline: float, weeks_missed: int, pos: str, params: Dict) -> float:
    decay = params["inactive_decay"] ** max(0, int(weeks_missed))
    floor = params["pos_floor"].get(pos, params["pos_floor"]["RB"])
    return max(baseline * decay, floor)


def _apply_return_ramp(baseline: float, weeks_since_return: int, pos: str, params: Dict) -> float:
    mult = params["return_ramp_mult"].get(int(weeks_since_return), 1.0)
    floor = params["pos_floor"].get(pos, params["pos_floor"]["RB"])
    return max(baseline * mult, floor)


def _games_played_weight(gp: float, params: Dict) -> float:
    return float(min(1.0, (gp or 0.0) / float(params["sample_target"])))


def compute_signal_strength(ewma: float, ppg: float, games_played: float, career_snaps: float) -> str:
    """Return 'production' | 'mixed' | 'speculative'"""
    ew = float(ewma or 0.0); pp = float(ppg or 0.0)
    gp = float(games_played or 0.0); cs = float(career_snaps or 0.0)
    if (ew > 0 or pp > 0) and gp >= 3:
        return "production"
    if (ew > 0 or pp > 0) or (cs > 0):
        return "mixed"
    return "speculative"


def project_points(
    df: pd.DataFrame,
    weekly_df: Optional[pd.DataFrame] = None,
    params: Optional[Dict] = None,
    *,
    season: Optional[int] = None,
    week: Optional[int] = None,
    return_signals: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Produce projected_points with three-bucket logic:
      A) no NFL snaps ever -> speculative cap on market prior
      B) has history but inactive -> decay baseline by missed weeks
      C) returning -> ramp baseline over 1–2 games
      else) blend EWMA/PPG with market prior using sample-size weight
    """
    # ---- setup ----
    if params is None:
        params = DEFAULT_PROJ_PARAMS

    # Ensure pos_series exists even if 'pos' is missing
    pos_series = df["pos"].astype(str).str.upper() if "pos" in df.columns else pd.Series("FLEX", index=df.index)

    out = pd.Series(index=df.index, dtype=float)

    # ---- derive helper features ----
    # career_snaps: ensure it's a Series aligned to df.index
    if "career_snaps" in df.columns:
        career_snaps = pd.to_numeric(df["career_snaps"], errors="coerce")
    else:
        career_snaps = pd.Series(np.nan, index=df.index, dtype=float)
    # Force Series (in case something upstream passed a scalar)
    if not isinstance(career_snaps, pd.Series):
        career_snaps = pd.Series(career_snaps, index=df.index, dtype=float)
    

    # Infer career_snaps from weekly_df if missing
    # Infer career_snaps from weekly_df if missing
# Infer career_snaps from weekly_df if missing
    if career_snaps.isna().all() and weekly_df is not None and not weekly_df.empty:
        key = "player_id" if "player_id" in df.columns else ("id" if "id" in df.columns else None)
        if key and key in weekly_df.columns:
            # --- SAFE SNAP PROXY (works even if 'snaps' column is missing) ---
            snaps_series  = pd.to_numeric(weekly_df["snaps"],  errors="coerce") if "snaps"  in weekly_df.columns else pd.Series(0, index=weekly_df.index, dtype=float)
            points_series = pd.to_numeric(weekly_df["points"], errors="coerce") if "points" in weekly_df.columns else pd.Series(0, index=weekly_df.index, dtype=float)

            snaps_by = (
                weekly_df.assign(
                    _snap_proxy=((snaps_series.fillna(0) > 0) | (points_series.fillna(0) > 0)).astype(int)
                )
                .groupby(key)["_snap_proxy"].sum()
            )
            career_snaps = df[key].map(snaps_by).fillna(0)



    # current week
    if week is not None:
        cur_week = int(week)
    else:
        if "week" in df.columns:
            try:
                cur_week = int(pd.to_numeric(df["week"], errors="coerce").max() or 0)
            except Exception:
                cur_week = 0
        else:
            cur_week = 0
            



    # weeks_since_last_snap if we can
    weeks_since_last = pd.Series(0, index=df.index, dtype=float)
    # weeks_since_last if we can
    weeks_since_last = pd.Series(0, index=df.index, dtype=float)
    if weekly_df is not None and not weekly_df.empty:
        key = "player_id" if ("player_id" in df.columns and "player_id" in weekly_df.columns) else (
            "id" if ("id" in df.columns and "id" in weekly_df.columns) else None
        )
        if key is not None and "week" in weekly_df.columns:
            snaps_series  = pd.to_numeric(weekly_df["snaps"],  errors="coerce") if "snaps"  in weekly_df.columns else pd.Series(0, index=weekly_df.index, dtype=float)
            points_series = pd.to_numeric(weekly_df["points"], errors="coerce") if "points" in weekly_df.columns else pd.Series(0, index=weekly_df.index, dtype=float)

            last_w = weekly_df.loc[
                (snaps_series.fillna(0) > 0) | (points_series.fillna(0) > 0)
            ].groupby(key)["week"].max()

            weeks_since_last = cur_week - df[key].map(last_w).fillna(cur_week)


        # last game points (for spike guard)
    last_points = pd.Series(0.0, index=df.index, dtype=float)
    if weekly_df is not None and not weekly_df.empty:
        key = "player_id" if ("player_id" in df.columns and "player_id" in weekly_df.columns) else (
            "id" if ("id" in df.columns and "id" in weekly_df.columns) else None
        )
        if key is not None and "week" in weekly_df.columns and "points" in weekly_df.columns:
            # take the most recent week’s points per player
            _last_pts = (weekly_df.sort_values("week")
                                   .groupby(key)["points"]
                                   .last())
            last_points = df[key].map(_last_pts).fillna(0.0)
    # season total points if available (merged in app.py earlier)
    season_points = _series_or_default(df, "points_total", 0.0)
        



    # market prior
    prior_series = _market_prior(df.get("market_value"), pos_series, params)

    # inputs
    ewma = _series_or_default(df, "ewma", 0.0)
    ppg  = _series_or_default(df, "ppg", 0.0)
    gp   = _series_or_default(df, "games_played", 0.0)

    # bucket decisions
    _cs = pd.to_numeric(career_snaps, errors="coerce")
    if not isinstance(_cs, pd.Series):
        _cs = pd.Series(_cs, index=df.index, dtype=float)
    no_snaps = (_cs.fillna(0.0) <= 0.0)

    no_prod  = (ewma <= 0.0) & (ppg <= 0.0)

    # Prefer ramp over decay when both true
    mask_C = (~no_snaps) & (weeks_since_last.isin([1, 2])) & params.get("return_ramp", True)        # returning
    mask_B = (~no_snaps) & (weeks_since_last >= 1) & (~mask_C) & params.get("injury_decay", True)   # inactive (not returning)
    mask_A = no_snaps & no_prod & params.get("rookie_caps", True)                                   # rookies/no snaps
    mask_blend = ~(mask_A | mask_B | mask_C)

    # A) cap prior (rookies/no-snaps)
    if mask_A.any():
        prior_vals = prior_series.loc[mask_A]
        pos_vals = pos_series.loc[mask_A]
        out.loc[mask_A] = [
            _cap_if_rookie(float(pr), str(p), params) for pr, p in zip(prior_vals.tolist(), pos_vals.tolist())
        ]

    # helper to compute baseline per row
    def _compute_baseline_row(r: pd.Series) -> float:
        ew = float(r.get("ewma") or 0.0)
        pp = float(r.get("ppg") or 0.0)
        pp_last = float(r.get("ppg_last_year") or 0.0)
        return max(ew, 0.7 * pp, 0.5 * pp_last, 0.0)

    # B) inactive decay
    if mask_B.any():
        idx = mask_B[mask_B].index
        base = df.loc[idx].apply(_compute_baseline_row, axis=1)
        posv = pos_series.loc[idx]
        wmiss = weeks_since_last.loc[idx]
        out.loc[idx] = [
            _apply_inactive_decay(float(b), int(wm), str(p), params)
            for b, wm, p in zip(base.tolist(), wmiss.tolist(), posv.tolist())
        ]

    # C) return ramp
    if mask_C.any():
        idx = mask_C[mask_C].index
        base = df.loc[idx].apply(_compute_baseline_row, axis=1)
        posv = pos_series.loc[idx]
        wsince = weeks_since_last.loc[idx]
        out.loc[idx] = [
            _apply_return_ramp(float(b), int(w), str(p), params)
            for b, w, p in zip(base.tolist(), wsince.tolist(), posv.tolist())
        ]

    # Blend for everyone else (active with some data)
    if mask_blend.any():
        idx = mask_blend[mask_blend].index

        def _w(g: float) -> float:
            return float(min(1.0, (g or 0.0) / float(params.get("sample_target", 5))))

        w_sample = gp.loc[idx].apply(_w)
        alpha = float(params.get("alpha_ewma", 0.65))
        beta  = float(params.get("beta_ppg", 0.35))
        gamma = float(params.get("gamma_prior", 0.25))

        base_blend = alpha * ewma.loc[idx] + beta * ppg.loc[idx]
        out.loc[idx] = (w_sample * base_blend) + (
            (1.0 - w_sample) * (gamma * prior_series.loc[idx] + (1.0 - gamma) * base_blend)
        )

    # signal strength
    def _sig(i: int) -> str:
        ew = float(ewma.iloc[i] if i < len(ewma) else 0.0)
        pp = float(ppg.iloc[i] if i < len(ppg) else 0.0)
        g  = float(gp.iloc[i] if i < len(gp) else 0.0)
        cs = float(career_snaps.iloc[i] if i < len(career_snaps) else 0.0)
        if (ew > 0 or pp > 0) and g >= 3:
            return "production"
        if (ew > 0 or pp > 0) or (cs > 0):
            return "mixed"
        return "speculative"

        # --- simple spike dampener (works without targets/snaps) ---
    if params.get("spike_guard", True):
        denom = season_points.replace(0.0, np.nan)
        spike_ratio = (last_points / denom).fillna(0.0)
        # “usage didn’t really rise” proxy: ewma not meaningfully > ppg
        usage_flat = (ewma <= (ppg * 1.05))
        spike_mask = (spike_ratio >= float(params.get("spike_ratio", 0.35))) & usage_flat
        if spike_mask.any():
            out.loc[spike_mask] = out.loc[spike_mask] * float(params.get("spike_discount", 0.85))

        # --- two-week dominance dampener (catches Noah Brown/Gus streaks) ---
    if params.get("spike_guard", True) and (weekly_df is not None) and (not weekly_df.empty):
        key = "player_id" if ("player_id" in df.columns and "player_id" in weekly_df.columns) else (
            "id" if ("id" in df.columns and "id" in weekly_df.columns) else None
        )
        if key is not None and "week" in weekly_df.columns and "points" in weekly_df.columns:
            # last two weeks sum
            w2 = (weekly_df.sort_values("week")
                           .groupby(key)["points"]
                           .apply(lambda s: s.tail(2).sum()))
            last2 = df[key].map(w2).fillna(0.0)
            denom2 = season_points.replace(0.0, np.nan)
            two_wk_ratio = (last2 / denom2).fillna(0.0)

            # if last 2 weeks are >= 60% of all season points, and EWMA not > PPG, damp harder
            bursty = (two_wk_ratio >= 0.60) & (ewma <= (ppg * 1.05))
            if bursty.any():
                out.loc[bursty] = out.loc[bursty] * 0.70

    

    sig = pd.Series([_sig(i) for i in range(len(df))], index=df.index, dtype=object)

    return (out.fillna(0.0), sig) if return_signals else out.fillna(0.0)
