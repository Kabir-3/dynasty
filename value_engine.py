# value_engine.py — V4.1
# VAR + role momentum + 2-year decay + Bayesian blend + elite shrink in market attach.

from typing import Optional
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

        parts.append(agg.groupby(["name_key","pos"], as_index=False).apply(_comp, include_groups=False))

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

    stats = w.groupby(["name_key","pos"], as_index=False).apply(_agg_player, include_groups=False)
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

