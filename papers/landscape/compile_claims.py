"""Compile all paper claims from raw experiment data into claims.json.

Single source of truth: every number in the paper traces back to here.
Run: python compile_claims.py
"""
import json
import numpy as np
from pathlib import Path

_HERE = Path(__file__).parent
# Override via: python compile_claims.py --results-dir /path/to/results
RESULTS = _HERE / "data"
OUT = _HERE / "claims.json"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dz(diffs):
    """Cohen's d_z = mean / sd of paired differences."""
    arr = np.array([d for d in diffs if d is not None and not np.isnan(d)])
    if len(arr) < 2:
        return 0.0
    return float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0


def prevalence(diffs):
    """Fraction with positive inflation."""
    arr = np.array([d for d in diffs if d is not None and not np.isnan(d)])
    return float(np.mean(arr > 0)) if len(arr) > 0 else 0.0


def mean_diff(diffs):
    arr = np.array([d for d in diffs if d is not None and not np.isnan(d)])
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def rnd(v, dp=3):
    return round(v, dp)


def main():
    # Load data
    v1 = load_jsonl(RESULTS / "leakage_landscape_v1_final.jsonl")
    v1_ext = load_jsonl(RESULTS / "leakage_landscape_v1_extended.jsonl")
    v2 = load_jsonl(RESULTS / "leakage_landscape_v2.jsonl")
    v3_an = load_jsonl(RESULTS / "v3_an.jsonl")
    v3_ap = load_jsonl(RESULTS / "v3_ap.jsonl")
    v3_ao = load_jsonl(RESULTS / "v3_ao_merged.jsonl")
    phase1 = json.loads((RESULTS / "phase1_results.json").read_text())

    # Filter to successful rows
    v1_ok = [r for r in v1 if r.get("status") == "ok"]
    v1_ext_ok = [r for r in v1_ext if r.get("status") == "ok"]
    v2_ok = [r for r in v2 if r.get("v2_status") == "ok"]

    claims = {
        "_meta": {
            "generator": "compile_claims.py",
            "result_dir": "results",
            "v1_n": len(v1_ok),
            "v2_n": len(v2_ok),
        },
        "n_datasets": len(v1_ok),
        "n_total": len(v1),
    }

    # Corpus descriptives
    n_rows_all = [r["n_rows"] for r in v1_ok if r.get("n_rows")]
    n_feat_all = [r["n_features"] for r in v1_ok if r.get("n_features")]
    claims["corpus"] = {
        "median_n": int(np.median(n_rows_all)),
        "median_p": int(np.median(n_feat_all)),
        "max_n": int(max(n_rows_all)),
        "n_large": sum(1 for n in n_rows_all if n > 100000),
    }

    # V1: Normalization (Exp A)
    norm_lr_diffs = [r["a_lr_gap_diff"] for r in v1_ok if r.get("a_lr_gap_diff") is not None]
    norm_rf_diffs = [r["a_rf_gap_diff"] for r in v1_ok if r.get("a_rf_gap_diff") is not None]
    claims["norm_lr"] = {"dz": rnd(dz(norm_lr_diffs), 3), "auc": rnd(mean_diff(norm_lr_diffs), 5), "n": len(norm_lr_diffs)}
    claims["norm_rf"] = {"dz": rnd(dz(norm_rf_diffs), 3), "auc": rnd(mean_diff(norm_rf_diffs), 5), "n": len(norm_rf_diffs)}

    # V1: Peeking (Exp B)
    for k in [1, 2, 5, 10, 15, 19]:
        key = f"b_infl_k{k}"
        diffs = [r[key] for r in v1_ok if r.get(key) is not None and not np.isnan(r[key])]
        claims[f"peek_k{k}"] = {
            "dz": rnd(dz(diffs)),
            "auc": rnd(mean_diff(diffs), 4),
            "prev": rnd(prevalence(diffs)),
            "n": len(diffs),
        }
    # Primary peeking (k=10)
    claims["peek"] = claims["peek_k10"].copy()

    # V1: Peeking CI
    from scipy import stats as sp_stats_ci
    peek10_diffs = [r["b_infl_k10"] for r in v1_ok
                    if r.get("b_infl_k10") is not None and not np.isnan(r["b_infl_k10"])]
    if peek10_diffs:
        d_val = dz(peek10_diffs)
        n_val = len(peek10_diffs)
        se_dz_val = np.sqrt(1/n_val + d_val**2/(2*n_val))
        ci_lo = d_val - sp_stats_ci.t.ppf(0.975, n_val - 1) * se_dz_val
        ci_hi = d_val + sp_stats_ci.t.ppf(0.975, n_val - 1) * se_dz_val
        claims["peek"]["ci_lo"] = rnd(ci_lo, 2)
        claims["peek"]["ci_hi"] = rnd(ci_hi, 2)

    # V1: Peeking sample size correlation
    peek10 = [(r["b_infl_k10"], r["n_rows"]) for r in v1_ok
              if r.get("b_infl_k10") is not None and not np.isnan(r["b_infl_k10"])]
    if peek10:
        inflations, ns = zip(*peek10)
        log_ns = np.log(np.array(ns))
        r_corr = float(np.corrcoef(inflations, log_ns)[0, 1])
        claims["peek_n_corr"] = rnd(r_corr, 2)

        # Stratified d_z
        for label, lo, hi in [("small", 0, 500), ("medium", 500, 5000), ("large", 5000, 1e9)]:
            strat = [r["b_infl_k10"] for r in v1_ok
                     if r.get("b_infl_k10") is not None and not np.isnan(r["b_infl_k10"])
                     and lo <= r["n_rows"] < hi]
            claims[f"peek_strat_{label}"] = {"dz": rnd(dz(strat)), "n": len(strat)}

        # Discovery/confirmation
        for sp in ["discovery", "confirmation"]:
            sp_diffs = [r["b_infl_k10"] for r in v1_ok
                        if r.get("b_infl_k10") is not None and not np.isnan(r["b_infl_k10"])
                        and r.get("split") == sp]
            claims[f"peek_{sp}"] = {"dz": rnd(dz(sp_diffs)), "n": len(sp_diffs)}

    # V1: Outlier removal (Exp E)
    for pct in [10, 30]:
        lr_diffs = [r[f"e_lr_gap_diff_{pct}"] for r in v1_ok if r.get(f"e_lr_gap_diff_{pct}") is not None]
        rf_diffs = [r[f"e_rf_gap_diff_{pct}"] for r in v1_ok if r.get(f"e_rf_gap_diff_{pct}") is not None]
        # Average LR and RF
        combined = [(lr + rf) / 2 for lr, rf in zip(lr_diffs, rf_diffs) if not np.isnan(lr) and not np.isnan(rf)]
        claims[f"outlier_{pct}"] = {"dz": rnd(dz(combined)), "auc": rnd(mean_diff(combined), 4), "n": len(combined)}

    # V1: Feature encoding (Exp F)
    # F runs on datasets with categorical features (returns None otherwise)
    f_lr_diffs = [r["f_lr_gap_diff"] for r in v1_ok
                  if r.get("f_lr_gap_diff") is not None and not np.isnan(r.get("f_lr_gap_diff", float("nan")))]
    if f_lr_diffs:
        claims["feat_enc_lr"] = {"dz": rnd(dz(f_lr_diffs), 2), "auc": rnd(mean_diff(f_lr_diffs), 4), "n": len(f_lr_diffs)}

    # V1: Binning (Exp T)
    t_lr_diffs = [r["t_lr_gap_diff"] for r in v1_ok
                  if r.get("t_lr_gap_diff") is not None and not np.isnan(r.get("t_lr_gap_diff", float("nan")))]
    if t_lr_diffs:
        claims["binning_lr"] = {"dz": rnd(dz(t_lr_diffs), 2), "auc": rnd(mean_diff(t_lr_diffs), 4), "n": len(t_lr_diffs)}

    # V1: Duplicates (Exp H)
    # NB, LR, XGB, RF from final JSONL; KNN and DT from extended JSONL
    algos_h_final = {
        "nb": "h_nb", "lr": "h_lr", "xgb": "h_xgb", "rf": "h_rf",
    }
    algos_h_ext = {
        "knn": "h_knn", "dt": "h_dt",
    }
    for pct in ["05", "10", "30"]:
        for algo_name, prefix in algos_h_final.items():
            key = f"{prefix}_{pct}"
            diffs = [r[key] for r in v1_ok if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
            if diffs:
                claims[f"dup_{algo_name}_{pct}"] = {
                    "dz": rnd(dz(diffs)),
                    "auc": rnd(mean_diff(diffs), 4),
                    "n": len(diffs),
                }
        for algo_name, prefix in algos_h_ext.items():
            key = f"{prefix}_{pct}"
            diffs = [r[key] for r in v1_ext_ok if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
            if diffs:
                claims[f"dup_{algo_name}_{pct}"] = {
                    "dz": rnd(dz(diffs)),
                    "auc": rnd(mean_diff(diffs), 4),
                    "n": len(diffs),
                }
    # Primary dup references (10%)
    for algo in ["nb", "lr", "xgb", "rf", "knn", "dt"]:
        k = f"dup_{algo}_10"
        if k in claims:
            if "dup" not in claims:
                claims["dup"] = {}
            claims["dup"][algo] = claims[k].copy()

    # V1: Compound (Exp J)
    j_diffs = [r["j_compound_delta"] for r in v1_ok if r.get("j_compound_delta") is not None and not np.isnan(r.get("j_compound_delta", float("nan")))]
    # Sub-additivity: fraction of datasets where compound < sum of individual components
    _sub_count, _sub_total, _sub_ratios = 0, 0, []
    for r in v1_ok:
        _j = r.get("j_compound_delta")
        _a = r.get("a_lr_gap_diff")
        _b = r.get("b_infl_k10")
        _c = r.get("c_theory")
        _h = r.get("h_rf_10")
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [_j, _a, _b, _c, _h]):
            continue
        _indiv_sum = _a + _b + _c + _h
        _sub_total += 1
        if _indiv_sum > 0 and _j < _indiv_sum:
            _sub_count += 1
        if _indiv_sum > 0:
            _sub_ratios.append(_j / _indiv_sum)
    claims["compound"] = {
        "dz": rnd(dz(j_diffs)), "auc": rnd(mean_diff(j_diffs), 4), "n": len(j_diffs),
        "sub_additive_pct": rnd(_sub_count / _sub_total * 100 if _sub_total else 0, 1),
        "median_ratio": rnd(float(np.median(_sub_ratios)) if _sub_ratios else 0, 2),
    }

    # V2: Early stopping (Exp BB)
    bb_diffs = [r["bb_diff"] for r in v2_ok if r.get("bb_diff") is not None]
    claims["early_stop"] = {
        "dz": rnd(dz(bb_diffs)),
        "auc": rnd(mean_diff(bb_diffs), 4),
        "prev": rnd(prevalence(bb_diffs)),
        "n": len(bb_diffs),
    }

    # V2: Screen selection (Exp AQ)
    for k_val in [1, 5, 11]:
        key = f"aq_k{k_val}_optimism"
        diffs = [r[key] for r in v2_ok if r.get(key) is not None and not np.isnan(r.get(key, float("nan")))]
        claims[f"screen_k{k_val}"] = {
            "dz": rnd(dz(diffs)),
            "auc": rnd(mean_diff(diffs), 4),
            "n": len(diffs),
        }
    claims["screen"] = claims["screen_k1"].copy()  # primary

    # V2: Target encoding (Exp AC)
    ac_diffs = [r["ac_diff"] for r in v2_ok if r.get("ac_diff") is not None and not np.isnan(r.get("ac_diff", float("nan")))]
    claims["target_enc"] = {
        "dz": rnd(dz(ac_diffs)),
        "auc": rnd(mean_diff(ac_diffs), 4),
        "prev": rnd(prevalence(ac_diffs)),
        "n": len(ac_diffs),
    }

    # V2: Seed inflation (Exp AI)
    ai_diffs = [r["ai_inflation"] for r in v2_ok if r.get("ai_inflation") is not None]
    claims["seed"] = {
        "dz": rnd(dz(ai_diffs)),
        "auc": rnd(mean_diff(ai_diffs), 4),
        "prev": rnd(prevalence(ai_diffs)),
        "n": len(ai_diffs),
    }

    # V2: Chained estimation (Exp CE)
    ce_diffs = [r["ce_diff"] for r in v2_ok if r.get("ce_diff") is not None]
    claims["chained"] = {"dz": rnd(dz(ce_diffs)), "auc": rnd(mean_diff(ce_diffs), 4), "n": len(ce_diffs)}

    # V2: PCA leakage (Exp AB)
    ab_diffs = [r["ab_diff"] for r in v2_ok if r.get("ab_diff") is not None and not np.isnan(r.get("ab_diff", float("nan")))]
    claims["pca"] = {"dz": rnd(dz(ab_diffs)), "auc": rnd(mean_diff(ab_diffs), 4), "n": len(ab_diffs)}

    # V2: Calibration leakage (Exp AF)
    af_diffs = [r["af_diff"] for r in v2_ok if r.get("af_diff") is not None]
    claims["calibration"] = {"dz": rnd(dz(af_diffs)), "auc": rnd(mean_diff(af_diffs), 4), "n": len(af_diffs)}

    # V2: SMOTE (Exp BA)
    ba_smote = [r["ba_smote_diff"] for r in v2_ok if r.get("ba_smote_diff") is not None and not np.isnan(r.get("ba_smote_diff", float("nan")))]
    ba_random = [r["ba_random_diff"] for r in v2_ok if r.get("ba_random_diff") is not None and not np.isnan(r.get("ba_random_diff", float("nan")))]
    claims["smote"] = {"dz": rnd(dz(ba_smote)), "n": len(ba_smote)}
    claims["random_os"] = {"dz": rnd(dz(ba_random)), "n": len(ba_random)}

    # V2: Stack (Exp AK)
    ak_diffs = [r["ak_diff"] for r in v2_ok if r.get("ak_diff") is not None and not np.isnan(r.get("ak_diff", float("nan")))]
    claims["stack"] = {"dz": rnd(dz(ak_diffs)), "auc": rnd(mean_diff(ak_diffs), 4), "n": len(ak_diffs)}

    # V2: Seed noise floor (Exp AE)
    ae_rf_sds = [r["ae_rf_sd"] for r in v2_ok if r.get("ae_rf_sd") is not None]
    ae_flips = [r["ae_flip"] for r in v2_ok if r.get("ae_flip") is not None]
    claims["noise_floor"] = {
        "rf_sd_median": rnd(float(np.median(ae_rf_sds)), 4),
        "flip_rate": rnd(float(np.mean(ae_flips)), 3),
    }

    # V3: N-scaling (Exp AN)
    an_ok = [r for r in v3_an if r.get("v3_status") == "ok"]
    # Main set: datasets with n_full=2000 (levels 50–2000)
    an_main = [r for r in an_ok if r.get("an_n_full") == 2000]
    # Extension set: datasets with n_full=10000 (levels 50–10000)
    an_ext = [r for r in an_ok if r.get("an_n_full") == 10000]

    n_levels_main = [50, 100, 200, 500, 1000, 2000]
    nscale = {"n_datasets": len(an_ok), "n_levels": n_levels_main,
              "n_main": len(an_main), "n_ext": len(an_ext)}

    # Helper: extract mean ΔAUC for a given n-level from a row's means array
    def _get_level_val(row, exp_type, target_n):
        means_key = f"an_{exp_type}_means"
        levels_key = "an_n_levels"
        if means_key not in row or row[means_key] is None:
            return None
        means = row[means_key]
        levels = row.get(levels_key, n_levels_main)
        if target_n not in levels:
            return None
        idx = levels.index(target_n)
        if idx < len(means) and means[idx] is not None and not np.isnan(means[idx]):
            return means[idx]
        return None

    for exp_type in ["peeking", "seed", "normalize", "oversample"]:
        nscale[exp_type] = {}
        for n in n_levels_main:
            vals = []
            # For n=2000, only use main rows (extension rows have different composition)
            pool = an_main if n == 2000 else an_ok
            for r in pool:
                v = _get_level_val(r, exp_type, n)
                if v is not None:
                    vals.append(v)
            if vals:
                nscale[exp_type][str(n)] = rnd(float(np.mean(vals)), 4)

    # Count oversample datasets (N at first level, broadest pool)
    os_n50 = sum(1 for r in an_ok if _get_level_val(r, "oversample", 50) is not None)
    nscale["n_oversample"] = os_n50

    # Extension: n=5000 and n=10000 from extension rows
    nscale_ext = {"n_datasets": len(an_ext)}
    for exp_type in ["peeking", "seed", "normalize", "oversample"]:
        for target_n in [5000, 10000]:
            vals = []
            for r in an_ext:
                v = _get_level_val(r, exp_type, target_n)
                if v is not None:
                    vals.append(v)
            if vals:
                nscale_ext.setdefault(exp_type, {})[str(target_n)] = rnd(float(np.mean(vals)), 4)
    nscale["ext"] = nscale_ext

    claims["nscale"] = nscale

    # V3: Seed dose-response (Exp AP)
    ap_ok = [r for r in v3_ap if r.get("v3_status") == "ok"]
    seed_dose = {}
    for k in [5, 10, 25, 50, 100]:
        lr_vals = [r[f"ap_lr_inflation_k{k}"] for r in ap_ok
                   if r.get(f"ap_lr_inflation_k{k}") is not None
                   and not np.isnan(r.get(f"ap_lr_inflation_k{k}", float("nan")))]
        rf_vals = [r[f"ap_rf_inflation_k{k}"] for r in ap_ok
                   if r.get(f"ap_rf_inflation_k{k}") is not None
                   and not np.isnan(r.get(f"ap_rf_inflation_k{k}", float("nan")))]
        seed_dose[str(k)] = {
            "lr": rnd(float(np.mean(lr_vals)), 4) if lr_vals else 0.0,
            "rf": rnd(float(np.mean(rf_vals)), 4) if rf_vals else 0.0,
        }
    claims["seed_dose"] = seed_dose
    claims["seed_dose_n"] = len(ap_ok)

    # V3: CV coverage (Exp AO)
    cv_cov = {}
    all_z_covs = []
    all_t_covs = []
    for algo in ["lr", "rf", "dt"]:
        z_vals = [r[f"ao_{algo}_coverage_z"] for r in v3_ao
                  if r.get(f"ao_{algo}_coverage_z") is not None
                  and not np.isnan(r.get(f"ao_{algo}_coverage_z", float("nan")))]
        t_vals = [r[f"ao_{algo}_coverage_t"] for r in v3_ao
                  if r.get(f"ao_{algo}_coverage_t") is not None
                  and not np.isnan(r.get(f"ao_{algo}_coverage_t", float("nan")))]
        if z_vals:
            cv_cov[algo] = {
                "mean": rnd(float(np.mean(z_vals)), 3),
                "t_mean": rnd(float(np.mean(t_vals)), 3) if t_vals else None,
                "n": len(z_vals),
            }
            all_z_covs.extend(z_vals)
            all_t_covs.extend(t_vals)

    if all_z_covs:
        cv_cov["grand_mean"] = rnd(float(np.mean(all_z_covs)), 3)
        cv_cov["grand_t_mean"] = rnd(float(np.mean(all_t_covs)), 3) if all_t_covs else None

    claims["cv_cov"] = cv_cov

    # Phase 2: 6-method CI comparison (Exp AO v2)
    phase2_path = RESULTS / "phase2_ao_v2.jsonl"
    if phase2_path.exists():
        p2 = load_jsonl(phase2_path)
        ci_methods = {}
        for model in ["LR", "DT"]:
            mr = [r for r in p2 if r.get("model") == model]
            model_cov = {}
            for method in ["M1", "M2", "M3", "M4", "M5", "M6"]:
                covs = [r["coverage"][method] for r in mr
                        if r.get("coverage", {}).get(method) is not None]
                if covs:
                    model_cov[method] = rnd(100 * sum(covs) / len(covs), 1)
            ci_methods[model] = {"coverage": model_cov, "n": len(mr)}
        claims["ci_methods"] = ci_methods

    # Meta-regression (Phase 1)
    meta = phase1["summary"]
    claims["meta"] = {
        "alpha_I": {"mean": meta["mean"]["alpha_class[Class I]"],
                     "hdi_lo": phase1["summary"]["hdi_3%"]["alpha_class[Class I]"],
                     "hdi_hi": phase1["summary"]["hdi_97%"]["alpha_class[Class I]"]},
        "alpha_II": {"mean": meta["mean"]["alpha_class[Class II]"],
                      "hdi_lo": phase1["summary"]["hdi_3%"]["alpha_class[Class II]"],
                      "hdi_hi": phase1["summary"]["hdi_97%"]["alpha_class[Class II]"]},
        "alpha_III": {"mean": meta["mean"]["alpha_class[Class III]"],
                       "hdi_lo": phase1["summary"]["hdi_3%"]["alpha_class[Class III]"],
                       "hdi_hi": phase1["summary"]["hdi_97%"]["alpha_class[Class III]"]},
        "beta_log_n": {"mean": meta["mean"]["beta[z_log_n]"],
                        "hdi_lo": phase1["summary"]["hdi_3%"]["beta[z_log_n]"],
                        "hdi_hi": phase1["summary"]["hdi_97%"]["beta[z_log_n]"]},
        "beta_log_p": {"mean": meta["mean"]["beta[z_log_p]"],
                        "hdi_lo": phase1["summary"]["hdi_3%"]["beta[z_log_p]"],
                        "hdi_hi": phase1["summary"]["hdi_97%"]["beta[z_log_p]"]},
        "beta_imbalance": {"mean": meta["mean"]["beta[z_imbalance]"],
                            "hdi_lo": phase1["summary"]["hdi_3%"]["beta[z_imbalance]"],
                            "hdi_hi": phase1["summary"]["hdi_97%"]["beta[z_imbalance]"]},
        "tau_exp": meta["mean"]["tau_exp"],
        "tau_ds": meta["mean"]["tau_ds"],
        "sigma_resid": meta["mean"]["sigma_resid"],
        "tau_ratio": rnd(meta["mean"]["tau_exp"] / meta["mean"]["tau_ds"], 1),
    }
    # MCMC diagnostics from phase1_results.json
    ess_bulk = meta["ess_bulk"]
    claims["meta"]["min_ess_bulk"] = int(min(ess_bulk.values()))
    claims["meta"]["max_rhat"] = float(max(meta["r_hat"].values()))
    # n_obs from the parquet input data
    phase1_data_path = RESULTS / "phase1_data.parquet"
    if phase1_data_path.exists():
        import pandas as _pd
        claims["meta"]["n_obs"] = len(_pd.read_parquet(phase1_data_path))

    # N-scaling floor model
    # Fit d(n) = a * n^(-b) + c vs d(n) = a * n^(-b) on main nscale data
    from scipy.optimize import curve_fit as _curve_fit
    _n_arr = np.array([50, 100, 200, 500, 1000, 2000], dtype=float)
    def _floor_model(x, a, b, c): return a * x**(-b) + c
    def _null_model(x, a, b): return a * x**(-b)
    floor_claims = {}
    for _exp in ["peeking", "seed"]:
        _y = np.array([nscale[_exp][str(int(ni))] for ni in _n_arr])
        _popt_f, _ = _curve_fit(_floor_model, _n_arr, _y, p0=[1, 0.5, 0.04])
        _c_hat = float(_popt_f[2])
        _N = len(_n_arr)
        _ss_mle = float(np.sum((_y - _floor_model(_n_arr, *_popt_f))**2))
        _logL_mle = -_N / 2 * np.log(_ss_mle / _N)
        # Profile likelihood 95% CI for floor parameter c
        _threshold = _logL_mle - 1.92  # chi2(1) 0.95 / 2
        _c_grid = np.linspace(-0.05, 0.15, 1000)
        _profile = []
        for _ci in _c_grid:
            def _model_ci(x, a, b, _c=_ci): return a * x**(-b) + _c
            try:
                _p, _ = _curve_fit(_model_ci, _n_arr, _y, p0=[_popt_f[0], _popt_f[1]], maxfev=5000)
                _ss = float(np.sum((_y - _model_ci(_n_arr, *_p))**2))
                _profile.append(-_N / 2 * np.log(_ss / _N))
            except Exception:
                _profile.append(-1e10)
        _profile = np.array(_profile)
        _above = _profile >= _threshold
        _ci_lo = float(_c_grid[_above][0]) if _above.any() else float("nan")
        _ci_hi = float(_c_grid[_above][-1]) if _above.any() else float("nan")
        floor_claims[_exp] = {
            "c": rnd(_c_hat, 3),
            "ci_lo": rnd(_ci_lo, 3),
            "ci_hi": rnd(_ci_hi, 3),
        }
    claims["floor"] = floor_claims

    # Internal validation (discovery/confirmation)
    # V2 lacks split field — join with V1 by (name, source) to get split assignment
    v1_split_lookup = {(r.get("name"), r.get("source")): r.get("split") for r in v1_ok}
    for sp in ["discovery", "confirmation"]:
        seed_sp = [r["ai_inflation"] for r in v2_ok
                   if r.get("ai_inflation") is not None
                   and v1_split_lookup.get((r.get("name"), r.get("source"))) == sp]
        if seed_sp:
            claims[f"seed_{sp}"] = {"dz": rnd(dz(seed_sp)), "auc": rnd(mean_diff(seed_sp), 4)}

    # Dup discovery/confirmation
    for sp in ["discovery", "confirmation"]:
        dup_sp = [r["h_rf_10"] for r in v1_ok
                  if r.get("h_rf_10") is not None and not np.isnan(r.get("h_rf_10", float("nan")))
                  and r.get("split") == sp]
        if dup_sp:
            claims[f"dup_rf_{sp}"] = {"dz": rnd(dz(dup_sp))}

    # Seed dose-response log fit
    # Compute from raw AP data (not rounded seed_dose) to match figure script
    if ap_ok:
        ks = [5, 10, 25, 50, 100]
        rf_raw_means = []
        for k in ks:
            vals = [r[f"ap_rf_inflation_k{k}"] for r in ap_ok
                    if r.get(f"ap_rf_inflation_k{k}") is not None
                    and not np.isnan(r.get(f"ap_rf_inflation_k{k}", float("nan")))]
            rf_raw_means.append(float(np.mean(vals)) if vals else 0.0)
        log_ks = np.log(np.array(ks, dtype=float))
        rf_arr = np.array(rf_raw_means)
        # Linear fit: y = a * log(K) + b
        coeffs = np.polyfit(log_ks, rf_arr, 1)
        y_pred = np.polyval(coeffs, log_ks)
        ss_res = np.sum((rf_arr - y_pred) ** 2)
        ss_tot = np.sum((rf_arr - np.mean(rf_arr)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        claims["seed_logfit"] = {
            "a": rnd(float(coeffs[0]), 5),
            "b": rnd(float(coeffs[1]), 5),
            "r2": rnd(float(r2), 4),
        }

    # Moderator correlations
    # V2 lacks dataset characteristics — join with V1 by (name, source)
    from scipy import stats as sp_stats
    v1_lookup = {(r.get("name"), r.get("source")): r for r in v1_ok}
    moderators = {}
    for exp_key, exp_field in [("ba", "ba_smote_diff"), ("ai", "ai_inflation"),
                                ("aq", "aq_k1_optimism"), ("bb", "bb_diff")]:
        mod_vals = {}
        for mod_name, v1_field in [("p_over_n", "p_over_n"), ("log_n", "n_rows"), ("imbalance", "imbalance")]:
            pairs = []
            for r2 in v2_ok:
                r1 = v1_lookup.get((r2.get("name"), r2.get("source")))
                if (r1 and r2.get(exp_field) is not None
                        and not np.isnan(r2.get(exp_field, float("nan")))):
                    mod_val = r1.get(v1_field)
                    if mod_val is not None and not np.isnan(mod_val):
                        x = r2[exp_field]
                        y = np.log(mod_val) if mod_name == "log_n" else mod_val
                        pairs.append((x, y))
            if len(pairs) > 10:
                xs, ys = zip(*pairs)
                rho, _ = sp_stats.spearmanr(xs, ys)
                mod_vals[mod_name] = rnd(float(rho), 3)
        moderators[exp_key] = mod_vals
    claims["moderators"] = moderators

    # Robustness checks
    robust = {}

    # Bootstrap floor CI (10K resamples over datasets)
    for _exp in ["peeking", "seed"]:
        _y_main = {n: [] for n in [50, 100, 200, 500, 1000, 2000]}
        for r in an_ok:
            if r.get("an_n_full") != 2000:
                continue
            levels = r.get("an_n_levels", [])
            means = r.get(f"an_{_exp}_means", [])
            for n in _y_main:
                if n in levels:
                    idx = levels.index(n)
                    if idx < len(means) and means[idx] is not None:
                        _y_main[n].append(means[idx])

        ns = sorted(_y_main.keys())
        rng = np.random.RandomState(42)
        boot_floors = []
        for _ in range(10000):
            boot_ys = []
            for n in ns:
                v = _y_main[n]
                boot_sample = rng.choice(v, size=len(v), replace=True)
                boot_ys.append(float(np.mean(boot_sample)))
            try:
                bp, _ = _curve_fit(_floor_model, np.array(ns, dtype=float),
                                   np.array(boot_ys), p0=[1, 0.5, 0.04],
                                   maxfev=5000)
                if -0.5 < bp[2] < 0.5:
                    boot_floors.append(bp[2])
            except Exception:
                pass
        boot_floors = np.array(boot_floors)
        robust[f"boot_{_exp}_ci_lo"] = rnd(float(np.percentile(boot_floors, 2.5)), 3)
        robust[f"boot_{_exp}_ci_hi"] = rnd(float(np.percentile(boot_floors, 97.5)), 3)

    # Two-parameter floor model (b=0.5 fixed)
    _n_arr2 = np.array([50, 100, 200, 500, 1000, 2000], dtype=float)
    def _floor2(x, a, c): return a * x**(-0.5) + c
    for _exp in ["peeking", "seed"]:
        _y2 = np.array([nscale[_exp][str(int(ni))] for ni in _n_arr2])
        try:
            _p2, _ = _curve_fit(_floor2, _n_arr2, _y2, p0=[1, 0.04])
            robust[f"floor2_{_exp}_c"] = rnd(float(_p2[1]), 3)
        except Exception:
            pass

    # d_z across n-levels
    for _exp in ["peeking", "seed"]:
        dz_by_n = {}
        for n in [50, 100, 200, 500, 1000, 2000]:
            vals = []
            for r in an_ok:
                if r.get("an_n_full") != 2000:
                    continue
                levels = r.get("an_n_levels", [])
                means = r.get(f"an_{_exp}_means", [])
                if n in levels:
                    idx = levels.index(n)
                    if idx < len(means) and means[idx] is not None:
                        vals.append(means[idx])
            if len(vals) > 1:
                dz_by_n[str(n)] = rnd(float(np.mean(vals) / np.std(vals, ddof=1)), 2)
        robust[f"dz_nscale_{_exp}"] = dz_by_n
        vals_list = list(dz_by_n.values())
        if vals_list:
            robust[f"dz_nscale_{_exp}_min"] = rnd(min(vals_list), 2)
            robust[f"dz_nscale_{_exp}_max"] = rnd(max(vals_list), 2)

    # CV coverage by dataset size
    ao_ok_robust = [r for r in v3_ao if r.get("ao_lr_coverage_z") is not None]
    cov_by_size = {}
    for lo, hi, label in [(0, 200, "lt200"), (200, 500, "200_500"),
                           (500, 1000, "500_1k"), (1000, 5000, "1k_5k"),
                           (5000, 999999, "gte5k")]:
        sub = [r for r in ao_ok_robust if lo <= r.get("n_rows", 0) < hi]
        if sub:
            lr = [r["ao_lr_coverage_z"] for r in sub
                  if r.get("ao_lr_coverage_z") is not None
                  and not np.isnan(r["ao_lr_coverage_z"])]
            if lr:
                cov_by_size[label] = rnd(float(np.mean(lr)), 2)
    robust["cv_cov_by_size"] = cov_by_size

    # Survivorship bias check (intersection vs full at n=200)
    for _exp in ["peeking", "seed"]:
        all_200, inter_200 = [], []
        for r in an_ok:
            if r.get("an_n_full") != 2000:
                continue
            levels = r.get("an_n_levels", [])
            means = r.get(f"an_{_exp}_means", [])
            if 200 in levels:
                idx = levels.index(200)
                if idx < len(means) and means[idx] is not None:
                    all_200.append(means[idx])
                    has_all = all(
                        n in levels and levels.index(n) < len(means)
                        and means[levels.index(n)] is not None
                        for n in [50, 100, 200, 500, 1000, 2000])
                    if has_all:
                        inter_200.append(means[idx])
        if all_200 and inter_200:
            robust[f"survivor_{_exp}_diff"] = rnd(
                float(np.mean(inter_200) - np.mean(all_200)), 4)

    claims["robust"] = robust

    # Write
    with open(OUT, "w") as f:
        json.dump(claims, f, indent=2)

    print(f"Written {len(claims)} top-level keys to {OUT}")
    print(f"V1: {len(v1_ok)} datasets, V2: {len(v2_ok)} datasets")
    print("Key values:")
    print(f"  peek.dz = {claims['peek']['dz']}")
    print(f"  peek.auc = {claims['peek']['auc']}")
    print(f"  seed.dz = {claims['seed']['dz']}")
    print(f"  early_stop.dz = {claims['early_stop']['dz']}")
    print(f"  screen.dz = {claims['screen']['dz']}")
    print(f"  target_enc.dz = {claims['target_enc']['dz']}")
    print(f"  meta.tau_ratio = {claims['meta']['tau_ratio']}")
    print(f"  cv_cov.grand_mean = {claims.get('cv_cov', {}).get('grand_mean', 'N/A')}")


if __name__ == "__main__":
    import sys
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--results-dir" and i < len(sys.argv) - 1:
            RESULTS = Path(sys.argv[i + 1])
    main()
