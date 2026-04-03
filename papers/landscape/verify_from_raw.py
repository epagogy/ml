"""Verify paper claims directly from raw JSONL. No dependencies beyond numpy.

Usage: python verify_from_raw.py
Expected output: all PASS.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
CLAIMS = json.load(open(HERE / "claims.json"))


def dz(diffs):
    d = np.array([x for x in diffs if x is not None and not np.isnan(x)])
    return float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 1 else 0.0


def check(name, expected, got, tol=0.002):
    ok = abs(expected - got) <= tol
    print(f"  {name}: expected={expected}, got={round(got, 4)}, {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    v1 = [json.loads(l) for l in open(HERE / "data/leakage_landscape_v1_final.jsonl")]
    v1_ok = [r for r in v1 if r.get("status") == "ok"]

    v2 = [json.loads(l) for l in open(HERE / "data/leakage_landscape_v2.jsonl")]
    v2_ok = v2  # V2 has no status field; all rows are usable

    v3_an = [json.loads(l) for l in open(HERE / "data/v3/v3_an.jsonl")]
    v3_an_ok = [r for r in v3_an if r.get("v3_status") == "ok"]

    passed = 0
    total = 0

    print("=== Dataset counts ===")
    total += 1; passed += check("n_datasets", CLAIMS["n_datasets"], len(v1_ok), tol=0)
    total += 1; passed += check("corpus.median_n", CLAIMS["corpus"]["median_n"],
                                 int(np.median([r["n_rows"] for r in v1_ok])), tol=0)

    print("\n=== Class I: Estimation ===")
    d = [r["a_lr_gap_diff"] for r in v1_ok if r.get("a_lr_gap_diff") is not None]
    total += 1; passed += check("norm_lr.dz", CLAIMS["norm_lr"]["dz"], dz(d))
    total += 1; passed += check("norm_lr.auc", CLAIMS["norm_lr"]["auc"], float(np.mean(d)), tol=0.0005)

    print("\n=== Class II: Peeking ===")
    d = [r["b_infl_k10"] for r in v1_ok if r.get("b_infl_k10") is not None]
    total += 1; passed += check("peek.dz", CLAIMS["peek"]["dz"], dz(d))
    total += 1; passed += check("peek.auc", CLAIMS["peek"]["auc"], float(np.mean(d)), tol=0.001)

    print("\n=== Class II: Seed ===")
    d = [r["ai_inflation"] for r in v2_ok
         if r.get("ai_inflation") is not None and not np.isnan(r["ai_inflation"])]
    total += 1; passed += check("seed.dz", CLAIMS["seed"]["dz"], dz(d))

    print("\n=== Class II: Screen ===")
    d = [r["aq_k1_optimism"] for r in v2_ok
         if r.get("aq_k1_optimism") is not None and not np.isnan(r["aq_k1_optimism"])]
    total += 1; passed += check("screen.dz", CLAIMS["screen"]["dz"], dz(d))

    print("\n=== N-scaling: dataset counts ===")
    main_2k = [r for r in v3_an_ok if r.get("an_n_full") == 2000]
    ext_10k = [r for r in v3_an_ok if r.get("an_n_full") == 10000]
    total += 1; passed += check("nscale.n_main", CLAIMS["nscale"]["n_main"], len(main_2k), tol=0)
    total += 1; passed += check("nscale.ext.n_datasets", CLAIMS["nscale"]["ext"]["n_datasets"],
                                 len(ext_10k), tol=0)

    print(f"\n{'='*40}")
    print(f"RESULT: {passed}/{total} checks passed")
    if passed == total:
        print("ALL CLAIMS VERIFIED FROM RAW DATA.")
    else:
        print(f"WARNING: {total - passed} checks FAILED.")


if __name__ == "__main__":
    main()
