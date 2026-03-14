"""Verify provenance integrity of a saved model.

The provenance receipt is a lightweight chain of evidence:
split parameters → training fingerprint → assess ceremony status.

This is a receipt, not a blockchain. It catches accidental self-deception
(loading a model and re-assessing it, using test data from a different split)
rather than adversarial tampering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def verify(path_or_model: str | Any) -> dict:
    """Verify provenance integrity of a model or saved model file.

    Checks:
    1. Provenance exists (model was trained through ml.split → ml.fit pipeline)
    2. Split receipt is present (cross-language verifiable split identity)
    3. Assess ceremony status (has it been assessed? how many times?)
    4. Chain integrity (training fingerprint → split lineage consistent)

    Parameters
    ----------
    path_or_model : str or Model
        Path to a .ml/.pyml file, or a fitted Model object.

    Returns
    -------
    dict
        Verification report with keys:
        - status: "verified" | "unverified" | "warning"
        - checks: list of individual check results
        - provenance: the raw provenance dict (if present)
        - assess_count: number of times assessed

    Examples
    --------
    >>> report = ml.verify("model.pyml")
    >>> report["status"]
    'verified'
    >>> report["assess_count"]
    1
    >>> ml.verify(model)["status"]
    'verified'
    """
    from ._types import Model, TuningResult

    checks = []

    # Load model if path
    if isinstance(path_or_model, (str, Path)):
        from .io import load
        model = load(str(path_or_model))
    elif isinstance(path_or_model, TuningResult):
        model = path_or_model.best_model
    elif isinstance(path_or_model, Model):
        model = path_or_model
    else:
        return {
            "status": "unverified",
            "checks": [{"check": "type", "ok": False,
                         "detail": f"Expected Model or path, got {type(path_or_model).__name__}"}],
            "provenance": None,
            "assess_count": 0,
        }

    provenance = getattr(model, "_provenance", None) or {}
    assess_count = getattr(model, "_assess_count", 0)

    # Check 1: Provenance exists
    has_provenance = bool(provenance.get("train_fingerprint"))
    checks.append({
        "check": "provenance_exists",
        "ok": has_provenance,
        "detail": "Training fingerprint recorded" if has_provenance
                  else "No provenance — model was not trained through ml.split → ml.fit pipeline",
    })

    # Check 2: Split receipt
    has_receipt = bool(provenance.get("split_receipt"))
    checks.append({
        "check": "split_receipt",
        "ok": has_receipt,
        "detail": f"Split receipt: {provenance['split_receipt']}" if has_receipt
                  else "No split receipt — model may predate v1.0 provenance or was trained on unsplit data",
    })

    # Check 3: Split lineage
    has_lineage = bool(provenance.get("split_id"))
    checks.append({
        "check": "split_lineage",
        "ok": has_lineage,
        "detail": f"Split ID: {provenance['split_id']}" if has_lineage
                  else "No split lineage recorded",
    })

    # Check 4: Assess ceremony
    if assess_count == 0:
        assess_detail = "Not yet assessed — test data untouched"
        assess_ok = True
    elif assess_count == 1:
        assess_detail = "Assessed once — ceremony completed correctly"
        assess_ok = True
    else:
        assess_detail = (
            f"Assessed {assess_count} times — one-shot ceremony violated. "
            "Results after the first assessment are statistically invalid."
        )
        assess_ok = False
    checks.append({
        "check": "assess_ceremony",
        "ok": assess_ok,
        "detail": assess_detail,
    })

    # Overall status
    all_ok = all(c["ok"] for c in checks)
    any_fail = any(not c["ok"] for c in checks)
    if all_ok:
        status = "verified"
    elif any_fail and has_provenance:
        status = "warning"
    else:
        status = "unverified"

    return {
        "status": status,
        "checks": checks,
        "provenance": provenance if has_provenance else None,
        "assess_count": assess_count,
    }
