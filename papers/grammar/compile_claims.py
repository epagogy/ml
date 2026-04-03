"""compile_claims.py — Generate grammar paper claims from landscape paper claims.

The grammar paper references a subset of the landscape study's empirical claims.
This script extracts exactly the keys used by paper2_grammar_v3.qmd,
ensuring the grammar paper never drifts from the landscape paper's data.

Usage:
    python compile_claims.py                          # default
    python compile_claims.py --landscape ../landscape/claims.json
"""

import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent
LANDSCAPE_CLAIMS = _HERE / "../landscape/claims.json"
QMD = _HERE / "paper.qmd"
OUT = _HERE / "claims.json"


def extract_keys(qmd_path):
    """Extract all c("key.path", ...) keys from the paper source."""
    text = qmd_path.read_text()
    return sorted(set(re.findall(r'c\("([^"]+)"', text)))


def resolve(obj, dotpath):
    """Traverse nested dict by dot-separated path."""
    for part in dotpath.split("."):
        obj = obj[part]
    return obj


def set_nested(d, dotpath, value):
    """Set a value in a nested dict by dot-separated path."""
    parts = dotpath.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def main():
    landscape = json.loads(LANDSCAPE_CLAIMS.read_text())
    keys = extract_keys(QMD)

    claims = {
        "_meta": {
            "generator": "compile_claims.py",
            "source": "../landscape/claims.json",
            "note": "Auto-generated subset of landscape claims used by grammar paper. Source of truth: ../landscape/claims.json",
        }
    }

    missing = []
    for key in keys:
        try:
            val = resolve(landscape, key)
            set_nested(claims, key, val)
        except (KeyError, TypeError):
            missing.append(key)

    with open(OUT, "w") as f:
        json.dump(claims, f, indent=2)

    print(f"Extracted {len(keys)} keys from {QMD.name}")
    if missing:
        print(f"WARNING: {len(missing)} keys not found in landscape claims: {missing}")
    else:
        print("All keys resolved.")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--landscape" and i < len(sys.argv) - 1:
            LANDSCAPE_CLAIMS = Path(sys.argv[i + 1])
    main()
