"""Claim reader for Quarto inline computation.

Load once per paper render. Used in QMD via:

    ```{python}
    #| echo: false
    from claims import c
    ```

Then inline: `{python} c("peek.dz")` renders as `0.93`.

Dot notation navigates the claims.json tree:
    c("peek.dz")          → "0.93"
    c("peek.auc", "+")    → "+0.040"
    c("dup.nb.dz")        → "0.37"
    c("cv_cov.grand_mean", "%")  → "55%"
    c("n_datasets")       → "2,047"
"""
import json
from pathlib import Path

_CLAIMS = json.loads((Path(__file__).parent / "claims.json").read_text())


def c(key, fmt=""):
    """Look up a claim by dot-notation key.

    Args:
        key: Dot-separated path into claims.json (e.g. "peek.dz")
        fmt: Format hint:
            ""   → auto (int gets commas, float gets minimal decimals)
            "+"  → prefix with + for positive values
            "%"  → multiply by 100 and append %
            ".N" → force N decimal places (e.g. ".3" → 3 dp)
    """
    val = _CLAIMS
    for part in key.split("."):
        if isinstance(val, dict):
            val = val[part]
        else:
            raise KeyError(f"Cannot navigate into {type(val)} at '{part}' in key '{key}'")

    if fmt.startswith("%"):
        dp = int(fmt[1:]) if len(fmt) > 1 else 0
        return f"{val * 100:.{dp}f}%"

    if isinstance(val, int):
        return f"{val:,}"

    if isinstance(val, float):
        # Auto-detect appropriate decimal places
        s = f"{val:g}"
        if fmt.startswith("."):
            dp = int(fmt[1:])
            s = f"{val:.{dp}f}"
            # Suppress negative zero: "-0.000" → "0.000"
            if s.lstrip("-") == "0" + "." + "0" * dp:
                s = "0" + "." + "0" * dp
        elif fmt == "+":
            s = f"+{val:g}" if val > 0 else f"{val:g}"
        return s

    return str(val)
