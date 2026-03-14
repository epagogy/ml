"""Native statistical tests — zero scipy dependency.

Provides KS 2-sample, chi-squared contingency, paired t-test, and chi2 CDF.
P-values use regularized incomplete gamma/beta via numpy only.
"""

from __future__ import annotations

import math

import numpy as np

# ── Distribution functions ────────────────────────────────────────────────


def _gammaln(x: float) -> float:
    """Log-gamma via Stirling + Lanczos coefficients (15-digit accuracy)."""
    # Lanczos approximation (g=7, n=9)
    coef = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - _gammaln(1 - x)
    x -= 1
    t = coef[0]
    for i in range(1, 9):
        t += coef[i] / (x + i)
    w = x + 7.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(w) - w + math.log(t)


def _regularized_gamma_inc(a: float, x: float) -> float:
    """Lower regularized incomplete gamma function P(a, x).

    Uses series expansion for x < a+1, continued fraction otherwise.
    """
    if x < 0 or a <= 0:
        return 0.0
    if x == 0:
        return 0.0

    gln = _gammaln(a)

    if x < a + 1:
        # Series expansion
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(200):
            ap += 1
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - gln)
    else:
        # Continued fraction (Lentz's method)
        b = x + 1 - a
        c = 1e30
        d = 1.0 / b
        h = d
        for i in range(1, 200):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-15:
                break
        return 1.0 - h * math.exp(-x + a * math.log(x) - gln)


def chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    P(chi2 > x | df) = 1 - regularized_gamma_inc(df/2, x/2).
    """
    if x <= 0:
        return 1.0
    return 1.0 - _regularized_gamma_inc(df / 2.0, x / 2.0)


def chi2_cdf(x: float, df: int) -> float:
    """CDF of chi-squared distribution."""
    if x <= 0:
        return 0.0
    return _regularized_gamma_inc(df / 2.0, x / 2.0)


def _t_sf(t_stat: float, df: int) -> float:
    """Two-tailed p-value from t distribution using incomplete beta.

    p = I_{df/(df+t^2)}(df/2, 1/2)  (regularized incomplete beta).
    """
    if df <= 0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return _regularized_beta_inc(df / 2.0, 0.5, x)


def _regularized_beta_inc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses continued fraction expansion (Lentz's method).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_beta_inc(b, a, 1.0 - x)

    lbeta = _gammaln(a) + _gammaln(b) - _gammaln(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a

    # Continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 200):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-15:
            break

    return front * f


# ── Statistical tests ─────────────────────────────────────────────────────


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (statistic, p_value).
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n_a, n_b = len(a), len(b)

    # Merge and compute empirical CDFs
    combined = np.concatenate([a, b])
    combined.sort()

    cdf_a = np.searchsorted(a, combined, side="right") / n_a
    cdf_b = np.searchsorted(b, combined, side="right") / n_b

    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))

    # Asymptotic p-value: Kolmogorov distribution
    # P(D > d) ≈ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k^2*lambda^2)
    # where lambda = d * sqrt(n_a * n_b / (n_a + n_b))
    n_eff = n_a * n_b / (n_a + n_b)
    lam = d_stat * math.sqrt(n_eff)

    if lam == 0:
        return d_stat, 1.0

    # Kolmogorov limiting distribution (converges fast)
    p = 0.0
    for k in range(1, 100):
        term = 2 * ((-1) ** (k + 1)) * math.exp(-2 * k * k * lam * lam)
        p += term
        if abs(term) < 1e-15:
            break

    p = max(0.0, min(1.0, p))
    return d_stat, p


def chi2_contingency(table: np.ndarray) -> tuple[float, float, int]:
    """Chi-squared test of independence on a contingency table.

    Parameters
    ----------
    table : ndarray, shape (R, C)
        Contingency table of observed frequencies.

    Returns
    -------
    chi2_stat : float
    p_value : float
    dof : int
    """
    table = np.asarray(table, dtype=np.float64)
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    total = table.sum()

    if total == 0:
        return 0.0, 1.0, 0

    expected = np.outer(row_sums, col_sums) / total

    # Avoid division by zero
    mask = expected > 0
    chi2_stat = float(np.sum((table[mask] - expected[mask]) ** 2 / expected[mask]))

    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p = chi2_sf(chi2_stat, dof) if dof > 0 else 1.0

    return chi2_stat, p, dof


def ttest_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test.

    Returns (t_statistic, two_tailed_p_value).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    n = len(diff)

    if n < 2:
        return 0.0, 1.0

    mean_d = np.mean(diff)
    std_d = np.std(diff, ddof=1)

    if std_d == 0:
        return 0.0, 1.0

    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1

    p = _t_sf(abs(t_stat), df)
    return float(t_stat), p
