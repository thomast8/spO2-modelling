"""Contract numerics (`Fit.swift`): t90, Normal-Inverse-Gamma, phi, newton1D, weighted median.

No scipy at runtime: the t table is the literal the Swift side carries, so both agree to the bit.
"""

import math

# scipy.stats.t.ppf(0.9, df) for df = 1...30 (index 0 = df 1). Checked against scipy in tests.
T90 = (
    3.0776835371752544,
    1.8856180831641272,
    1.6377443536962093,
    1.533206274058944,
    1.475884048824481,
    1.4397557472651485,
    1.414923927650509,
    1.396815309743865,
    1.3830287383966324,
    1.372183641110336,
    1.3634303180205405,
    1.3562173340232053,
    1.3501712887800552,
    1.3450303744546508,
    1.3406056078504558,
    1.3367571673273149,
    1.3333793897216268,
    1.330390943569909,
    1.3277282090267986,
    1.3253407069850467,
    1.3231878738651728,
    1.321236741613362,
    1.3194602398161621,
    1.3178359336731496,
    1.3163450726738701,
    1.3149718642705175,
    1.3137029128292739,
    1.3125267815926667,
    1.3114336473015513,
    1.3104150253913955,
)
Z90 = 1.2815515655446004


def t90(df):
    """§5: table for integer df, linear between floor and ceil, normal above 30, df 1 below 1."""
    if df > 30:
        return Z90
    if df < 1:
        return T90[0]
    lo = math.floor(df)
    if lo == df:
        return T90[lo - 1]
    a, b = T90[lo - 1], T90[lo]
    return a + (df - lo) * (b - a)


def phi(z):
    """Standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def log_survival(z):
    """log(1 − Φ(z)), §16: log1p below 0 (1 − Φ is near 1 there and a plain log loses the digits
    Newton needs), erfc up to 30, the asymptotic series above."""
    if z < 0:
        return math.log1p(-0.5 * math.erfc(-z / math.sqrt(2.0)))
    if z < 30:
        return math.log(0.5 * math.erfc(z / math.sqrt(2.0)))
    return (
        -z * z / 2 - math.log(z) - 0.5 * math.log(2 * math.pi) + math.log(1 - 1 / z**2 + 3 / z**4)
    )


def mills(z):
    """Inverse Mills ratio φ(z) / (1 − Φ(z)) = exp(−z²/2 − ½·log(2π) − logSurvival(z))."""
    return math.exp(-z * z / 2 - 0.5 * math.log(2 * math.pi) - log_survival(z))


def nig_update(prior, xs):
    """§4 Normal-Inverse-Gamma update. prior = (mu0, k0, a0, b0); returns (mun, kn, an, bn).

    With no rows the posterior is the prior.
    """
    mu0, k0, a0, b0 = prior
    n = len(xs)
    if n == 0:
        return mu0, k0, a0, b0
    xbar = sum(xs) / n
    s = sum((x - xbar) ** 2 for x in xs)
    kn = k0 + n
    mun = (k0 * mu0 + n * xbar) / kn
    an = a0 + n / 2
    bn = b0 + s / 2 + k0 * n * (xbar - mu0) ** 2 / (2 * kn)
    return mun, kn, an, bn


def nig_mean_interval(post):
    """80 % interval on the mean: mun ± t90(2an)·sqrt(bn/(an·kn))."""
    mu, k, a, b = post
    half = t90(2 * a) * math.sqrt(b / (a * k))
    return mu - half, mu, mu + half


def nig_predictive_interval(post):
    """80 % interval for one new row (harness only): scale sqrt(bn(1+kn)/(an·kn))."""
    mu, k, a, b = post
    half = t90(2 * a) * math.sqrt(b * (1 + k) / (a * k))
    return mu - half, mu, mu + half


def nig_rows_to_reach(post, prior_width, rho):
    """§4 rowsToReach: smallest m in 0...200 with 2·t90(2an+m)·sqrt(s2/(kn+m)) ≤ (1−rho)·w_prior."""
    _, k, a, b = post
    s2 = b / a
    for m in range(201):
        if 2 * t90(2 * a + m) * math.sqrt(s2 / (k + m)) <= (1 - rho) * prior_width:
            return m
    return None


def newton1d(fgh, x0, max_steps=50, tol=1e-12, halvings=30):
    """`Fit.newton1D`: maximum of a concave f. `fgh(x)` → (value, first, second derivative).

    Each step is −f′/f″; converged (returns x, f″(x)) when that full step is shorter than `tol`;
    otherwise the step is halved up to `halvings` times until the value does not decrease, with
    four ulp of slack (§16). None on a non-finite value, f″ ≥ 0, no acceptable step, or no
    convergence within `max_steps`.
    """
    x = x0
    cur = fgh(x)
    for _ in range(max_steps):
        f, g, h = cur
        if not (math.isfinite(f) and math.isfinite(g) and math.isfinite(h) and h < 0):
            return None
        step = -g / h
        if abs(step) < tol:
            return x, h
        for _ in range(halvings + 1):
            cand = fgh(x + step)
            if math.isfinite(cand[0]) and cand[0] >= f - 4 * math.ulp(f):
                x += step
                cur = cand
                break
            step /= 2
        else:
            return None
    return None


def weighted_median(values, weights):
    """§8: ascending by value (ties keep row order), smallest value whose cumulative weight
    reaches half the total. None when there is nothing to weigh."""
    pairs = sorted(zip(values, weights, strict=True), key=lambda p: p[0])
    total = sum(w for _, w in pairs)
    if not pairs or total <= 0:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def median(values):
    """Plain median; an even count averages the two middle values."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def logistic_predictive(mean, cov, z):
    """§6: p(full) = σ(m / sqrt(1 + π·v/8)), m = meanᵀz, v = zᵀΣz."""
    m = sum(a * b for a, b in zip(mean, z, strict=True))
    v = sum(z[i] * cov[i][j] * z[j] for i in range(len(z)) for j in range(len(z)))
    return sigmoid(m / math.sqrt(1 + math.pi * v / 8))
