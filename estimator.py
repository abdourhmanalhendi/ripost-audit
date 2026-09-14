"""Empirical epsilon estimation for differential-privacy auditing.

The audit follows the standard hypothesis-testing view of DP.  A mechanism M
satisfies (eps, delta)-DP if, for every pair of neighbouring datasets D ~ D'
and every measurable output set S,

    P[M(D) in S]  <=  exp(eps) * P[M(D') in S] + delta.

Given an event S we can therefore *lower bound* the privacy loss actually
exhibited by an implementation:

    eps_emp  =  log( (p - delta) / q ),      p = P[M(D) in S],  q = P[M(D') in S].

Because p and q are estimated from a finite number of runs, using the raw
frequencies would produce an optimistic number that is partly sampling noise.
We therefore replace p by a *lower* confidence bound and q by an *upper*
confidence bound (Clopper-Pearson, exact for the binomial), so that the
resulting eps_emp is a statistically valid lower bound at the chosen
confidence level.

If eps_emp > eps_claimed, the implementation leaks more than it promises.
If eps_emp <= eps_claimed, the audit is inconclusive: it means *this* attack
did not find a violation, never that the implementation is correct.

References
----------
Ding, Wang, Wang, Zhang, Kifer. "Detecting Violations of Differential
    Privacy." ACM CCS 2018.
Jagielski, Ullman, Oprea. "Auditing Differentially Private Machine Learning:
    How Private is Private SGD?" NeurIPS 2020.
Bichsel, Steffen, Bogunovic, Vechev. "DP-Sniper: Black-Box Discovery of
    Differential Privacy Violations using Classifiers." IEEE S&P 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import beta as beta_dist


# --------------------------------------------------------------------------
# Clopper-Pearson binomial confidence bounds
# --------------------------------------------------------------------------

def cp_lower(successes: int, trials: int, alpha: float) -> float:
    """One-sided lower Clopper-Pearson bound at confidence 1 - alpha."""
    if trials <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(beta_dist.ppf(alpha, successes, trials - successes + 1))


def cp_upper(successes: int, trials: int, alpha: float) -> float:
    """One-sided upper Clopper-Pearson bound at confidence 1 - alpha."""
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, successes + 1, trials - successes))


# --------------------------------------------------------------------------
# Epsilon lower bound from a single event
# --------------------------------------------------------------------------

@dataclass
class EpsEstimate:
    """Result of estimating eps from one event S."""

    eps_emp: float
    p_hat: float
    q_hat: float
    p_lower: float
    q_upper: float
    n_d: int
    n_dprime: int
    direction: str          # "D>D'" or "D'>D"
    threshold: float | None = None
    statistic: str = ""

    def as_row(self) -> dict:
        return {
            "statistic": self.statistic,
            "direction": self.direction,
            "threshold": self.threshold,
            "p_hat": round(self.p_hat, 4),
            "q_hat": round(self.q_hat, 4),
            "p_lower": round(self.p_lower, 4),
            "q_upper": round(self.q_upper, 4),
            "eps_emp": round(self.eps_emp, 4),
            "n_D": self.n_d,
            "n_Dprime": self.n_dprime,
        }


def eps_lower_bound(
    k_d: int,
    n_d: int,
    k_dprime: int,
    n_dprime: int,
    delta: float = 0.0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Lower bound on eps from counts of the event S under D and D'.

    Parameters
    ----------
    k_d, n_d
        Number of runs on D where the event occurred, and total runs on D.
    k_dprime, n_dprime
        Same for D'.
    delta
        The delta of the claimed (eps, delta)-DP guarantee.
    alpha
        Each Clopper-Pearson bound is taken at level alpha, so the pair holds
        jointly with confidence at least 1 - 2*alpha (union bound).

    Returns
    -------
    (eps_emp, p_lower, q_upper)
    """
    p_lower = cp_lower(k_d, n_d, alpha)
    q_upper = cp_upper(k_dprime, n_dprime, alpha)

    numerator = p_lower - delta
    if numerator <= 0.0 or q_upper <= 0.0:
        return 0.0, p_lower, q_upper

    ratio = numerator / q_upper
    if ratio <= 1.0:
        return 0.0, p_lower, q_upper

    return math.log(ratio), p_lower, q_upper


# --------------------------------------------------------------------------
# Threshold search with an honest train/test split
# --------------------------------------------------------------------------

def _candidate_thresholds(values: np.ndarray, max_candidates: int = 60) -> np.ndarray:
    """Candidate cut points: midpoints between consecutive distinct values."""
    uniq = np.unique(values)
    if uniq.size <= 1:
        return uniq.astype(float)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    if mids.size > max_candidates:
        idx = np.linspace(0, mids.size - 1, max_candidates).astype(int)
        mids = mids[idx]
    return mids.astype(float)


def audit_statistic(
    samples_d: Sequence[float],
    samples_dprime: Sequence[float],
    eps_claimed: float,
    delta: float = 0.0,
    alpha: float = 0.05,
    split: float = 0.5,
    statistic_name: str = "",
    rng: np.random.Generator | None = None,
) -> EpsEstimate:
    """Audit one scalar statistic of the mechanism output.

    The event family is S_t = {T >= t}.  Selecting t and estimating (p, q) on
    the *same* runs would inflate eps_emp through multiple testing, so the runs
    are split: the first half selects the threshold and the direction, the
    second half — untouched during selection — produces the confidence bounds.
    This is what makes the reported number a defensible lower bound rather
    than the maximum of a noisy search.
    """
    rng = rng or np.random.default_rng(0)

    a = np.asarray(samples_d, dtype=float)
    b = np.asarray(samples_dprime, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("empty sample set")

    # shuffle so the split is not correlated with run order
    a = a[rng.permutation(a.size)]
    b = b[rng.permutation(b.size)]

    cut_a = max(1, int(a.size * split))
    cut_b = max(1, int(b.size * split))
    a_sel, a_eval = a[:cut_a], a[cut_a:]
    b_sel, b_eval = b[:cut_b], b[cut_b:]

    if a_eval.size == 0 or b_eval.size == 0:
        raise ValueError("not enough runs to split; increase --runs")

    candidates = _candidate_thresholds(np.concatenate([a_sel, b_sel]))
    if candidates.size == 0:
        candidates = np.array([float(np.mean(np.concatenate([a_sel, b_sel])))])

    # --- selection phase: point estimates only, no confidence bounds ---
    best_t, best_dir, best_score = None, "D>D'", -np.inf
    for t in candidates:
        p = float(np.mean(a_sel >= t))
        q = float(np.mean(b_sel >= t))
        # both directions of the DP inequality
        for direction, num, den in (("D>D'", p, q), ("D'>D", q, p)):
            den_adj = max(den, 1.0 / (2 * max(a_sel.size, b_sel.size)))
            num_adj = num - delta
            if num_adj <= 0:
                continue
            score = math.log(max(num_adj / den_adj, 1e-12))
            if score > best_score:
                best_score, best_t, best_dir = score, float(t), direction

    if best_t is None:
        return EpsEstimate(0.0, 0.0, 0.0, 0.0, 1.0, a_eval.size, b_eval.size,
                           "D>D'", None, statistic_name)

    # --- evaluation phase: held-out runs, exact bounds ---
    k_a = int(np.sum(a_eval >= best_t))
    k_b = int(np.sum(b_eval >= best_t))

    if best_dir == "D>D'":
        eps, p_lo, q_hi = eps_lower_bound(k_a, a_eval.size, k_b, b_eval.size, delta, alpha)
        p_hat, q_hat = k_a / a_eval.size, k_b / b_eval.size
        n1, n2 = a_eval.size, b_eval.size
    else:
        eps, p_lo, q_hi = eps_lower_bound(k_b, b_eval.size, k_a, a_eval.size, delta, alpha)
        p_hat, q_hat = k_b / b_eval.size, k_a / a_eval.size
        n1, n2 = b_eval.size, a_eval.size

    return EpsEstimate(
        eps_emp=eps, p_hat=p_hat, q_hat=q_hat, p_lower=p_lo, q_upper=q_hi,
        n_d=n1, n_dprime=n2, direction=best_dir, threshold=best_t,
        statistic=statistic_name,
    )


# --------------------------------------------------------------------------
# Multiple-statistic correction
# --------------------------------------------------------------------------

def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Split the confidence budget across the statistics we audit.

    We test several statistics on the same runs; without correction the
    largest eps_emp among them is biased upward.  Bonferroni is conservative
    but simple to defend.
    """
    return alpha / max(1, n_tests)


def summarise(estimates: Iterable[EpsEstimate], eps_claimed: float) -> dict:
    ests = list(estimates)
    if not ests:
        return {"eps_claimed": eps_claimed, "eps_emp_max": 0.0, "violation": False}
    best = max(ests, key=lambda e: e.eps_emp)
    return {
        "eps_claimed": eps_claimed,
        "eps_emp_max": best.eps_emp,
        "best_statistic": best.statistic,
        "ratio": (best.eps_emp / eps_claimed) if eps_claimed > 0 else float("nan"),
        "violation": best.eps_emp > eps_claimed,
    }
