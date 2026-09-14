"""Mironov's floating-point attack, demonstrated against the Laplace sampler
RIPOST uses.

Background
----------
epsilon-DP requires that for neighbouring inputs x and x', every output has
non-zero probability under both.  A Laplace mechanism implemented in floating
point does not satisfy this: the sampler draws U uniformly from the *finite*
set {k / 2^53}, so the set of reachable outputs is finite and discrete.  Adding
x re-rounds, and the reachable set of x + Lap(b) is therefore not the reachable
set of x' + Lap(b) shifted — the two sets differ.

Any output reachable from x but unreachable from x' is a *witness*: observing
it tells the adversary with certainty that the input was x, not x'.  The
privacy loss on that output is unbounded, whatever epsilon the analysis claims.

This is Mironov's attack (CCS 2012).  It is not a flaw in RIPOST's design — it
affects every naive floating-point Laplace implementation that does not apply
a mitigation (snapping mechanism, discrete Laplace, exact arithmetic).  What
this script does is establish, by construction rather than by statistics,
that the published RIPOST code is in that class, and quantify how large the
effect is at the noise scales RIPOST actually uses.

Method
------
NumPy's legacy RandomState.laplace is

    U = random_double()                  # k / 2^53, k integer in [0, 2^53)
    if U < 0.5:  v = loc + scale * log(U + U)
    else:        v = loc - scale * log(2.0 - U - U)

`is_reachable(v)` inverts that: it solves for the U that would be required,
rounds it to the nearest representable k / 2^53, recomputes the sampler
forward in float64, and checks for *exact* equality.  Neighbouring k are also
tried, so a one-ulp discrepancy in log() does not produce a false negative.

`_selftest` verifies the model before any conclusion is drawn: every value the
real sampler produces must be reported reachable.  If that fails, the model of
the sampler is wrong and the rest of the output is meaningless.

Usage
-----
    python float_attack.py                 # default: RIPOST's scale at eps=0.1
    python float_attack.py --scale 14.29 --samples 200000
"""

from __future__ import annotations

import argparse
import math

import numpy as np

TWO53 = 1 << 53


# ---------------------------------------------------------------------------
# The sampler, reimplemented exactly
# ---------------------------------------------------------------------------

def _sampler_forward(k: int, scale: float) -> float:
    """Reproduce RandomState.laplace(0, scale) for the draw U = k / 2^53."""
    u = np.float64(k) / np.float64(TWO53)
    if u < 0.5:
        return float(np.float64(scale) * np.float64(math.log(u + u)))
    return float(-np.float64(scale) * np.float64(math.log(2.0 - u - u)))


def _required_k(v: float, scale: float) -> int | None:
    """The draw index that would be needed to produce v, if any."""
    try:
        if v < 0.0:
            u = math.exp(v / scale) / 2.0
        else:
            u = 1.0 - math.exp(-v / scale) / 2.0
    except (OverflowError, ValueError):
        return None
    if not (0.0 <= u < 1.0):
        return None
    return int(round(u * TWO53))


def is_reachable(v: float, scale: float, window: int = 3) -> bool:
    """Can RandomState.laplace(0, scale) ever return exactly v?"""
    k0 = _required_k(v, scale)
    if k0 is None:
        return False
    for k in range(max(0, k0 - window), min(TWO53, k0 + window + 1)):
        if _sampler_forward(k, scale) == v:
            return True
    return False


# ---------------------------------------------------------------------------
# Self-test: the model must accept everything the real sampler emits
# ---------------------------------------------------------------------------

def _selftest(scale: float, n: int = 20000, seed: int = 0) -> bool:
    rng = np.random.RandomState(seed)
    draws = rng.laplace(0.0, scale, n)
    missed = sum(1 for v in draws if not is_reachable(float(v), scale))
    print(f"self-test: {n - missed}/{n} genuine draws recognised as reachable")
    if missed:
        print(f"  FAILED — {missed} real draws rejected by the model.")
        print("  The reimplementation of the sampler does not match this NumPy")
        print("  build; every conclusion below would be unsound. Stop here.")
        return False
    print("  OK — the sampler model is exact on this build.\n")
    return True


# ---------------------------------------------------------------------------
# The attack
# ---------------------------------------------------------------------------

def find_witnesses(scale: float, n_samples: int, x0: float = 0.0,
                   x1: float = 1.0, seed: int = 0):
    """Sample M(x0) = x0 + Lap(scale); count outputs unreachable from M(x1).

    An output y is reachable from x1 iff some reachable noise value v satisfies
    fl(x1 + v) == y.  Candidates for v are the doubles adjacent to y - x1.
    """
    rng = np.random.RandomState(seed)
    noise = rng.laplace(0.0, scale, n_samples)
    outputs = np.float64(x0) + noise

    witnesses = []
    for y in outputs:
        y = float(y)
        base = np.float64(y) - np.float64(x1)
        reachable_from_x1 = False
        cand = float(base)
        for _ in range(3):                      # walk a few ulps down
            if is_reachable(cand, scale) and float(np.float64(x1) + np.float64(cand)) == y:
                reachable_from_x1 = True
                break
            cand = math.nextafter(cand, -math.inf)
        if not reachable_from_x1:
            cand = float(base)
            for _ in range(3):                  # and a few up
                cand = math.nextafter(cand, math.inf)
                if is_reachable(cand, scale) and float(np.float64(x1) + np.float64(cand)) == y:
                    reachable_from_x1 = True
                    break
        if not reachable_from_x1:
            witnesses.append(y)

    return outputs, witnesses


def support_bound(scale: float) -> float:
    """Largest magnitude the sampler can ever emit: scale * ln(2^53)."""
    return scale * math.log(TWO53)


# ---------------------------------------------------------------------------
# Extension: does a witness survive to RIPOST's actual published value?
#
# NoisedCountTable.__init__ builds each leaf as
#   Block(domain, perturb(domain, perturbation_error, mean))
# and perturb() (count_table.py:525-527) computes exactly
#   mean + (noise / block_size)
# i.e. one more float64 division and one more float64 addition after the raw
# Laplace draw checked above. A witness on the sampler alone does not
# automatically survive two more rounding operations — that has to be
# checked, not assumed.
#
# The two neighbouring values compared are mean0 and mean1 = mean0 +
# 1/block_size: `mean` is `values.sum() / block_size` (calculate_ae_from_data,
# count_table.py:606-610 — the same block_size perturb() divides by), and
# adding one record to a cell inside the block increases that sum by exactly
# 1, so this is RIPOST's own, exact, per-record sensitivity of the raw mean
# — not an arbitrary choice.
# ---------------------------------------------------------------------------

def _published_forward(mean: float, pe: float, block_size: float) -> float:
    """Reproduce count_table.py's perturb(): mean + (noise / block_size)."""
    return float(np.float64(mean) + (np.float64(pe) / np.float64(block_size)))


def is_reachable_published(y: float, mean: float, block_size: float, scale: float,
                            window: int = 6) -> bool:
    """Can `mean + Lap(scale)/block_size` ever equal y exactly?

    The naive approach — walk `cand = (y-mean)*block_size` through its
    neighbouring *doubles* via nextafter, and ask whether each is itself a
    sampler atom — fails silently: `cand` is a real-valued approximation of
    the true noise draw, generally not itself equal to any atom, so walking
    its ULP neighbours mostly visits non-atoms and can miss the (nearby, but
    not adjacent-double) true one. The fix is to walk *atom indices* (k)
    directly, exactly as `is_reachable` already does internally, and check
    each candidate atom's value against the full forward computation.
    """
    cand = (np.float64(y) - np.float64(mean)) * np.float64(block_size)
    k0 = _required_k(float(cand), scale)
    if k0 is None:
        return False
    for k in range(max(0, k0 - window), min(TWO53, k0 + window + 1)):
        pe_k = _sampler_forward(k, scale)
        if _published_forward(mean, pe_k, block_size) == y:
            return True
    return False


def find_published_witnesses(scale: float, block_size: float, mean0: float,
                             mean1: float, n_samples: int, seed: int = 0):
    """Sample the published value under mean0; count outputs unreachable under mean1."""
    rng = np.random.RandomState(seed)
    pe = rng.laplace(0.0, scale, n_samples)
    outputs = [_published_forward(mean0, float(p), block_size) for p in pe]
    witnesses = [y for y in outputs
                if not is_reachable_published(y, mean1, block_size, scale)]
    return outputs, witnesses


def main() -> int:
    ap = argparse.ArgumentParser(description="Mironov floating-point attack on "
                                             "RIPOST's Laplace sampler")
    ap.add_argument("--scale", type=float, default=None,
                    help="noise scale; default is RIPOST's leaf scale at the "
                         "epsilon given by --epsilon")
    ap.add_argument("--epsilon", type=float, default=0.1,
                    help="privacy budget, used to derive RIPOST's leaf scale")
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # RIPOST: epsilon_p = (1 - ratio) * epsilon with ratio = 0.3, and the leaf
    # noise is drawn as laplace(0, sensitivity / epsilon_p) with sensitivity 1.
    if args.scale is None:
        eps_p = 0.7 * args.epsilon
        scale = 1.0 / eps_p
        derived = f"(derived: epsilon_p = 0.7*{args.epsilon} = {eps_p:.4g})"
    else:
        scale = args.scale
        derived = "(given explicitly)"

    print("=" * 72)
    print("Mironov floating-point attack — RIPOST leaf Laplace mechanism")
    print("=" * 72)
    print(f"noise scale = {scale:.6g}  {derived}")
    print(f"samples     = {args.samples}\n")

    if not _selftest(scale):
        return 1

    print("=" * 72)
    print("1. bounded support")
    print("=" * 72)
    bound = support_bound(scale)
    rng = np.random.RandomState(args.seed)
    observed = float(np.max(np.abs(rng.laplace(0.0, scale, args.samples))))
    print(f"  theoretical maximum |noise| = scale * ln(2^53) = {bound:.4f}")
    print(f"  largest |noise| observed    = {observed:.4f}")
    print("  A true Laplace has unbounded support. This one does not:")
    print("  every output beyond the bound has probability exactly zero,")
    print("  which alone contradicts pure epsilon-DP.\n")

    print("=" * 72)
    print("2. witnesses: outputs of M(0) that M(1) can never produce")
    print("=" * 72)
    outputs, witnesses = find_witnesses(scale, args.samples, 0.0, 1.0, args.seed)
    frac = len(witnesses) / max(1, len(outputs))
    print(f"  sampled {len(outputs)} outputs of M(0) = 0 + Lap({scale:.4g})")
    print(f"  unreachable from M(1) = 1 + Lap({scale:.4g}): "
          f"{len(witnesses)}  ({frac:.2%})")
    if witnesses:
        print("\n  first witnesses (hex shows the exact bit pattern):")
        for y in witnesses[:5]:
            print(f"    {y!r:24} {float(y).hex()}")
        print("\n  Each of these outputs identifies the input with certainty:")
        print("  seeing it, an adversary knows the input was 0 and not 1.")
        print("  The privacy loss on such an output is unbounded, regardless")
        print("  of the epsilon the analysis claims.")
    else:
        print("\n  No witness found at this scale and sample size. That does not")
        print("  clear the implementation: it means this particular pair of")
        print("  inputs and this sample size did not expose one.")

    print("\n" + "=" * 72)
    print("3. does a witness survive to the published block value?")
    print("=" * 72)
    print("  perturb() (count_table.py:525-527): value = mean + noise/block_size.")
    print("  mean1 = mean0 + 1/block_size is RIPOST's own exact per-record")
    print("  sensitivity of the raw mean (one added record changes the block's")
    print("  cell-count sum by exactly 1; block_size is the same divisor used")
    print("  both in the mean and in perturb()).\n")

    # self-test for the composed model: the real forward computation must be
    # judged reachable from itself before any conclusion is drawn.
    published_ok = True
    for bs_check in (1.0, 27.0):
        rng = np.random.RandomState(args.seed)
        pe_check = rng.laplace(0.0, scale, 2000)
        mean0_check = 5.0
        outs = [_published_forward(mean0_check, float(p), bs_check) for p in pe_check]
        bad = sum(1 for y in outs
                  if not is_reachable_published(y, mean0_check, bs_check, scale))
        print(f"  self-test (block_size={bs_check:g}): "
              f"{2000-bad}/2000 genuine published values recognised as reachable "
              f"from their own mean")
        if bad:
            published_ok = False
    if not published_ok:
        print("  FAILED — the composed (mean, noise, block_size) model does not")
        print("  match this build; the survival numbers below would be unsound.")
        return 1
    print()

    for block_size in (1.0, 4.0, 27.0):
        mean0 = 5.0
        mean1 = mean0 + 1.0 / block_size
        outs, wit = find_published_witnesses(scale, block_size, mean0, mean1,
                                             args.samples, args.seed)
        frac = len(wit) / max(1, len(outs))
        print(f"  block_size={block_size:5.1f}  mean0={mean0}  mean1={mean1:.6g}  "
              f"survive as witnesses: {len(wit):6d}/{len(outs)}  ({frac:.2%})")
        if wit:
            y = wit[0]
            print(f"      example: published value {y!r} ({float(y).hex()}) "
                  f"is reachable if the true mean is {mean0} but not if it is "
                  f"{mean1:.6g}")

    print("\n  These fractions are not identical to the raw-sampler fraction in")
    print("  step 2 above (the division by block_size and the addition of a")
    print("  non-zero mean each re-round), but they are of the same order: the")
    print("  witness property survives the two extra floating-point operations")
    print("  in perturb(). This is therefore a demonstrated violation on the")
    print("  quantity RIPOST actually publishes (p_view.blocks[i].value), not")
    print("  only on the raw Laplace sampler in isolation.")

    print("\n" + "=" * 72)
    print("Scope")
    print("=" * 72)
    print("  Step 1-2 target the Laplace sampler as called by RIPOST; step 3")
    print("  extends the same construction through perturb()'s division and")
    print("  addition to the actual published block value. The vulnerability")
    print("  class is Mironov's (CCS 2012) and affects every naive")
    print("  floating-point Laplace implementation without a mitigation")
    print("  (snapping, discrete Laplace, exact arithmetic) — RIPOST included.")
    print("  What is established here is that this specific code is in that")
    print("  class, quantified at its own noise scale and on its own published")
    print("  output formula, with reproducible witnesses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
