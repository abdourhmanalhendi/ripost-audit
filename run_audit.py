"""Empirical differential-privacy audit of RIPOST.

Usage
-----
    # 1. always run this first
    python mechanism.py --repo ../RIPOST --selftest

    # 2. sanity check: a 50-record difference must be detectable
    python run_audit.py --repo ../RIPOST --runs 200 --n-canary 50 --epsilon 1.0

    # 3. the real audit
    python run_audit.py --repo ../RIPOST --runs 1000 --epsilon 0.1 0.5 1.0

What it does
------------
For each privacy budget and each canary placement, the mechanism is executed
`--runs` times on D and `--runs` times on D' (which differ by a single
record).  Several scalar statistics of the published view are extracted, and
for each one we compute a statistically valid lower bound on the privacy loss
the implementation exhibits.  If that bound exceeds the claimed epsilon, the
implementation leaks more than it promises.

Interpreting the outcome
------------------------
eps_emp > eps_claimed
    A violation, at the stated confidence level.  Report it with the exact
    dataset pair, seed and statistic so it can be reproduced.
eps_emp <= eps_claimed
    Inconclusive.  It means these attacks, at this sample size, did not find a
    violation.  It is *not* evidence that the implementation is correct, and
    the report must say so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from estimator import audit_statistic, bonferroni_alpha, summarise
from mechanism import run_mechanism
from neighbors import build_pair

# Statistics extracted from each published view.  Each is a quantity an
# adversary can read off the released output.
STATISTICS = {
    "n_blocks":        lambda o: float(o.n_blocks),
    "max_depth":       lambda o: float(o.max_depth),
    "canary_mean":     lambda o: float(o.canary_block_mean),
    "canary_volume":   lambda o: float(o.canary_block_volume),
    "sum_estimate":    lambda o: float(np.nansum(o.block_means * o.block_volumes)),
    "max_block_mean":  lambda o: float(np.nanmax(o.block_means)) if o.block_means.size else 0.0,
}

# Runtime is observable in a deployment but is not part of the released view;
# it is collected separately and reported on its own.
TIMING_STATISTIC = ("runtime_s", lambda o: float(o.runtime_s))


def collect(pair, epsilon, runs, repo, base_seed, verbose=True, hp=None):
    """Execute the mechanism `runs` times on each of D and D'."""
    obs = {"D": [], "Dprime": []}
    t0 = time.time()

    for i in range(runs):
        for label, data in (("D", pair.d), ("Dprime", pair.d_prime)):
            # distinct seeds per run and per dataset: we are sampling the
            # mechanism's output distribution, not comparing single draws
            seed = base_seed + i * 2 + (0 if label == "D" else 1)
            out = run_mechanism(data, pair.attrs, pair.shape, epsilon, seed,
                                repo, canary_cell=pair.canary_cell, hp=hp)
            obs[label].append(out)

        if verbose and (i + 1) % max(1, runs // 10) == 0:
            done = i + 1
            rate = (time.time() - t0) / done
            eta = rate * (runs - done)
            print(f"    {done:5d}/{runs} runs   eta {eta/60:5.1f} min", flush=True)

    return obs


def audit_pair(pair, epsilon, runs, repo, base_seed, alpha, delta,
               include_timing=False, hp=None):
    obs = collect(pair, epsilon, runs, repo, base_seed, hp=hp)

    stats = dict(STATISTICS)
    if include_timing:
        stats[TIMING_STATISTIC[0]] = TIMING_STATISTIC[1]

    alpha_corrected = bonferroni_alpha(alpha, len(stats))
    rng = np.random.default_rng(base_seed)

    estimates = []
    for name, fn in stats.items():
        a = np.array([fn(o) for o in obs["D"]], dtype=float)
        b = np.array([fn(o) for o in obs["Dprime"]], dtype=float)

        # a statistic that is constant or undefined carries no signal
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        if not finite:
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        if np.unique(np.concatenate([a, b])).size <= 1:
            continue

        est = audit_statistic(a, b, epsilon, delta=delta, alpha=alpha_corrected,
                              statistic_name=name, rng=rng)
        estimates.append(est)

    return estimates, obs


def _guard_against_reentry() -> None:
    """Stop child processes from re-executing the whole campaign.

    RIPOST.run creates a multiprocessing.Manager and a Pool on every call.
    Under the 'spawn' / 'forkserver' start methods a child re-imports the main
    module, and any code that runs at import time runs again in every child —
    which turns one audit into an exponential fork storm.

    Two defences: force 'fork' (children inherit memory, nothing is re-imported)
    and, if that is refused by the platform, mark the environment so that a
    re-entrant child exits immediately instead of starting its own campaign.
    """
    import multiprocessing as mp
    import os

    # Line-buffer our own output.  When stdout is redirected to a file Python
    # switches to block buffering, so anything printed but not yet flushed sits
    # in the buffer; RIPOST then forks worker processes, each inherits a *copy*
    # of that buffer, and each rewrites it on exit.  The result is every header
    # duplicated once per forked child.  Line buffering keeps the buffer empty
    # across fork points and removes the duplication entirely.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if os.environ.get("_RIPOST_AUDIT_RUNNING") == "1":
        sys.exit(0)
    os.environ["_RIPOST_AUDIT_RUNNING"] = "1"

    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        print("warning: could not select the 'fork' start method; "
              "runs may be slower.", file=sys.stderr)


def main() -> int:
    _guard_against_reentry()

    ap = argparse.ArgumentParser(description="Empirical DP audit of RIPOST")
    ap.add_argument("--repo", type=Path, default=Path("./RIPOST"),
                    help="path to the cloned RIPOST repository")
    ap.add_argument("--epsilon", type=float, nargs="+", default=[1.0],
                    help="claimed privacy budget(s) to audit")
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--runs", type=int, default=500,
                    help="executions per dataset (half are held out for the bounds)")
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--shape", type=int, nargs="+", default=[8, 8, 4])
    ap.add_argument("--placements", nargs="+",
                    default=["boundary", "sparse", "dense"])
    ap.add_argument("--n-canary", type=int, default=1,
                    help="records distinguishing D from D' (1 = real audit)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="confidence level is 1 - alpha, before correction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-timing", action="store_true",
                    help="also audit wall-clock runtime (side-channel, not covered by eps)")
    ap.add_argument("--null-test", action="store_true",
                    help="control experiment: compare D against itself. Any audit "
                         "reporting eps_emp > 0 here is measuring its own bias, "
                         "not a privacy leak. Run this before believing any result.")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--ratio", type=float, default=None,
                    help="override RIPOST's 'ratio' hyperparameter (paper's "
                         "alpha, share of budget given to decomposition vs. "
                         "leaf perturbation). Default reproduces the authors' "
                         "own setup (0.3). Used to approximately isolate a "
                         "sub-mechanism: near 0 starves the decomposition "
                         "(cc/ss) of budget and gives the leaf Laplace "
                         "release nearly all of epsilon; near 1 does the "
                         "opposite.")
    ap.add_argument("--dump-raw", action="store_true",
                    help="also write one row per run (n_blocks, runtime_s, ...) "
                         "to <out>/raw_<epsilon>_<placement>.csv, for analyses "
                         "(e.g. timing correlation) that need more than the "
                         "summary statistics.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    attrs = tuple("abcdefgh"[: len(args.shape)])
    shape = tuple(args.shape)

    if args.n_canary != 1:
        print(f"NOTE: n_canary={args.n_canary} — this is a harness sanity check,\n"
              f"      not a differential-privacy audit (the unit of privacy is 1 record).\n")

    rows = []
    for epsilon in args.epsilon:
        for placement in args.placements:
            pair = build_pair(placement=placement, n_rows=args.rows,
                              shape=shape, attrs=attrs, seed=args.seed,
                              n_canary=args.n_canary)

            if args.null_test:
                # D' := D.  The two sample sets are then draws from the *same*
                # distribution, so a correct audit must report eps_emp ~ 0.
                # Anything larger is bias in the harness — threshold selection
                # leaking across the split, a statistic that is not actually a
                # function of the released output, or too few held-out runs.
                pair.d_prime = pair.d.copy()
                pair.placement = f"NULL-TEST (D vs D, {placement})"

            print("=" * 72)
            print(f"epsilon={epsilon}   {pair.describe()}")
            print("=" * 72)

            hp_override = {"ratio": args.ratio} if args.ratio is not None else None
            try:
                estimates, obs = audit_pair(
                    pair, epsilon, args.runs, args.repo, args.seed * 1000 + 1,
                    args.alpha, args.delta, args.include_timing, hp=hp_override)
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                print("  run `python mechanism.py --selftest` first.")
                return 1

            if args.dump_raw:
                raw_rows = []
                for label, outs in obs.items():
                    for o in outs:
                        raw_rows.append({
                            "label": label, "n_blocks": o.n_blocks,
                            "max_depth": o.max_depth, "runtime_s": o.runtime_s,
                            "canary_mean": o.canary_block_mean,
                            "canary_volume": o.canary_block_volume,
                            "sum_estimate": float(np.nansum(o.block_means * o.block_volumes)),
                        })
                safe_placement = pair.placement.replace(" ", "_").replace("/", "_")
                raw_path = args.out / f"raw_{epsilon}_{safe_placement}.csv"
                pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
                print(f"  raw per-run stats written to {raw_path}")

            for est in estimates:
                row = est.as_row()
                row.update(epsilon_claimed=epsilon, placement=pair.placement,
                           canary_cell=str(pair.canary_cell))
                rows.append(row)

            # Group privacy: datasets differing in k records are k steps apart
            # in the neighbouring relation, so the guarantee that applies is
            # k * epsilon, not epsilon.  Comparing against epsilon when k > 1
            # would manufacture a violation out of correct behaviour.
            k = args.n_canary
            eps_applicable = epsilon * k

            summary = summarise(estimates, eps_applicable)
            print(f"\n  {'statistic':<16}{'eps_emp':>10}{'p_lo':>9}{'q_hi':>9}  dir")
            print("  " + "-" * 52)
            for est in sorted(estimates, key=lambda e: -e.eps_emp):
                print(f"  {est.statistic:<16}{est.eps_emp:>10.4f}"
                      f"{est.p_lower:>9.3f}{est.q_upper:>9.3f}  {est.direction}")

            if k == 1:
                verdict = ("VIOLATION DETECTED" if summary["violation"]
                           else "no violation found (inconclusive)")
                print(f"\n  eps_claimed={epsilon}   eps_emp_max="
                      f"{summary['eps_emp_max']:.4f}   -> {verdict}\n")
            else:
                detected = summary["eps_emp_max"] > 0.05
                verdict = ("harness detects the difference — OK"
                           if detected else
                           "HARNESS FAILED to detect a blatant difference")
                print(f"\n  sanity check (k={k} records): applicable bound is "
                      f"k*eps={eps_applicable:g}, eps_emp_max="
                      f"{summary['eps_emp_max']:.4f}")
                print(f"  -> {verdict}")
                print("  (no privacy claim is being tested here; k>1 is not "
                      "the unit of privacy)\n")

    if rows:
        df = pd.DataFrame(rows)
        csv = args.out / "audit_results.csv"
        df.to_csv(csv, index=False)
        meta = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()}
        (args.out / "audit_config.json").write_text(json.dumps(meta, indent=2))
        print(f"results written to {csv}")

        worst = df.loc[df["eps_emp"].idxmax()]
        print("\nlargest empirical epsilon observed:")
        print(f"  {worst['eps_emp']:.4f} via '{worst['statistic']}' "
              f"(placement={worst['placement']}, eps_claimed={worst['epsilon_claimed']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
