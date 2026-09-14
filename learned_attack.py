"""DP-Sniper-style learned attack (Bichsel et al., IEEE S&P 2021).

The hand-picked statistics in run_audit.py (n_blocks, sum_estimate, ...) are
each a *fixed* scalar functional of the output. DP-Sniper's insight is that
the most distinguishing event need not be expressible that simply: instead,
train a classifier on the full output vector to approximate the likelihood-
ratio statistic directly, then threshold the classifier's own score.

To keep this honest we use three disjoint splits of the runs, never two:

    clf_train        -- fits the classifier (feature scaling + logistic
                         regression). The classifier never sees the other
                         two splits.
    threshold_select  -- used by estimator.audit_statistic (internally) to
    + eval               pick the best cut point on the classifier's score
                         axis, and then, on the *other* half of this split,
                         to compute the Clopper-Pearson bounds.

Reusing estimator.audit_statistic for the last two splits means the same
honesty guarantee already documented there (README.md, "Selection du seuil
sur des executions distinctes") applies here unchanged: the classifier's
score is just one more scalar statistic to that function, and it was fit on
data disjoint from both the selection and the evaluation runs.

Feature vector
--------------
The six hand-picked statistics, plus the sorted (descending) block means and
block volumes, truncated/zero-padded to a fixed length. Padding with 0 rather
than dropping variable-length information lets a linear classifier react to
"this run had fewer/more large blocks than usual" -- a shape signal none of
the individual scalar statistics captures on its own.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from estimator import audit_statistic, EpsEstimate
from mechanism import run_mechanism
from neighbors import build_pair

PAD_K = 16  # fixed-length truncation for the sorted block-means / volumes


def featurize(out) -> np.ndarray:
    means = np.sort(np.nan_to_num(out.block_means))[::-1]
    vols = np.sort(np.nan_to_num(out.block_volumes))[::-1]
    means_p = np.zeros(PAD_K); means_p[:min(PAD_K, means.size)] = means[:PAD_K]
    vols_p = np.zeros(PAD_K); vols_p[:min(PAD_K, vols.size)] = vols[:PAD_K]
    scalar = np.array([
        out.n_blocks,
        out.max_depth,
        np.nan_to_num(out.canary_block_mean),
        np.nan_to_num(out.canary_block_volume),
        float(np.nansum(out.block_means * out.block_volumes)),
        float(np.nanmax(out.block_means)) if out.block_means.size else 0.0,
    ])
    return np.concatenate([scalar, means_p, vols_p])


def collect(pair, epsilon, runs, repo, base_seed):
    obs = {"D": [], "Dprime": []}
    for i in range(runs):
        for label, data in (("D", pair.d), ("Dprime", pair.d_prime)):
            seed = base_seed + i * 2 + (0 if label == "D" else 1)
            out = run_mechanism(data, pair.attrs, pair.shape, epsilon, seed,
                                 repo, canary_cell=pair.canary_cell)
            obs[label].append(out)
        if (i + 1) % max(1, runs // 10) == 0:
            print(f"    {i+1:4d}/{runs} runs", flush=True)
    return obs


def learned_audit(pair, epsilon, runs, repo, base_seed, alpha, delta) -> EpsEstimate:
    obs = collect(pair, epsilon, runs, repo, base_seed)

    Xd = np.array([featurize(o) for o in obs["D"]])
    Xp = np.array([featurize(o) for o in obs["Dprime"]])

    rng = np.random.default_rng(base_seed)
    idx_d = rng.permutation(len(Xd))
    idx_p = rng.permutation(len(Xp))
    cut_d, cut_p = len(Xd) // 2, len(Xp) // 2

    train_X = np.concatenate([Xd[idx_d[:cut_d]], Xp[idx_p[:cut_p]]])
    train_y = np.concatenate([np.zeros(cut_d), np.ones(cut_p)])
    hold_Xd = Xd[idx_d[cut_d:]]
    hold_Xp = Xp[idx_p[cut_p:]]

    scaler = StandardScaler().fit(train_X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(train_X), train_y)

    score_d = clf.decision_function(scaler.transform(hold_Xd))
    score_p = clf.decision_function(scaler.transform(hold_Xp))

    return audit_statistic(score_d, score_p, epsilon, delta=delta, alpha=alpha,
                            statistic_name="learned_dpsniper",
                            rng=np.random.default_rng(base_seed + 1))


def main():
    ap = argparse.ArgumentParser(description="DP-Sniper-style learned attack on RIPOST")
    ap.add_argument("--repo", type=Path, default=Path("./RIPOST"))
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--rows", type=int, default=800)
    ap.add_argument("--shape", type=int, nargs="+", default=[6, 6, 3])
    ap.add_argument("--placement", default="boundary")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        print("warning: 'fork' unavailable; using platform default.", file=sys.stderr)

    attrs = tuple("abcdefgh"[: len(args.shape)])
    pair = build_pair(placement=args.placement, n_rows=args.rows,
                       shape=tuple(args.shape), attrs=attrs, seed=args.seed)
    print(pair.describe())

    est = learned_audit(pair, args.epsilon, args.runs, args.repo,
                         args.seed * 1000 + 1, args.alpha, args.delta)
    print("\nlearned attack result:")
    for k, v in est.as_row().items():
        print(f"  {k}: {v}")
    verdict = "VIOLATION" if est.eps_emp > args.epsilon else "no violation (inconclusive)"
    print(f"\neps_claimed={args.epsilon}  eps_emp={est.eps_emp:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
