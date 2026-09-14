"""Demonstrates bugs #4/#5: exp_mech's cut selection and the final leaf
perturbation are driven by NumPy's *global* RNG, not by the `prng` argument
RIPOST.run receives — so a fixed prng seed does not make a run reproducible.

Holds the prng seed fixed across two calls and varies only the global
np.random seed beforehand. If the output changes, the prng seed alone did
not determine it.
"""
import sys
from pathlib import Path

import numpy as np

from mechanism import run_mechanism
from neighbors import build_pair


def main():
    pair = build_pair(placement="boundary", n_rows=800, shape=(6, 6, 3), seed=0)
    repo = Path("./RIPOST/unzipped")

    results = []
    for global_seed in (111, 222):
        np.random.seed(global_seed)  # only the GLOBAL numpy RNG changes
        out = run_mechanism(pair.d, pair.attrs, pair.shape, epsilon=1.0,
                            seed=42,  # prng seed held FIXED
                            repo=repo, canary_cell=pair.canary_cell)
        means = tuple(sorted(np.round(np.nan_to_num(out.block_means)[:8], 6)))
        results.append((global_seed, out.n_blocks, means))
        print(f"global np.random.seed={global_seed:4d}   prng_seed=42 (fixed)  "
              f"->  n_blocks={out.n_blocks:3d}   first block means={means}")

    same = results[0][1:] == results[1][1:]
    print()
    print("same prng seed (42) both times, only the global RNG state differed"
          " ->", "IDENTICAL output (unexpected)" if same else
          "DIFFERENT output — confirms the prng seed alone does not "
          "determine the mechanism's output; the global RNG does.")
    return 0 if not same else 1


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        print("warning: could not select 'fork'.", file=sys.stderr)
    sys.exit(main())
