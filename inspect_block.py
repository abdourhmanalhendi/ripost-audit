"""Check that the audit reads the *published* block value, not a clear one.

If a block carries both the true mean and the perturbed mean, and the wrapper
happens to read the true one, every "leak" the audit reports is an artefact:
it would be measuring the raw data, not the released view.

Two checks:
  1. list every numeric attribute a block exposes;
  2. run the mechanism twice on identical data and report which attributes
     change.  An attribute that never changes across independent runs is not
     noised, and must not be used as an audit statistic.

Everything runs inside main(), guarded by ``if __name__ == "__main__"``.
RIPOST.run() creates a multiprocessing.Manager() on every call; on Windows
(no 'fork', only 'spawn') that re-imports this file as ``__mp_main__``, and
without the guard the whole script — including the top-level one_run() calls
below — runs again inside that bootstrap step, which Python's own
`_check_not_importing_main` correctly refuses. An unguarded top-level script
is not just bad style here, it makes the script fail outright on Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mechanism import (_add_repo_to_path, _block_value, _block_volume,
                        _build_count_table, _call_ripost, DEFAULT_HP,
                        WideRandomState)
from neighbors import build_pair


def main():
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else "./RIPOST")
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    pair = build_pair(placement="boundary", n_rows=800, shape=(6, 6, 3), seed=0)
    _add_repo_to_path(repo)

    def one_run(seed: int):
        table = _build_count_table(pair.d, pair.attrs, pair.shape)
        out = _call_ripost(table, eps, WideRandomState(seed), DEFAULT_HP)
        # out[0] is p_view, the published/noised view; out[1] is the pre-noise
        # scaffolding (see mechanism.py's run_mechanism for the full story).
        view = out[0] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
        return list(getattr(view, "blocks", view))

    print("=" * 70)
    print(f"block introspection   epsilon={eps}   |D|={len(pair.d)}")
    print("=" * 70)

    b0 = one_run(1)
    sample = b0[0]
    attrs = [a for a in dir(sample) if not a.startswith("_")
             and not callable(getattr(sample, a, None))]
    print("attributes on a block:")
    for a in attrs:
        v = getattr(sample, a, None)
        kind = type(v).__name__
        shown = v if isinstance(v, (int, float, bool, str)) else kind
        print(f"  {a:24} = {shown}")

    print("\n" + "=" * 70)
    print("which attributes vary across two independent runs on the SAME data?")
    print("=" * 70)

    r1 = one_run(11)
    r2 = one_run(22)
    print(f"n_blocks: run1={len(r1)}  run2={len(r2)}"
          f"   {'-> structure is data/noise dependent' if len(r1) != len(r2) else ''}")

    n = min(len(r1), len(r2))
    for a in attrs:
        try:
            v1 = np.array([float(getattr(b, a)) for b in r1[:n]], dtype=float)
            v2 = np.array([float(getattr(b, a)) for b in r2[:n]], dtype=float)
        except (TypeError, ValueError):
            continue
        identical = np.allclose(np.sort(v1), np.sort(v2), equal_nan=True)
        verdict = ("IDENTICAL -> not noised, unusable as an audit statistic"
                   if identical else "varies -> perturbed, usable")
        print(f"  {a:24} {verdict}")

    print("\n" + "=" * 70)
    print("sum over blocks of value*volume, five independent runs on the SAME data")
    print("(the spread here is the noise the audit has to overcome; if it is tiny")
    print(" compared with 1 record, the published total is effectively unprotected)")
    print("=" * 70)

    totals = []
    for s in (101, 102, 103, 104, 105):
        bl = one_run(s)
        tot = float(np.nansum([_block_value(b) * _block_volume(b) for b in bl]))
        totals.append(tot)
        print(f"  seed {s}: n_blocks={len(bl):4d}   sum_estimate={tot:14.3f}")

    arr = np.array(totals)
    print(f"\n  true |D| = {len(pair.d)}")
    print(f"  mean     = {arr.mean():.3f}")
    print(f"  std dev  = {arr.std(ddof=1):.3f}")
    print(f"\n  A one-record difference is detectable only if this spread is")
    print(f"  comparable to 1.  Spread = {arr.std(ddof=1):.3f}")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        print("warning: could not select the 'fork' start method; "
              "runs may be slower.", file=sys.stderr)
    main()
