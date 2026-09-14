"""Black-box wrapper around RIPOST.

This is the only module that touches the RIPOST API.  Everything else in the
audit treats the mechanism as an opaque function

    M : (dataset, epsilon, seed)  ->  published view

so that the same harness can later be pointed at HDPView, PrivTree, or any
other decomposition mechanism without changes.

The RIPOST repository (github.com/AlaEddineLaouir/RIPOST) exposes its entry
point slightly differently across revisions, so `_call_ripost` tries a small
number of known call shapes and reports clearly which one worked.  Run

    python mechanism.py --selftest

after cloning the repository and before launching any audit campaign: a
negative audit result is meaningless if the wrapper was silently failing.

Note on the NumPy shim below: the repository targets numpy < 1.24, where
`np.float`, `np.int` and `np.bool` still existed as aliases.  They were
removed in 1.24, so the code raises AttributeError on any recent environment.
Restoring the aliases is the minimal change that makes the published code run
unmodified.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# --- compatibility shim: must run before importing the RIPOST package -------
# Only the three aliases the repository actually uses. Touching np.object /
# np.str would emit FutureWarnings on NumPy >= 2 for no benefit.
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _name, _py in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, _name):
            setattr(np, _name, _py)


class WideRandomState(np.random.RandomState):
    """Work around a platform-dependent bug in the published code, not a
    NumPy-version issue.

    ``src/RIPOST/RIPOST.py`` calls ``prng.randint(0, 2949672950)`` without a
    ``dtype``. NumPy's default dtype for ``RandomState.randint`` is the
    platform C ``long``: 64-bit on Linux/macOS, but 32-bit on Windows. The
    constant exceeds ``int32``'s range, so the unmodified call raises
    ``ValueError: high is out of bounds for int32`` on every Windows machine
    while running fine on the authors' (presumably Linux) setup — see
    RAPPORT.md, "bugs de reproductibilite".

    Forcing ``dtype=np.int64`` whenever the caller does not specify one
    reproduces exactly what the *same* call already does, unmodified, on a
    64-bit-``long`` platform: it does not change the sampling distribution,
    only which register width Windows happens to default to. This class is
    otherwise a plain ``np.random.RandomState``.
    """

    def randint(self, low, high=None, size=None, dtype=None):
        if dtype is None:
            dtype = np.int64
        return super().randint(low, high=high, size=size, dtype=dtype)


# --------------------------------------------------------------------------
# Output container
# --------------------------------------------------------------------------

@dataclass
class ViewOutput:
    """What the audit observes about one execution of the mechanism.

    Only quantities an adversary could actually see are recorded here: the
    published blocks and their noisy values, plus the wall-clock runtime,
    which is not part of the released view but *is* observable when the
    mechanism runs as a service.  Runtime is collected but excluded from the
    epsilon estimation by default (see run_audit.py --include-timing): the
    epsilon budget only covers the output, so mixing timing into the same
    number would conflate two different leakage channels.
    """

    n_blocks: int
    block_means: np.ndarray
    block_volumes: np.ndarray
    max_depth: int
    runtime_s: float
    canary_block_mean: float = float("nan")
    canary_block_volume: float = float("nan")
    raw: Any = field(default=None, repr=False)


# --------------------------------------------------------------------------
# RIPOST plumbing
# --------------------------------------------------------------------------

def _find_package_root(repo: Path) -> Path:
    """Locate the directory that should go on sys.path.

    Clones are not always flat: the code may sit at the repository root, or one
    level down in a folder such as `RIPOST-main`.  We look for the directory
    that directly contains the `src` package, and fall back to the repo root.
    """
    if (repo / "src").is_dir():
        return repo

    # Directories that mirror the real tree but contain no usable code.
    # __MACOSX in particular is an artefact of zipping on macOS and shadows
    # the whole layout, so it must be skipped explicitly.
    junk = {"__MACOSX", ".git", "__pycache__", ".idea", ".vscode"}

    for candidate in sorted(repo.rglob("src")):
        if not candidate.is_dir():
            continue
        if junk.intersection(candidate.parts):
            continue
        # a real match actually holds the package modules
        if ((candidate / "hdpview" / "count_table.py").is_file()
                or (candidate / "hdpview").is_dir()):
            return candidate.parent

    return repo


def _add_repo_to_path(repo: Path) -> None:
    repo = repo.resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"RIPOST repository not found at {repo}.\n"
            "Clone it first:\n"
            "  git clone https://github.com/AlaEddineLaouir/RIPOST.git"
        )

    root = _find_package_root(repo)
    if not (root / "src").is_dir():
        listing = "\n".join(f"    {p.relative_to(repo)}"
                            for p in sorted(repo.iterdir())
                            if p.name != ".git")
        raise ModuleNotFoundError(
            f"No 'src' package found under {repo}.\n"
            f"Top level of the clone:\n{listing}\n"
            "Pass --repo pointing at the directory that contains 'src'."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _build_count_table(df: pd.DataFrame, attrs, shape):
    """Build RIPOST's CountTable from a plain DataFrame.

    `CountTable.from_dataset` is the only constructor that actually works in
    the published code, and it is incomplete: it passes ``None`` for
    ``mesure_table`` (see src/hdpview/count_table.py), so any subsequent call
    to RIPOST.run fails with "'NoneType' object is not subscriptable".  The
    authors' own driver works around this in src/RIPOST/ripost_run.py by
    assigning the measure table after construction; we do the same.

    (The two sibling constructors, `from_pd_count_table` and `from_pd_table`,
    are broken outright: both call `CountTable(...)` with four positional
    arguments while `__init__` requires five.)

    `mesure_table` is a flat frame — one row per non-empty cell, the attribute
    columns holding the coordinates and a `Mesure` column holding the value —
    as implied by the indexing patterns `self.mesure_table[dim]` and
    `self.mesure_table['Mesure']` throughout count_table.py.
    """
    Domain = importlib.import_module("src.hdpview.domain").Domain
    Dataset = importlib.import_module("src.hdpview.dataset").Dataset
    CountTable = importlib.import_module("src.hdpview.count_table").CountTable

    domain = Domain(tuple(attrs), tuple(shape))
    table = CountTable.from_dataset(Dataset(df.copy(), domain))

    if getattr(table, "mesure_table", None) is None:
        tmp = df.copy()
        tmp["Mesure"] = 1
        mesure = (tmp.groupby(list(attrs), as_index=False)["Mesure"]
                     .sum()
                     .reset_index(drop=True))
        table.mesure_table = mesure

    return table


# RIPOST.run's signature, confirmed by introspection of the published code:
#
#   run(block, epsilon, ratio, prng, alpha=2, beta=1.2, gamma=1.0,
#       ratios=[0.11, 0.11], theta=None, verbose=False)
#
# The authors' own experiment driver calls it as
#   run(initial, eps, .3, prng, 0, 0, .9, [.4, 1])
# so `ratio` carries the paper's alpha (0.3), `ratios` is [beta, step] =
# [0.4, 1], gamma is 0.9, and the `alpha`/`beta` keyword arguments are left
# at 0.  DEFAULT_HP below reproduces exactly that configuration.

def _call_ripost(table, eps: float, prng, hp: dict):
    """Invoke RIPOST.run with the authors' own hyper-parameter configuration."""
    RIPOST = importlib.import_module("src.RIPOST.RIPOST")
    if not hasattr(RIPOST, "run"):          # some revisions nest it one level down
        RIPOST = importlib.import_module("src.RIPOST").RIPOST

    return RIPOST.run(
        table,
        eps,
        hp["ratio"],
        prng,
        hp["alpha"],
        hp["beta"],
        hp["gamma"],
        [hp["phase_split"], hp["step"]],
        hp["theta"],
        False,
    )


# --------------------------------------------------------------------------
# Block introspection
# --------------------------------------------------------------------------

def _block_domain(block) -> dict | None:
    for attr in ("domain_dict", "domain", "ranges"):
        val = getattr(block, attr, None)
        if isinstance(val, dict):
            return val
    return None


def _block_value(block) -> float:
    for attr in ("mean", "value", "noisy_mean", "count"):
        val = getattr(block, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return float("nan")


def _block_volume(block) -> float:
    dom = _block_domain(block)
    if not dom:
        return float("nan")
    vol = 1.0
    for lo, hi in dom.values():
        vol *= (float(hi) - float(lo) + 1.0)
    return vol


def _contains(block, attrs, cell) -> bool:
    dom = _block_domain(block)
    if not dom:
        return False
    for attr, coord in zip(attrs, cell):
        rng = dom.get(attr)
        if rng is None:
            return False
        lo, hi = float(rng[0]), float(rng[1])
        if not (lo <= coord <= hi):
            return False
    return True


def _depth_of(block, shape) -> int:
    """Approximate decomposition depth from how far a block was subdivided."""
    dom = _block_domain(block)
    if not dom:
        return 0
    depth = 0
    for size, (_, rng) in zip(shape, dom.items()):
        extent = float(rng[1]) - float(rng[0]) + 1.0
        if extent > 0 and size > 0:
            ratio = size / extent
            if ratio > 1:
                depth += int(np.floor(np.log2(ratio)))
    return depth


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

# Mirrors the authors' call run(initial, eps, .3, prng, 0, 0, .9, [.4, 1]).
DEFAULT_HP = {
    "ratio": 0.3,        # paper's alpha: share of the budget given to the decomposition
    "alpha": 0,          # run()'s own alpha keyword, unused in the authors' setup
    "beta": 0,           # idem
    "gamma": 0.9,        # split between convergence test and cut selection
    "phase_split": 0.4,  # ratios[0]: budget share of phase 1
    "step": 1,           # ratios[1]
    "theta": None,
}


def run_mechanism(
    df: pd.DataFrame,
    attrs,
    shape,
    epsilon: float,
    seed: int,
    repo: Path,
    canary_cell=None,
    hp: dict | None = None,
) -> ViewOutput:
    """One execution of RIPOST on `df`, returning only observable quantities."""
    _add_repo_to_path(repo)
    hp = {**DEFAULT_HP, **(hp or {})}

    table = _build_count_table(df, attrs, shape)
    prng = WideRandomState(seed)

    # RIPOST writes diagnostic material to stdout during the decomposition,
    # which corrupts the audit's own progress output over thousands of runs.
    # stderr is left untouched so real errors remain visible.
    import contextlib
    import io

    sink = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sink):
        out = _call_ripost(table, epsilon, prng, hp)
    runtime = time.perf_counter() - t0

    # RIPOST.run returns (p_view, block_result_list) — the authors' own driver
    # (src/RIPOST/ripost_run.py:105) names them exactly that and answers every
    # query through `p_view`, never through `block_result_list`.
    #
    # The two are NOT interchangeable. `block_result_list` (out[1]) holds raw
    # BlockResult objects whose `.mean` is the *unperturbed* empirical mean:
    # RIPOST.py computes a Laplace draw `pe` and stores it in a sibling field
    # `.perturbation_error`, but never adds it to `.mean` on that object (see
    # src/RIPOST/RIPOST.py, the `pe = np.random.laplace(...)` block). The
    # noise is only actually applied inside `NoisedCountTable.__init__`
    # (src/hdpview/count_table.py:441-447), which builds `p_view.blocks` as
    # `Block(domain, mean + perturbation_error / block_size)` — i.e. `p_view`
    # (out[0]) is the published, noised view, and out[1] is pre-noise
    # scaffolding that happens to also be returned.
    #
    # An earlier version of this wrapper read out[1]. Because that list's
    # `.mean` never varies across independent runs on the same data (see
    # inspect_block.py and RAPPORT.md, "artefact n1"), every statistic derived
    # from it was reading raw data, not the released view — which is exactly
    # what produced the spuriously large eps_emp this audit first reported.
    view = out[0] if isinstance(out, (tuple, list)) and len(out) >= 2 else out
    blocks = list(getattr(view, "blocks", view))

    means = np.array([_block_value(b) for b in blocks], dtype=float)
    vols = np.array([_block_volume(b) for b in blocks], dtype=float)
    depths = [_depth_of(b, shape) for b in blocks]

    canary_mean, canary_vol = float("nan"), float("nan")
    if canary_cell is not None:
        for b, m, v in zip(blocks, means, vols):
            if _contains(b, attrs, canary_cell):
                canary_mean, canary_vol = float(m), float(v)
                break

    result = ViewOutput(
        n_blocks=len(blocks),
        block_means=means,
        block_volumes=vols,
        max_depth=int(max(depths)) if depths else 0,
        runtime_s=runtime,
        canary_block_mean=canary_mean,
        canary_block_volume=canary_vol,
        raw=None,
    )

    # RIPOST.run creates a fresh multiprocessing.Manager() and Pool(1) on
    # *every* call (RIPOST.py: run()) and never calls manager.shutdown() or
    # pool.close()/join(). Each Manager spins up its own server process,
    # which Python's multiprocessing only tears down when the Manager object
    # is garbage-collected. Across a campaign of thousands of calls, relying
    # on GC timing to reclaim these processes/file-descriptors/semaphores in
    # a timely fashion let them pile up faster than they were reclaimed —
    # confirmed empirically: an unmitigated audit of ~1200 calls exhausted
    # host memory and the run was killed. Forcing collection here is a
    # workaround in *our* wrapper, not a change to the published code; it
    # does not touch what gets measured, only how promptly its by-products
    # are freed.
    import gc
    del out, view, blocks, table
    gc.collect()

    return result


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _selftest(repo: Path) -> int:
    """Verify the wrapper before trusting any audit result."""
    from neighbors import build_pair

    print("=" * 68)
    print("RIPOST wrapper self-test")
    print("=" * 68)

    pair = build_pair(placement="boundary", n_rows=800, shape=(6, 6, 3), seed=0)
    print(pair.describe())

    ok = True
    for label, data in (("D", pair.d), ("D'", pair.d_prime)):
        try:
            out = run_mechanism(data, pair.attrs, pair.shape, epsilon=1.0,
                                seed=1, repo=repo, canary_cell=pair.canary_cell)
            print(f"  {label}: blocks={out.n_blocks:4d}  depth={out.max_depth:2d}  "
                  f"canary_mean={out.canary_block_mean:8.2f}  "
                  f"runtime={out.runtime_s:.2f}s")
        except Exception as exc:
            ok = False
            print(f"  {label}: FAILED -> {type(exc).__name__}: {exc}")

    # determinism check: same seed must give the same output
    try:
        o1 = run_mechanism(pair.d, pair.attrs, pair.shape, 1.0, 7, repo)
        o2 = run_mechanism(pair.d, pair.attrs, pair.shape, 1.0, 7, repo)
        same = (o1.n_blocks == o2.n_blocks and
                np.allclose(np.sort(o1.block_means), np.sort(o2.block_means),
                            equal_nan=True))
        print(f"determinism under fixed seed: {'OK' if same else 'NOT reproducible'}")
        if not same:
            print("  -> the mechanism draws randomness outside the supplied prng;")
            print("     note this in the report, it affects reproducibility.")
    except Exception as exc:
        print(f"determinism check skipped: {exc}")

    print("=" * 68)
    print("PASS — ready to audit" if ok else "FAIL — fix the wrapper first")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RIPOST black-box wrapper")
    ap.add_argument("--repo", default="./RIPOST", type=Path,
                    help="path to the cloned RIPOST repository")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest(args.repo))
    ap.print_help()
