"""Neighbouring-dataset construction for the RIPOST audit.

Two datasets are *neighbours* when one is obtained from the other by adding or
removing a single record (unbounded DP, the notion used by decomposition-based
view publishing).  The audit compares the mechanism's output distribution on
D and on D' = D + {one record}.

Where that record is placed matters enormously.  RIPOST decides where to split
the domain *from the data*, so a record dropped in the middle of a dense
cluster changes almost nothing, whereas a record placed where it can flip a
split decision may change the entire shape of the published tree.  We
therefore implement several placement strategies and audit each one: the
comparison between them is itself a result.

Placement strategies
--------------------
dense
    Inside the densest region.  Control case: the record is drowned in a large
    count, so this should leak least.
sparse
    In an otherwise empty region of the domain.  The extra record turns an
    empty sub-domain into a non-empty one, which is precisely the distinction
    RIPOST's first phase is built on.
boundary
    Immediately next to a dense cluster, in a cell that is empty in D.  This is
    the adversarial choice: it targets the convergence condition, where the
    presence of one record may or may not justify a further split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Placement = Literal["dense", "sparse", "boundary"]


@dataclass
class DatasetPair:
    """A pair of neighbouring datasets plus the cell that distinguishes them."""

    d: pd.DataFrame
    d_prime: pd.DataFrame
    attrs: tuple[str, ...]
    shape: tuple[int, ...]
    canary_cell: tuple[int, ...]
    placement: str

    def describe(self) -> str:
        return (
            f"placement={self.placement}  canary={self.canary_cell}  "
            f"|D|={len(self.d)}  |D'|={len(self.d_prime)}  "
            f"domain={'x'.join(str(s) for s in self.shape)}"
        )


def make_base_dataset(
    n_rows: int = 2000,
    shape: tuple[int, ...] = (8, 8, 4),
    attrs: tuple[str, ...] = ("a", "b", "c"),
    concentration: float = 0.75,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic dataset with a realistic dense core and a sparse periphery.

    A uniform dataset would be the easy case for any decomposition algorithm
    and would not exercise the data-aware splitting we want to audit.  We
    therefore concentrate most of the mass in a small corner of the domain and
    scatter the rest, which mirrors the skew of the real tensors used in the
    RIPOST evaluation.
    """
    rng = np.random.RandomState(seed)
    n_dense = int(n_rows * concentration)
    n_tail = n_rows - n_dense

    dense_extent = [max(1, s // 4) for s in shape]
    dense = np.column_stack([
        rng.randint(0, dense_extent[i], n_dense) for i in range(len(shape))
    ])
    tail = np.column_stack([
        rng.randint(0, shape[i], n_tail) for i in range(len(shape))
    ])

    data = np.vstack([dense, tail])
    rng.shuffle(data)
    return pd.DataFrame(data, columns=list(attrs))


def _cell_counts(df: pd.DataFrame, attrs: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros(shape, dtype=np.int64)
    idx = tuple(df[a].to_numpy() for a in attrs)
    np.add.at(grid, idx, 1)
    return grid


def _pick_cell(grid: np.ndarray, placement: Placement) -> tuple[int, ...]:
    if placement == "dense":
        return tuple(int(i) for i in np.unravel_index(int(np.argmax(grid)), grid.shape))

    empty = np.argwhere(grid == 0)
    if empty.size == 0:
        # degenerate: no empty cell, fall back to the least populated one
        return tuple(int(i) for i in np.unravel_index(int(np.argmin(grid)), grid.shape))

    if placement == "sparse":
        # the empty cell farthest from the dense core
        core = np.array(np.unravel_index(int(np.argmax(grid)), grid.shape))
        dists = np.abs(empty - core).sum(axis=1)
        return tuple(int(i) for i in empty[int(np.argmax(dists))])

    if placement == "boundary":
        # the empty cell closest to the dense core: adding one record here
        # sits exactly where a split decision can tip either way
        core = np.array(np.unravel_index(int(np.argmax(grid)), grid.shape))
        dists = np.abs(empty - core).sum(axis=1)
        return tuple(int(i) for i in empty[int(np.argmin(dists))])

    raise ValueError(f"unknown placement: {placement}")


def build_pair(
    placement: Placement = "boundary",
    n_rows: int = 2000,
    shape: tuple[int, ...] = (8, 8, 4),
    attrs: tuple[str, ...] = ("a", "b", "c"),
    seed: int = 0,
    n_canary: int = 1,
) -> DatasetPair:
    """Build D and D' = D + n_canary record(s) at the chosen cell.

    n_canary should be 1 for a genuine DP audit (one record is the unit of
    privacy).  Larger values are only useful as a sanity check: if the audit
    cannot detect even a blatant 50-record difference, the harness itself is
    broken and the negative result on n_canary=1 means nothing.
    """
    d = make_base_dataset(n_rows, shape, attrs, seed=seed)
    grid = _cell_counts(d, attrs, shape)
    cell = _pick_cell(grid, placement)

    extra = pd.DataFrame([list(cell)] * n_canary, columns=list(attrs))
    d_prime = pd.concat([d, extra], ignore_index=True)

    return DatasetPair(
        d=d, d_prime=d_prime, attrs=attrs, shape=shape,
        canary_cell=cell, placement=f"{placement}(n={n_canary})",
    )
