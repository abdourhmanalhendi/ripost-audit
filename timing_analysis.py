"""Timing side-channel analysis on --dump-raw campaigns (Job 4).

Reads raw_<eps>_<placement>.csv files (one row per run: label, n_blocks,
max_depth, runtime_s, ...) and reports:
  - Pearson correlation between runtime_s and n_blocks / max_depth (pooled D+D')
  - Mann-Whitney U test comparing runtime_s between D and D' (the 1-record
    difference itself)
"""
import sys
import glob

import pandas as pd
from scipy import stats

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results_v2/raw_*.csv")):
    df = pd.read_csv(path)
    if "runtime_s" not in df.columns or len(df) < 4:
        continue
    r_blocks, p_blocks = stats.pearsonr(df["runtime_s"], df["n_blocks"])
    r_depth, p_depth = stats.pearsonr(df["runtime_s"], df["max_depth"])
    rt_d = df[df.label == "D"]["runtime_s"]
    rt_p = df[df.label == "Dprime"]["runtime_s"]
    u, p_mw = stats.mannwhitneyu(rt_d, rt_p, alternative="two-sided")
    print(f"{path:45s}  n={len(df):4d}  "
          f"corr(rt,n_blocks)={r_blocks:+.3f} (p={p_blocks:.2g})   "
          f"corr(rt,max_depth)={r_depth:+.3f} (p={p_depth:.2g})   "
          f"MW D-vs-D' p={p_mw:.3g}   "
          f"median_rt_D={rt_d.median():.3f}s  median_rt_D'={rt_p.median():.3f}s")
