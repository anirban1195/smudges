#!/usr/bin/env python3
"""
Make the fixed SMUDGes train/val/test splits used by the experiments.

  1. Randomly permute ALL rows of table2.dat (numpy default_rng(42)) and cut
     70/15/15 by row count (round()). This is done before any quality cut,
     so a source's split never depends on the cuts.
  2. Keep only rows that pass load_table2(max_re_arcsec=10): VC == 0, no
     missing values/biases, positive corrected Re/n/b-a, Re_corr <= 10".
     (load_table2 filters row by row, so filtering the whole table once is
     the same as filtering each split separately.)
  3. Write {train,val,test}_reff10.dat: original table2 lines, verbatim, in
     catalog order.

Recovered from the inline code of the 2026-10-28 session that made the
committed split files; reproduces them byte for byte.
"""
import argparse
from pathlib import Path

import numpy as np

from aion_smudges_common import TABLE2_DEFAULT, load_table2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table2", default=TABLE2_DEFAULT)
    p.add_argument("--out-dir", default=str(Path(TABLE2_DEFAULT).parent))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-reff", type=float, default=10.0)
    args = p.parse_args()

    lines = open(args.table2).readlines()
    idx = np.random.default_rng(args.seed).permutation(len(lines))
    n_tr, n_va = round(0.70 * len(lines)), round(0.15 * len(lines))
    splits = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    keep = set(load_table2(args.table2, max_re_arcsec=args.max_reff)["name"])
    tag = f"reff{args.max_reff:g}"
    for name, ii in splits.items():
        out = [lines[i] for i in sorted(ii) if lines[i][:18].strip() in keep]
        path = Path(args.out_dir) / f"{name}_{tag}.dat"
        path.write_text("".join(out))
        print(f"{name}: {len(ii)} rows -> {len(out)} after cuts -> {path}")


if __name__ == "__main__":
    main()
