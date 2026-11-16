#!/usr/bin/env python3
"""
Make the fixed SMUDGes train/val/test splits used by the experiments.

  1. Drop sources whose cutout has missing data: any band with > 1% of
     pixels exactly zero / non-finite in the central 480x480 (the same rule
     and region load_sample uses, so exp1/exp3 drop nothing further).
  2. Keep only rows that pass load_table2(max_re_arcsec=10): VC == 0, no
     missing values/biases, positive corrected Re/n/b-a, Re_corr <= 10".
  3. Randomly permute the remaining rows (numpy default_rng(42)) and cut
     70/15/15 by row count (round()).
  4. Write {train,val,test}_reff10.dat: original table2 lines, verbatim, in
     catalog order.

The split is made AFTER the image cut, so it depends on the H5 contents and
the missing-data rule; changing either changes the splits.
(Before 2026-11-16 the split was made on all of table2.dat before any cut.)
"""
import argparse
from pathlib import Path

import numpy as np

from aion_smudges_common import (
    TABLE2_DEFAULT, H5_DEFAULT, MAX_MISSING_FRAC, load_table2, get_or_build_binned,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table2", default=TABLE2_DEFAULT)
    p.add_argument("--h5-path", default=H5_DEFAULT)
    p.add_argument("--out-dir", default=str(Path(TABLE2_DEFAULT).parent))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-reff", type=float, default=10.0)
    args = p.parse_args()

    # missing fractions are measured on the native 480x480 crop, so they are
    # the same for every bin factor; x2 is used only to reuse its cache.
    h5_names, _, missing = get_or_build_binned(args.h5_path, "sum", 2)
    has_data = set(np.asarray(h5_names, dtype=str)[(missing <= MAX_MISSING_FRAC).all(axis=1)])
    print(f"{len(has_data)} of {len(h5_names)} cutouts have no band with "
          f">{MAX_MISSING_FRAC:.0%} missing pixels")

    keep = set(load_table2(args.table2, max_re_arcsec=args.max_reff)["name"]) & has_data
    lines = [l for l in open(args.table2) if l[:18].strip() in keep]
    print(f"{len(lines)} sources with data and Re_corr <= {args.max_reff:g}\"")

    idx = np.random.default_rng(args.seed).permutation(len(lines))
    n_tr, n_va = round(0.70 * len(lines)), round(0.15 * len(lines))
    splits = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}

    tag = f"reff{args.max_reff:g}"
    for name, ii in splits.items():
        path = Path(args.out_dir) / f"{name}_{tag}.dat"
        path.write_text("".join(lines[i] for i in sorted(ii)))
        print(f"{name}: {len(ii)} -> {path}")


if __name__ == "__main__":
    main()
