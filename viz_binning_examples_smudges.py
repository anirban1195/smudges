#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show a few SMUDGes cutouts before and after the 480-crop + 5x5 binning.

For each randomly chosen source (from the same filtered sample the
experiments use) two g/r/z colour panels are drawn side by side:
  left : raw central 480x480 crop (0.262 arcsec/px)
  right: binned 96x96 (1.31 arcsec/px), what AION sees
A circle of radius Re_corr is overlaid on both. Each panel is stretched
independently (per-band median/MAD, then arcsinh), so absolute scales
(sum vs mean) don't matter for the comparison.

Not run in the sandbox.
"""
import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from aion_smudges_common import (
    H5_DEFAULT, TABLE2_DEFAULT, CROP_SIZE, OUT_SIZE, PIXEL_SCALE, BINNED_SCALE,
    load_h5_names, load_sample,
)

RGB_BANDS = [3, 1, 0]  # z, r, g  -> R, G, B  (H5 band order is g, r, i, z)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table2", default=TABLE2_DEFAULT)
    p.add_argument("--h5-path", default=H5_DEFAULT)
    p.add_argument("--bin-mode", choices=["sum", "mean"], default="sum")
    p.add_argument("--n", type=int, default=8, help="Number of sources to show (5-10).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./binning_examples.png")
    return p.parse_args()


def stretch_rgb(img):
    """img: (4, H, W) -> (H, W, 3) arcsinh-stretched g/r/z composite."""
    chans = []
    for b in RGB_BANDS:
        c = img[b].astype(np.float64)
        med = np.median(c)
        sig = 1.4826 * np.median(np.abs(c - med)) or 1.0
        chans.append(np.arcsinh((c - med) / sig))
    rgb = np.stack(chans, axis=-1)
    lo, hi = np.percentile(rgb, [1, 99.5])
    return np.clip((rgb - lo) / (hi - lo + 1e-12), 0, 1)


def main():
    args = parse_args()
    df, fluxes, _ = load_sample(args.table2, args.h5_path, args.bin_mode)

    rng = np.random.default_rng(args.seed)
    picks = np.sort(rng.choice(len(df), size=min(args.n, len(df)), replace=False))

    h5_pos = {n: i for i, n in enumerate(load_h5_names(args.h5_path))}
    y0 = x0 = (512 - CROP_SIZE) // 2  # H5 cutouts are 512x512

    ncols_pairs = 2
    nrows = int(np.ceil(len(picks) / ncols_pairs))
    fig, axes = plt.subplots(nrows, 2 * ncols_pairs,
                              figsize=(4 * 2 * ncols_pairs, 4.2 * nrows))
    axes = np.atleast_2d(axes)

    with h5py.File(args.h5_path, "r") as h5f:
        dset = h5f["fluxes"]
        for k, i in enumerate(picks):
            row, col = divmod(k, ncols_pairs)
            ax_raw, ax_bin = axes[row, 2 * col], axes[row, 2 * col + 1]
            r = df.iloc[i]

            raw = dset[h5_pos[r["name"]], :, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
            raw = np.nan_to_num(raw.astype(np.float32))

            for ax, img, npix, scale, title in [
                (ax_raw, raw, CROP_SIZE, PIXEL_SCALE, "raw 480x480"),
                (ax_bin, fluxes[i], OUT_SIZE, BINNED_SCALE, f"binned 96x96 ({args.bin_mode})"),
            ]:
                ax.imshow(stretch_rgb(img), origin="lower")
                c = (npix - 1) / 2
                ax.add_patch(plt.Circle((c, c), r["Re_corr"] / scale, fill=False,
                                         color="white", lw=1, ls="--"))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=9)
            ax_raw.set_ylabel(f"{r['name']}\nRe={r['Re_corr']:.1f}\" "
                              f"mu0g={r['mu0g_corr']:.1f} n={r['n_corr']:.2f}",
                              fontsize=8)

    for j in range(len(picks), nrows * ncols_pairs):
        row, col = divmod(j, ncols_pairs)
        axes[row, 2 * col].axis("off")
        axes[row, 2 * col + 1].axis("off")

    fig.suptitle("g/r/z composites, dashed circle = Re (bias-corrected)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {Path(args.out).resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
