#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 3 (SMUDGes): AION-1 image tokenizer reconstruction stress test.

Tests the image TOKENIZER's own encode/decode round-trip
(codec_manager.encode -> codec_manager.decode), not the transformer. The
per-object result table is cached (reconstruction_results_cache.csv) so
re-running to tweak a plot doesn't repeat GPU inference.

Port of exp3_reconstruction_stress_test.py to SMUDGes:
  - images are the 96x96 binned cutouts (480 crop, 5x5 bin, 1.31 arcsec/px)
  - truths are the bias-corrected table2.dat values
  - the reconstruction mask is a disc of radius 2 * Re_corr, with Re_corr
    converted to BINNED pixels
  - the old fixed mask-area regimes (1000-2000, >7000 px) don't fit a 96x96
    frame of 4-8 px galaxies, so the histograms compare the lowest and
    highest TERTILES of mask area

Run on your own machine (needs AION weights + GPU). Not run in the sandbox.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from aion_smudges_common import (
    H5_DEFAULT, TABLE2_DEFAULT, BANDS, BINNED_SCALE, load_sample,
    plot_metric_heatmaps,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table2", default=TABLE2_DEFAULT)
    p.add_argument("--h5-path", default=H5_DEFAULT)
    p.add_argument("--out-dir", default="./exp3_results_smudges")
    p.add_argument("--bin-mode", choices=["sum", "mean"], default="mean")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-bins", type=int, default=3)
    p.add_argument("--max-sources", type=int, default=None,
                    help="Optional cap on sample size.")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"reconstruction_results_cache_{args.bin_mode}.csv"

    if cache_path.exists() and not args.force_recompute:
        print(f"Loading cached reconstruction results from {cache_path}")
        df = pd.read_csv(cache_path)
    else:
        df = compute_reconstruction_results(args)
        df.to_csv(cache_path, index=False)
        print(f"Saved reconstruction results to {cache_path}")

    make_plots(df, args, out_dir)


def compute_reconstruction_results(args):
    from aion.codecs import CodecManager
    from aion.modalities import LegacySurveyImage

    matched_df, fluxes, dropped = load_sample(
        args.table2, args.h5_path, args.bin_mode,
        force_recompute=args.force_recompute,
    )
    dropped.to_csv(Path(args.out_dir) / "dropped_missing_band.csv", index=False)
    if args.max_sources:
        matched_df = matched_df.head(args.max_sources)
        fluxes = fluxes[:args.max_sources]
    n_total = len(matched_df)
    print(f"Running reconstruction inference on {n_total} sources ...")

    codec_manager = CodecManager(device=args.device)

    mug = matched_df["mu0g_corr"].to_numpy()
    rhlg = (matched_df["Re_corr"] / BINNED_SCALE).to_numpy()  # binned pixels
    n_sersic = matched_df["n_corr"].to_numpy()

    records = []
    for start in range(0, n_total, args.batch_size):
        end = min(start + args.batch_size, n_total)
        flux_tensor = torch.tensor(fluxes[start:end].astype(np.float32)).to(args.device)
        image = LegacySurveyImage(flux=flux_tensor, bands=BANDS)

        with torch.no_grad():
            tokens = codec_manager.encode(image)
            recon = codec_manager.decode(tokens, LegacySurveyImage, bands=BANDS)

        B, C, H, W = image.flux.shape
        cy, cx = H // 2, W // 2
        batch_rhlg = torch.tensor(rhlg[start:end], device=args.device)
        h_box = torch.clamp((2.0 * batch_rhlg).view(-1, 1, 1), min=3.0, max=float(H // 2))
        y, x = torch.meshgrid(
            torch.arange(H, device=args.device), torch.arange(W, device=args.device),
            indexing="ij",
        )
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= h_box ** 2).unsqueeze(1)

        diff_sq = ((image.flux - recon.flux) ** 2) * mask
        sig_sq = (image.flux ** 2) * mask
        mask_area = mask[:, 0, :, :].sum(dim=(1, 2)).cpu().numpy()
        per_obj_mse = (diff_sq.sum(dim=(1, 2, 3)) / sig_sq.sum(dim=(1, 2, 3))).cpu().numpy()

        for i in range(end - start):
            records.append({
                "name": matched_df["name"].iloc[start + i],
                "mug": mug[start + i],
                "rhlg": rhlg[start + i],
                "reff_arcsec": matched_df["Re_corr"].iloc[start + i],
                "n": n_sersic[start + i],
                "mask_area": mask_area[i],
                "recon_mse": per_obj_mse[i],
            })
        if (start // args.batch_size) % 20 == 0:
            print(f"  {end}/{n_total}")

    df = pd.DataFrame.from_records(records)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mug", "rhlg", "n", "recon_mse"])
    print(f"Kept {len(df)} sources after cleaning")
    return df


def make_plots(df, args, out_dir):
    tag = args.bin_mode

    # 3x3 heatmap grids: (rhlg, mug) and (n, mug), each with a counts panel
    plot_metric_heatmaps(
        df, x_col="rhlg", y_col="mug", metric_cols=["recon_mse"], n_bins=args.n_bins,
        save_path=str(out_dir / f"heatmaps_rhlg_vs_mug_{tag}.png"),
        x_label="Effective radius (binned pixels, 1.31 arcsec/px)",
        y_label="Central surface brightness mu0g (mag/arcsec^2)",
    )
    plot_metric_heatmaps(
        df, x_col="n", y_col="mug", metric_cols=["recon_mse"], n_bins=args.n_bins,
        save_path=str(out_dir / f"heatmaps_n_vs_mug_{tag}.png"),
        x_label="Sersic index (n)",
        y_label="Central surface brightness mu0g (mag/arcsec^2)",
    )

    # mask area vs MSE scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["mask_area"], df["recon_mse"], alpha=0.3, s=5)
    ax.set_xlabel("Mask Area (binned pixels)")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Reconstruction Error vs. Aperture Size")
    fig.tight_layout()
    fig.savefig(out_dir / f"area_vs_mse_scatter_{tag}.png", dpi=150)
    plt.close(fig)

    # step histograms: lowest vs highest tertile of mask area
    lo_cut, hi_cut = df["mask_area"].quantile([1 / 3, 2 / 3])
    mse_small = df[df["mask_area"] <= lo_cut]["recon_mse"]
    mse_large = df[df["mask_area"] >= hi_cut]["recon_mse"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mse_small, bins=20, histtype="step", linewidth=2, density=True,
            label=f"Area <= {lo_cut:.0f} (lowest tertile)")
    ax.hist(mse_large, bins=20, histtype="step", linewidth=2, density=True,
            label=f"Area >= {hi_cut:.0f} (highest tertile)")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error by Mask Area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"mse_area_histogram_{tag}.png", dpi=150)
    plt.close(fig)

    print(f"Saved all plots to {out_dir}")


if __name__ == "__main__":
    main()
