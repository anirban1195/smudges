#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1 (SMUDGes): structural parameter regression from frozen AION-1
embeddings.

  1. Load table2.dat (bias-corrected truths, VC==0, size cut) and join to the
     512x512 griz H5 by SMDG name; centre-crop 480, bin 5x5 -> 96x96; drop
     sources with an empty / mostly-zero band.
  2. Extract (or load cached) frozen AION-1 embeddings.
  3. 70/30 train/test split.
  4. Train BOTH a linear head and a 2-layer(256) MLP head to predict
     [Re_corr (arcsec), n_corr, mu0g_corr, b/a_corr] from the embedding.
  5. Report Bias / RMSE / R^2 / NMAD / MAE per target, per head, next to the
     catalog's own median 1-sigma uncertainty (a floor on achievable RMSE).
  6. Save per-object predictions (keyed by SMDG name).
  7. 3x3 quantile-binned heatmaps of |residual| vs truth Re (x) and truth
     mu0g (y), plus a sample-count panel.

Run on your own machine (needs AION weights + GPU). Not run in the sandbox.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from aion_smudges_common import (
    H5_DEFAULT, TABLE2_DEFAULT, load_sample, get_or_extract_embeddings,
    train_regression_head, regression_metrics_table, plot_metric_heatmaps,
)

# friendly target name -> (bias-corrected truth column, catalog 1-sigma column)
TRUTH_COLS = {
    "reff": "Re_corr",
    "n": "n_corr",
    "mu0g": "mu0g_corr",
    "q": "ba_corr",
}
SIGMA_COLS = {
    "reff": "Re_sig",
    "n": "n_sig",
    "mu0g": "mu0g_sig",
    "q": "ba_sig",
}
TARGET_NAMES = list(TRUTH_COLS.keys())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--table2", default=TABLE2_DEFAULT)
    p.add_argument("--h5-path", default=H5_DEFAULT)
    p.add_argument("--model-path", default="/home/duttaa/aion/aion-base")
    p.add_argument("--out-dir", default="./exp1_results_smudges")
    p.add_argument("--bin-mode", choices=["sum", "mean"], default="sum",
                    help="How the 5x5 bins are combined (sum conserves flux; "
                         "mean keeps surface brightness per native pixel).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-bins", type=int, default=3)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-recompute", action="store_true",
                    help="Rebuild the binned cubes and the embeddings.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, fluxes, dropped = load_sample(
        args.table2, args.h5_path, args.bin_mode,
        force_recompute=args.force_recompute,
    )
    dropped.to_csv(out_dir / "dropped_missing_band.csv", index=False)
    if len(df) < 50:
        print(f"[WARN] only {len(df)} usable objects -- aborting")
        return

    y = df[list(TRUTH_COLS.values())].to_numpy(dtype=np.float32)

    embeddings = get_or_extract_embeddings(
        fluxes, df["name"].to_numpy(), out_dir / f"embeddings_{args.bin_mode}.npz",
        args.model_path, args.device, batch_size=args.batch_size,
        force_recompute=args.force_recompute,
    )

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=args.test_size, random_state=args.seed
    )
    X_train, X_test = embeddings[idx_train], embeddings[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    all_metrics = []
    per_object = df.iloc[idx_test].copy().reset_index(drop=True)
    catalog_sigma = [np.nanmedian(per_object[SIGMA_COLS[t]]) for t in TARGET_NAMES]

    for head_label, hidden in [("linear", 0), ("mlp2_256", 256)]:
        print(f"training head={head_label} (hidden={hidden}) "
              f"on {len(X_train)} train / {len(X_test)} test ...")
        _, y_pred = train_regression_head(
            X_train, y_train, X_test, y_test, args.device,
            hidden=hidden, epochs=args.epochs,
        )
        metrics_df = regression_metrics_table(y_test, y_pred, TARGET_NAMES)
        metrics_df["catalog_sigma"] = catalog_sigma
        metrics_df.insert(0, "head", head_label)
        all_metrics.append(metrics_df)

        for i, t in enumerate(TARGET_NAMES):
            per_object[f"{t}_pred_{head_label}"] = y_pred[:, i]
            per_object[f"{t}_absresid_{head_label}"] = np.abs(y_pred[:, i] - y_test[:, i])

    metrics_out = pd.concat(all_metrics, ignore_index=True)
    metrics_out.to_csv(out_dir / "metrics.csv", index=False)
    per_object.to_csv(out_dir / "predictions.csv", index=False)
    print(metrics_out.round(4).to_string(index=False))

    # 3x3 heatmaps (2-layer head's absolute residual as the per-object
    # statistic; truth Re / mu0g used for binning)
    metric_cols = [f"{t}_absresid_mlp2_256" for t in TARGET_NAMES]
    plot_metric_heatmaps(
        per_object, x_col=TRUTH_COLS["reff"], y_col=TRUTH_COLS["mu0g"],
        metric_cols=metric_cols, n_bins=args.n_bins,
        save_path=str(out_dir / "heatmaps.png"),
        x_label="Effective radius (truth, bias-corrected, arcsec)",
        y_label="Central surface brightness mu0g (truth, bias-corrected, mag/arcsec^2)",
    )
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
