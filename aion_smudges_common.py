#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aion_smudges_common.py

Shared plumbing for the AION-1 SMUDGes benchmarking suite (port of
aion_lsbg_common.py to the SMUDGes catalog, Zaritsky+ 2023, table2.dat).

  - table2.dat reader with bias-corrected truths (corrected = value - bias),
    VC / sentinel / size-cut filtering
  - H5 (download_smudges_cutouts_dr10.py output, 512x512 griz) -> central
    480x480 crop -> 5x5 bin -> 96x96, cached to disk
  - per-channel missing-data check (all-zero / NaN / mostly-zero bands are
    dropped)
  - frozen AION-1 embedding extraction, cached to disk
  - MLP head, regression training, metrics, 3x3 heatmaps (unchanged from
    aion_lsbg_common.py)

Run on your own machine (needs AION weights + GPU). Not run in the sandbox.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
H5_DEFAULT = str(BASE_DIR / "smudges_ls_dr10_512.h5")
TABLE2_DEFAULT = str(BASE_DIR / "catalog_folder" / "table2.dat")

BANDS = ["DES-G", "DES-R", "DES-I", "DES-Z"]  # H5 band order is g, r, i, z
H5_BANDS = list("griz")

PIXEL_SCALE = 0.262                       # arcsec / native pixel
CROP_SIZE = 480                           # central crop of the 512x512 cutout
BIN_FACTOR = 5                            # 5x5 binning
OUT_SIZE = CROP_SIZE // BIN_FACTOR        # 96, what AION's tokenizer uses
BINNED_SCALE = PIXEL_SCALE * BIN_FACTOR   # 1.31 arcsec / binned pixel
CUTOUT_ARCSEC = OUT_SIZE * BINNED_SCALE   # 125.76 arcsec on a side
MIN_RE_PER_CUTOUT = 6.0                   # cutout width must be >= 6 Re
MAX_MISSING_FRAC = 0.01                   # max fraction of zero/NaN pixels per band
IMAGE_NUM_TOKENS = 576                    # 96x96 image at AION's patch grid

SENTINEL = -99.0  # table2.dat uses -99.99 for "no value"


# ---------------------------------------------------------------------------
# table2.dat
# ---------------------------------------------------------------------------
# 1-based inclusive byte ranges from catalog_folder/ReadMe
_TABLE2_COLS = {
    "name": (1, 18),  # includes the literal "SMDG" prefix (bytes 1-4)
    "ra": (20, 28), "dec": (30, 38),
    "Re": (40, 45), "E_Re": (47, 52), "Re_bias": (54, 59), "e_Re": (61, 66),
    "ba": (70, 73), "E_ba": (75, 80), "ba_bias": (82, 87), "e_ba": (89, 94),
    "n": (98, 101), "E_n": (103, 108), "n_bias": (110, 115), "e_n": (117, 122),
    "mu0g": (156, 161), "E_mu0g": (163, 168), "mu0g_bias": (170, 175),
    "e_mu0g": (177, 182),
    "VC": (336, 336), "Ref": (401, 402),
}
# raw column -> corrected column (corrected = value - bias, per the ReadMe)
_CORRECTED = {"Re": "Re_corr", "ba": "ba_corr", "n": "n_corr", "mu0g": "mu0g_corr"}


def load_table2(path=TABLE2_DEFAULT, max_re_arcsec=None):
    """
    Read table2.dat and return one row per usable source with bias-corrected
    Re (arcsec), b/a, Sersic n and mu0g, plus a symmetric 1-sigma per
    parameter (<raw>_sig = mean(|upper|, |lower|)).

    Dropped: VC != 0 (visually rejected), any missing/sentinel value or bias,
    non-positive corrected Re/n/b-a, and Re_corr > max_re_arcsec (default:
    large enough that the 96x96 binned cutout is narrower than
    MIN_RE_PER_CUTOUT * Re).
    """
    df = pd.read_fwf(
        path, colspecs=[(a - 1, b) for a, b in _TABLE2_COLS.values()],
        names=list(_TABLE2_COLS), dtype={"name": str, "Ref": str},
    )
    print(f"table2.dat: {len(df)} rows")

    df = df[df["VC"] == 0]
    print(f"  after VC == 0: {len(df)}")

    for raw, corr in _CORRECTED.items():
        ok = (df[raw] > SENTINEL) & (df[f"{raw}_bias"] > SENTINEL)
        df[corr] = np.where(ok, df[raw] - df[f"{raw}_bias"], np.nan)
        sig_ok = (df[f"E_{raw}"] > SENTINEL) & (df[f"e_{raw}"] > SENTINEL)
        df[f"{raw}_sig"] = np.where(
            sig_ok, 0.5 * (df[f"E_{raw}"].abs() + df[f"e_{raw}"].abs()), np.nan)
    df = df.dropna(subset=list(_CORRECTED.values()))
    print(f"  after dropping missing value/bias: {len(df)}")

    df = df[(df["Re_corr"] > 0) & (df["n_corr"] > 0) & (df["ba_corr"] > 0)]
    print(f"  after requiring corrected Re, n, b/a > 0: {len(df)}")

    max_re = max_re_arcsec or CUTOUT_ARCSEC / MIN_RE_PER_CUTOUT
    df = df[df["Re_corr"] <= max_re]
    print(f"  after size cut (Re_corr <= {max_re:.2f} arcsec): {len(df)}")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# H5 -> central crop -> bin, cached
# ---------------------------------------------------------------------------
def _as_str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_h5_names(h5_path):
    import h5py
    with h5py.File(h5_path, "r") as h5f:
        names = np.array([_as_str(n) for n in h5f["name"][:]])
        size = int(h5f.attrs["image_size"])
        bands = [_as_str(b) for b in h5f.attrs["bands"]]
    if bands != H5_BANDS:
        raise ValueError(f"H5 bands are {bands}, expected {H5_BANDS}")
    if size < CROP_SIZE:
        raise ValueError(f"H5 image_size {size} is smaller than CROP_SIZE {CROP_SIZE}")
    print(f"H5 {h5_path}: {len(names)} sources, {size}x{size}, bands {bands}")
    return names


def _bin(x, bin_mode, bin_factor):
    n, c = x.shape[:2]
    x = x.reshape(n, c, OUT_SIZE, bin_factor, OUT_SIZE, bin_factor)
    if bin_mode == "sum":
        return x.sum(axis=(3, 5), dtype=np.float32)
    return x.mean(axis=(3, 5), dtype=np.float32)


def get_or_build_binned(h5_path, bin_mode, bin_factor=BIN_FACTOR,
                        force_recompute=False, chunk=32):
    """
    Central-crop every H5 cutout to OUT_SIZE * bin_factor pixels and bin
    bin_factor x bin_factor (sum or mean) to (4, 96, 96). bin_factor=5 is
    the 480 -> 96 binned case; bin_factor=1 is the unbinned central 96x96
    (25 arcsec) at native 0.262 arcsec/px. Also records, per source and
    band, the fraction of pixels in the central CROP_SIZE x CROP_SIZE region
    that are exactly 0 or non-finite -- always that same 480 region, so every
    bin_factor sees the identical set of dropped sources ("missing";
    the downloader turned NaN into 0, and real sky-subtracted pixels are
    essentially never exactly 0.0). A band with zero variance counts as
    fully missing. The result is cached next to the H5 file.

    Returns (names, binned (N,4,96,96), missing_frac (N,4)).
    """
    import h5py
    assert bin_mode in ("sum", "mean") and 1 <= bin_factor <= BIN_FACTOR
    h5_path = Path(h5_path)
    suffix = "" if bin_factor == BIN_FACTOR else f"_x{bin_factor}"
    cache_path = h5_path.with_name(f"{h5_path.stem}_binned_{bin_mode}{suffix}.npz")
    names = load_h5_names(h5_path)

    if cache_path.exists() and not force_recompute:
        cache = np.load(cache_path)
        if np.array_equal(cache["names"], names):
            print(f"  loaded cached binned cubes from {cache_path}")
            return names, cache["binned"], cache["missing"]
        print(f"  cache {cache_path} is out of date with the H5 -- rebuilding")

    crop = OUT_SIZE * bin_factor
    off = (CROP_SIZE - crop) // 2
    print(f"  building {bin_mode}-binned cubes ({crop} -> {OUT_SIZE}, x{bin_factor}) ...")
    with h5py.File(h5_path, "r") as h5f:
        dset = h5f["fluxes"]
        N, C, H, W = dset.shape
        y0, x0 = (H - CROP_SIZE) // 2, (W - CROP_SIZE) // 2
        binned = np.zeros((N, C, OUT_SIZE, OUT_SIZE), dtype=np.float32)
        missing = np.zeros((N, C), dtype=np.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = dset[s:e, :, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE].astype(np.float32)
            finite = np.isfinite(x)
            miss = (~finite | (x == 0)).mean(axis=(2, 3))
            x = np.where(finite, x, 0.0)
            miss[x.reshape(e - s, C, -1).std(axis=2) == 0] = 1.0
            missing[s:e] = miss
            binned[s:e] = _bin(x[:, :, off:off + crop, off:off + crop], bin_mode, bin_factor)
            if (s // chunk) % 20 == 0:
                print(f"    {e}/{N}")

    np.savez(cache_path, names=names, binned=binned, missing=missing)
    print(f"  saved {cache_path}")
    return names, binned, missing


def load_sample(table_path, h5_path, bin_mode, bin_factor=BIN_FACTOR,
                max_re_arcsec=None, max_missing_frac=MAX_MISSING_FRAC,
                force_recompute=False):
    """
    table2 (filtered, bias-corrected) joined by SMDG name to the binned H5
    cubes, with sources dropped if ANY band has more than max_missing_frac
    zero/NaN pixels.

    Returns (df, fluxes (N,4,96,96), dropped_df) aligned row-for-row.
    """
    table = load_table2(table_path, max_re_arcsec)
    h5_names, binned, missing = get_or_build_binned(
        h5_path, bin_mode, bin_factor, force_recompute=force_recompute)

    pos_lookup = {n: i for i, n in enumerate(h5_names)}
    table["_pos"] = table["name"].map(pos_lookup)
    table = table.dropna(subset=["_pos"]).reset_index(drop=True)
    pos = table["_pos"].to_numpy().astype(int)
    print(f"  {len(table)} table2 sources found in the H5")

    bad = missing[pos] > max_missing_frac
    empty = missing[pos] >= 0.999
    for b, n_bad, n_empty in zip(H5_BANDS, bad.sum(axis=0), empty.sum(axis=0)):
        print(f"    band {b}: {n_bad} with >{max_missing_frac:.0%} missing pixels "
              f"({n_empty} fully empty)")
    drop = bad.any(axis=1)
    print(f"  dropping {drop.sum()} sources with a missing/empty band; "
          f"{(~drop).sum()} remain")

    dropped = table.loc[drop, ["name", "ra", "dec"]].copy()
    for j, b in enumerate(H5_BANDS):
        dropped[f"missing_{b}"] = missing[pos[drop], j]

    df = table[~drop].drop(columns="_pos").reset_index(drop=True)
    return df, binned[pos[~drop]], dropped


# ---------------------------------------------------------------------------
# Embedding extraction, cached to disk
# ---------------------------------------------------------------------------
def get_or_extract_embeddings(fluxes, names, cache_path, model_path, device,
                               batch_size=64, num_encoder_tokens=IMAGE_NUM_TOKENS,
                               force_recompute=False):
    """
    Load cached AION-1 embeddings from `cache_path` if they exist and were
    computed for exactly these `names` (same order); otherwise run the frozen
    AION-1 encoder once and cache the result.
    """
    cache_path = Path(cache_path)
    names = np.asarray(names, dtype=str)  # df["name"] is an object array; keep the npz pickle-free
    if not force_recompute and cache_path.exists():
        # allow_pickle: caches written before this fix stored names as an object array
        cache = np.load(cache_path, allow_pickle=True)
        if np.array_equal(cache["names"].astype(str), names):
            print(f"  loaded cached embeddings from {cache_path} "
                  f"({len(fluxes)} objects)")
            return cache["embeddings"]
        print(f"  cache at {cache_path} was built for a different sample -- recomputing")

    from aion import AION
    from aion.codecs import CodecManager
    from aion.modalities import LegacySurveyImage

    print(f"  extracting AION-1 embeddings for {len(fluxes)} objects "
          f"(model={model_path}, device={device}) ...")
    model = AION.from_pretrained(model_path).to(device)
    model.eval()
    codec_manager = CodecManager(device=device)

    all_emb = []
    n_total = fluxes.shape[0]
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        flux_tensor = torch.tensor(fluxes[start:end].astype(np.float32)).to(device)
        image = LegacySurveyImage(flux=flux_tensor, bands=BANDS)

        with torch.no_grad():
            tokens = codec_manager.encode(image)
            emb = model.encode(tokens, num_encoder_tokens=num_encoder_tokens)
            pooled = torch.mean(emb, dim=1)  # mean pooling

        all_emb.append(pooled.cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"    {end}/{n_total}")

    embeddings = np.concatenate(all_emb, axis=0)
    np.savez(cache_path, embeddings=embeddings, names=names)
    print(f"  saved embeddings to {cache_path}")
    return embeddings


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """hidden=0 -> plain linear head. hidden>0 -> Linear-ReLU-Dropout-Linear."""

    def __init__(self, in_dim, out_dim, hidden=0, dropout=0.25):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Regression training
# ---------------------------------------------------------------------------
def train_regression_head(X_train, y_train, X_val, y_val, device, hidden=0,
                           epochs=500, lr=5e-4, patience=100):
    in_dim, out_dim = X_train.shape[1], y_train.shape[1]

    x_mean, x_std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-8
    y_mean, y_std = y_train.mean(0, keepdims=True), y_train.std(0, keepdims=True) + 1e-8

    Xtr = torch.tensor((X_train - x_mean) / x_std, dtype=torch.float32, device=device)
    Ytr = torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32, device=device)
    Xva = torch.tensor((X_val - x_mean) / x_std, dtype=torch.float32, device=device)
    Yva = torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32, device=device)

    model = MLPHead(in_dim, out_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr), Ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()

        if val_loss < best_val:
            best_val, best_state, bad = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xva).cpu().numpy() * y_std + y_mean
    return model, val_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def nmad(residuals):
    """Normalized Median Absolute Deviation."""
    return 1.4826 * np.median(np.abs(residuals - np.median(residuals)))


def regression_metrics_table(y_true, y_pred, targets):
    rows = []
    for i, t in enumerate(targets):
        yt, yp = y_true[:, i], y_pred[:, i]
        resid = yp - yt
        rows.append({
            "Target": t,
            "Bias": np.mean(resid),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "R2": r2_score(yt, yp),
            "NMAD": nmad(resid),
            "MAE": mean_absolute_error(yt, yp),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting: 3x3 metric heatmaps + counts
# ---------------------------------------------------------------------------
def plot_metric_heatmaps(df, x_col, y_col, metric_cols, n_bins=3,
                          save_path="metric_heatmaps.png",
                          x_label=None, y_label=None):
    """
    3x3 (or nxn) quantile-binned heatmaps of each metric in metric_cols
    as a function of truth x_col (columns) vs truth y_col (rows), plus a
    final panel showing sample counts per bin.
    """
    d = df.copy()
    d["_xbin"] = pd.qcut(d[x_col], q=n_bins, duplicates="drop")
    d["_ybin"] = pd.qcut(d[y_col], q=n_bins, duplicates="drop")

    n_panels = len(metric_cols) + 1
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    def _draw(ax, pivot, title, cmap):
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"({c.left:.2f},{c.right:.2f}]" for c in pivot.columns],
                            rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f"({i.left:.2f},{i.right:.2f}]" for i in pivot.index],
                            fontsize=8)
        ax.set_xlabel(x_label or x_col)
        ax.set_ylabel(y_label or y_col)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for (r, c), val in np.ndenumerate(pivot.values):
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2g}", ha="center", va="center",
                        color="white", fontsize=7)

    for i, metric in enumerate(metric_cols):
        pivot = d.pivot_table(values=metric, index="_ybin", columns="_xbin",
                               aggfunc="median", observed=True)
        _draw(axes[i], pivot, metric, "viridis")

    counts = d.pivot_table(values=metric_cols[0], index="_ybin", columns="_xbin",
                            aggfunc="count", observed=True)
    _draw(axes[len(metric_cols)], counts, "N samples", "plasma")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  saved heatmap grid to {save_path}")
