

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSBG Dataset Builder (Resumable) — Legacy Survey DR10-south (griz) version
Downloads Legacy Survey DR10 g, r, i, z image cutouts (DECam-only, southern
footprint) for a catalog of sources and appends them into an existing HDF5
file.

Only processes sources missing from the HDF5 archive.

Notes vs. the DES DR2 version:
  - griz is ONLY available from the DECam ("south") side of Legacy Survey.
    The northern BASS/MzLS imaging has no i-band, so we pin layer to
    'ls-dr10-south' rather than the merged 'ls-dr10' layer — using the
    merged layer would silently fail (missing i) for every source north of
    Dec ~32.375 deg.
  - Even within the DECam footprint, i-band coverage is NOT complete: it
    only exists where non-DECaLS programs (chiefly DES, plus DELVE, BLISS,
    DeROSITAS) observed. Expect a real "missing band" failure rate — those
    sources land in FAILED_CSV as before.
  - This is NOT a byte-for-byte match to your original DES DR2 cutouts:
    Legacy Survey re-reduces the DES exposures (among others) through its
    own Tractor/coaddition pipeline with its own sky subtraction and
    zeropoints, so pixel values and footprint edges will differ from a
    direct DES DR2 pull even where the underlying exposures overlap.
  - The Legacy Survey cutout service is a plain public HTTP endpoint that
    returns a single FITS file with all requested bands already stacked and
    already cropped to the requested pixel size — no SIA search, no
    Data Lab authentication, and no manual Cutout2D step needed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import logging
import os
import random
import time

from astropy.io import fits
import h5py
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — override via environment variables if desired
# ---------------------------------------------------------------------------
CSV_CATALOG = os.environ.get(
    "CSV_CATALOG",
    "/media/duttaa/Data/des_224x224_aplpy/matched_sources_michael.csv",
)
H5_FILENAME = os.environ.get(
    "H5_FILENAME",
    "/media/duttaa/Data/des_224x224_aplpy/lsbg_dataset_michael_ls_256.h5",
)
FAILED_CSV = os.environ.get(
    "FAILED_CSV",
    "/media/duttaa/Data/des_224x224_aplpy/failed_sources_ls.csv",
)

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))  # be polite to the public cutout server
IMAGE_SIZE = 256  # Pixels per side
PIXEL_SCALE = 0.262  # Arcsec/pixel (native Legacy Survey / Tractor pixel scale)
BANDS = "griz"  # i-band only available from DECam (south) side of Legacy Survey
NUM_BANDS = len(BANDS)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

# Legacy Survey cutout service. 'ls-dr10-south' is DECam-only, which is
# required for griz — the merged 'ls-dr10' layer has no i-band north of
# Dec ~32.375 deg (BASS/MzLS), so every northern source would come back
# with a "missing band" error if you used it.
LS_LAYER = os.environ.get("LS_LAYER", "ls-dr10")
LS_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Robust HTTP GET with Exponential Backoff
# ---------------------------------------------------------------------------
def _get_with_retries(url: str, params: dict, timeout: int = 30) -> requests.Response:
  for attempt in range(MAX_RETRIES):
    try:
      resp = requests.get(url, params=params, timeout=timeout)
      resp.raise_for_status()
      return resp
    except requests.RequestException as exc:
      if attempt == MAX_RETRIES - 1:
        raise
      wait = RETRY_BACKOFF * (2**attempt) + random.uniform(0.1, 0.5)
      log.debug(
          "Retry %d/%d after %.1fs — %s", attempt + 1, MAX_RETRIES, wait, exc
      )
      time.sleep(wait)


# ---------------------------------------------------------------------------
# Core download function (one source per call)
# ---------------------------------------------------------------------------
def download_cutout(idx: int, ra: float, dec: float, label: int):
  """Fetches a stacked g,r,i,z FITS cutout from the Legacy Survey viewer and

  returns a (NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE) cube.
  """
  # Stagger requests slightly to avoid burst-flooding the server
  time.sleep(random.uniform(0.05, 0.3))

  params = {
      "ra": ra,
      "dec": dec,
      "layer": LS_LAYER,
      "pixscale": PIXEL_SCALE,
      "bands": BANDS,
      "size": IMAGE_SIZE,
  }

  try:
    resp = _get_with_retries(LS_CUTOUT_URL, params=params)

    with fits.open(BytesIO(resp.content)) as hdul:
      data = hdul[0].data

    if data is None:
      return idx, ra, dec, label, None, "Empty response (likely outside footprint)"

    data = np.asarray(data, dtype=np.float32)

    # Expected shape (NUM_BANDS, size, size). Guard against edge cases where
    # the server returns a 2D array (e.g. a single-band request).
    if data.ndim == 2:
      data = data[np.newaxis, ...]

    if data.shape[0] != NUM_BANDS:
      return idx, ra, dec, label, None, f"Unexpected band count: {data.shape[0]}"

    cube = np.zeros((NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    h, w = data.shape[1], data.shape[2]
    sh, sw = min(h, IMAGE_SIZE), min(w, IMAGE_SIZE)
    cube[:, :sh, :sw] = np.nan_to_num(data[:, :sh, :sw], nan=0.0)

    # All-zero cube usually means the position had no coverage
    if not np.any(cube):
      return idx, ra, dec, label, None, "No data in survey footprint"

    return idx, ra, dec, label, cube, None

  except Exception as exc:
    return idx, ra, dec, label, None, str(exc)


# ---------------------------------------------------------------------------
# Main Resumable Pipeline
# ---------------------------------------------------------------------------
def build_hdf5_dataset():
  # -- 1. Load CSV catalog -------------------------------------------------
  log.info("Loading catalog from %s", CSV_CATALOG)
  df = pd.read_csv(CSV_CATALOG)

  if "label" not in df.columns:
    log.warning("No 'label' column found — setting label=1.")
    df["label"] = 1

  total_catalog = len(df)
  log.info("Total sources in CSV: %d", total_catalog)

  # -- 2. Inspect existing HDF5 file and find missing sources --------------
  file_exists = os.path.exists(H5_FILENAME)

  with h5py.File(H5_FILENAME, "a") as h5f:
    if not file_exists or "fluxes" not in h5f:
      log.info("Creating new dataset structure in %s", H5_FILENAME)
      h5f.attrs["image_size"] = IMAGE_SIZE
      h5f.attrs["pixel_scale"] = PIXEL_SCALE
      h5f.attrs["bands"] = list(BANDS)
      h5f.attrs["survey"] = f"Legacy Survey {LS_LAYER}"

      dset_ra = h5f.create_dataset(
          "ra", shape=(0,), maxshape=(None,), dtype="f8"
      )
      dset_dec = h5f.create_dataset(
          "dec", shape=(0,), maxshape=(None,), dtype="f8"
      )
      dset_label = h5f.create_dataset(
          "label", shape=(0,), maxshape=(None,), dtype="i8"
      )
      dset_flux = h5f.create_dataset(
          "fluxes",
          shape=(0, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE),
          maxshape=(None, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE),
          dtype="f4",
          chunks=(16, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE),
          compression="gzip",
          compression_opts=4,
      )
      already_saved = set()
      current_count = 0
    else:
      dset_ra = h5f["ra"]
      dset_dec = h5f["dec"]
      dset_label = h5f["label"]
      dset_flux = h5f["fluxes"]

      current_count = len(dset_ra)
      # Coordinate match up to 6 decimal places (~0.036 arcsec precision)
      existing_ras = np.round(dset_ra[:], 6)
      existing_decs = np.round(dset_dec[:], 6)
      already_saved = set(zip(existing_ras, existing_decs))
      log.info(
          "Found %d sources already stored in %s", current_count, H5_FILENAME
      )

    # Filter catalog for missing entries
    df["ra_round"] = df["ra"].round(6)
    df["dec_round"] = df["dec"].round(6)
    is_missing = ~df.apply(
        lambda r: (r["ra_round"], r["dec_round"]) in already_saved, axis=1
    )
    remaining_df = df[is_missing].reset_index()
    num_to_download = len(remaining_df)

    log.info(
        "Already completed: %d | Remaining to retry: %d",
        len(already_saved),
        num_to_download,
    )

    if num_to_download == 0:
      log.info("All sources are already stored. Nothing left to download.")
      return

    # -- 3. Retry missing cutouts in parallel ------------------------------
    new_success = 0
    failed_sources = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
      futures = {
          executor.submit(
              download_cutout,
              row["index"],
              row["ra"],
              row["dec"],
              int(row["label"]),
          ): row
          for _, row in remaining_df.iterrows()
      }

      for future in tqdm(
          as_completed(futures),
          total=num_to_download,
          desc="Retrying missing",
      ):
        idx, ra, dec, label, cube, err = future.result()

        if cube is not None:
          n = current_count + 1
          dset_ra.resize(n, axis=0)
          dset_dec.resize(n, axis=0)
          dset_label.resize(n, axis=0)
          dset_flux.resize(n, axis=0)

          dset_ra[current_count] = ra
          dset_dec[current_count] = dec
          dset_label[current_count] = label
          dset_flux[current_count] = cube

          current_count += 1
          new_success += 1
        else:
          failed_sources.append(
              {"index": idx, "ra": ra, "dec": dec, "error": err}
          )
          if len(failed_sources) <= 10:
            tqdm.write(
                f"  FAILED id={idx:05d} ra={ra:.4f} dec={dec:.4f} | {err}"
            )

  # -- 4. Log Summary & Export Permanent Failures --------------------------
  log.info("Retry run finished.")
  log.info("  Newly Added: %d", new_success)
  log.info("  Total in HDF5: %d / %d", current_count, total_catalog)
  log.info("  Still Failing: %d", len(failed_sources))

  if failed_sources:
    fail_df = pd.DataFrame(failed_sources)
    fail_df.to_csv(FAILED_CSV, index=False)
    log.info("Exported failure details to: %s", FAILED_CSV)


if __name__ == "__main__":
  build_hdf5_dataset()