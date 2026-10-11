#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMUDGes Cutout Downloader (Resumable) — Legacy Survey DR9 (grz) version
Downloads 1024x1024 Legacy Survey g, r, z image cutouts at the RA/Dec of every
UDG candidate in the SMUDGes catalog (Zaritsky+ 2023, J/ApJS/267/27,
catalog_folder/table2.dat) and appends them into an HDF5 file.

Only processes sources missing from the HDF5 archive, so it can be re-run
after an interruption.

Notes:
  - The SMUDGes catalog was built from Legacy Survey DR9 (DECaLS + MzLS/BASS),
    so we use the merged 'ls-dr9' layer and the g, r, z bands (no i-band in
    the north, and the catalog photometry is g, r, z).
  - 1024 px * 0.262 arcsec/px = ~268 arcsec on a side.
  - The cutout service returns one FITS file with all bands stacked.
  - Size: 3 x 1024 x 1024 float32 = ~12.6 MB per source before compression;
    the full catalog (7070 sources) is ~89 GB uncompressed.
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TABLE_DAT = os.environ.get(
    "TABLE_DAT", os.path.join(BASE_DIR, "catalog_folder", "table2.dat")
)
H5_FILENAME = os.environ.get(
    "H5_FILENAME", os.path.join(BASE_DIR, "smudges_ls_dr10_512.h5")
)
FAILED_CSV = os.environ.get(
    "FAILED_CSV", os.path.join(BASE_DIR, "failed_smudges_ls_dr10.csv")
)

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))  # be polite to the public cutout server
IMAGE_SIZE = 512  # Pixels per side
PIXEL_SCALE = 0.262  # Arcsec/pixel (native Legacy Survey / Tractor pixel scale)
BANDS = "griz"
NUM_BANDS = len(BANDS)
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

LS_LAYER = os.environ.get("LS_LAYER", "ls-dr10")
LS_CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# Set e.g. LIMIT=10 to test on the first 10 sources only
LIMIT = int(os.environ.get("LIMIT", 0))

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
# Catalog reader (fixed-width table2.dat, see ReadMe byte-by-byte description)
# ---------------------------------------------------------------------------
def load_catalog(path: str) -> pd.DataFrame:
  # 1-based bytes 5-18 = name, 20-28 = RAdeg, 30-38 = DEdeg
  df = pd.read_fwf(
      path,
      colspecs=[(0, 18), (19, 28), (29, 38)],
      names=["name", "ra", "dec"],
      dtype={"name": str},
  )
  return df


# ---------------------------------------------------------------------------
# Helper: Robust HTTP GET with Exponential Backoff
# ---------------------------------------------------------------------------
def _get_with_retries(url: str, params: dict, timeout: int = 120) -> requests.Response:
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
def download_cutout(idx: int, name: str, ra: float, dec: float):
  """Fetches a stacked g,r,z FITS cutout from the Legacy Survey viewer and

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
      return idx, name, ra, dec, None, "Empty response (likely outside footprint)"

    data = np.asarray(data, dtype=np.float32)

    if data.ndim == 2:
      data = data[np.newaxis, ...]

    if data.shape[0] != NUM_BANDS:
      return idx, name, ra, dec, None, f"Unexpected band count: {data.shape[0]}"

    cube = np.zeros((NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    h, w = data.shape[1], data.shape[2]
    sh, sw = min(h, IMAGE_SIZE), min(w, IMAGE_SIZE)
    cube[:, :sh, :sw] = np.nan_to_num(data[:, :sh, :sw], nan=0.0)

    # All-zero cube usually means the position had no coverage
    if not np.any(cube):
      return idx, name, ra, dec, None, "No data in survey footprint"

    return idx, name, ra, dec, cube, None

  except Exception as exc:
    return idx, name, ra, dec, None, str(exc)


# ---------------------------------------------------------------------------
# Main Resumable Pipeline
# ---------------------------------------------------------------------------
def build_hdf5_dataset():
  # -- 1. Load catalog -----------------------------------------------------
  log.info("Loading catalog from %s", TABLE_DAT)
  df = load_catalog(TABLE_DAT)
  if LIMIT:
    df = df.head(LIMIT)

  total_catalog = len(df)
  log.info("Total sources in catalog: %d", total_catalog)

  # -- 2. Inspect existing HDF5 file and find missing sources --------------
  file_exists = os.path.exists(H5_FILENAME)
  str_dtype = h5py.string_dtype(encoding="utf-8")

  with h5py.File(H5_FILENAME, "a") as h5f:
    if not file_exists or "fluxes" not in h5f:
      log.info("Creating new dataset structure in %s", H5_FILENAME)
      h5f.attrs["image_size"] = IMAGE_SIZE
      h5f.attrs["pixel_scale"] = PIXEL_SCALE
      h5f.attrs["bands"] = list(BANDS)
      h5f.attrs["survey"] = f"Legacy Survey {LS_LAYER}"

      dset_name = h5f.create_dataset(
          "name", shape=(0,), maxshape=(None,), dtype=str_dtype
      )
      dset_ra = h5f.create_dataset(
          "ra", shape=(0,), maxshape=(None,), dtype="f8"
      )
      dset_dec = h5f.create_dataset(
          "dec", shape=(0,), maxshape=(None,), dtype="f8"
      )
      dset_flux = h5f.create_dataset(
          "fluxes",
          shape=(0, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE),
          maxshape=(None, NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE),
          dtype="f4",
          chunks=(1, NUM_BANDS, 256, 256),
          compression="gzip",
          compression_opts=4,
      )
      already_saved = set()
      current_count = 0
    else:
      dset_name = h5f["name"]
      dset_ra = h5f["ra"]
      dset_dec = h5f["dec"]
      dset_flux = h5f["fluxes"]

      current_count = len(dset_ra)
      already_saved = set(
          n.decode() if isinstance(n, bytes) else n for n in dset_name[:]
      )
      log.info(
          "Found %d sources already stored in %s", current_count, H5_FILENAME
      )

    # Filter catalog for missing entries (matched on unique SMDG name)
    remaining_df = df[~df["name"].isin(already_saved)].reset_index()
    num_to_download = len(remaining_df)

    log.info(
        "Already completed: %d | Remaining to download: %d",
        len(already_saved),
        num_to_download,
    )

    if num_to_download == 0:
      log.info("All sources are already stored. Nothing left to download.")
      return

    # -- 3. Download missing cutouts in parallel ---------------------------
    new_success = 0
    failed_sources = []

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
      futures = {
          executor.submit(
              download_cutout,
              row["index"],
              row["name"],
              row["ra"],
              row["dec"],
          ): row
          for _, row in remaining_df.iterrows()
      }

      for future in tqdm(
          as_completed(futures),
          total=num_to_download,
          desc="Downloading",
      ):
        idx, name, ra, dec, cube, err = future.result()

        if cube is not None:
          n = current_count + 1
          dset_name.resize(n, axis=0)
          dset_ra.resize(n, axis=0)
          dset_dec.resize(n, axis=0)
          dset_flux.resize(n, axis=0)

          dset_name[current_count] = name
          dset_ra[current_count] = ra
          dset_dec[current_count] = dec
          dset_flux[current_count] = cube

          current_count += 1
          new_success += 1
        else:
          failed_sources.append(
              {"index": idx, "name": name, "ra": ra, "dec": dec, "error": err}
          )
          if len(failed_sources) <= 10:
            tqdm.write(
                f"  FAILED id={idx:05d} {name} ra={ra:.4f} dec={dec:.4f} | {err}"
            )

  # -- 4. Log Summary & Export Failures ------------------------------------
  log.info("Run finished.")
  log.info("  Newly Added: %d", new_success)
  log.info("  Total in HDF5: %d / %d", current_count, total_catalog)
  log.info("  Failing: %d", len(failed_sources))

  if failed_sources:
    fail_df = pd.DataFrame(failed_sources)
    fail_df.to_csv(FAILED_CSV, index=False)
    log.info("Exported failure details to: %s", FAILED_CSV)


if __name__ == "__main__":
  build_hdf5_dataset()
