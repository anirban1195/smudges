import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

base = os.path.dirname(os.path.abspath(__file__))
df = pd.read_fwf(os.path.join(base, "catalog_folder", "table2.dat"),
                 colspecs=[(39, 45)], names=["Re"])
re = df["Re"]
print(re.describe())
print("Re > 134 arcsec (would exceed half of 268\" cutout):", (re > 134).sum())

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(re, bins=60, color="steelblue", edgecolor="k", lw=0.3)
ax.axvline(re.median(), color="r", ls="--", label=f"median = {re.median():.1f}\"")
ax.set_xlabel("Effective radius R$_e$ (arcsec)")
ax.set_ylabel("Number of galaxies")
ax.set_title(f"SMUDGes table2.dat (N={len(re)})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(base, "re_histogram.png"), dpi=150)
