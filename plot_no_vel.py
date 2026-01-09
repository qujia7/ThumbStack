from pixell import enmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- config ---
catalog_file = "/scratch/jiaqu/desi/output/y3_final_renorm/filtered_catalog.csv"
outbase = "/scratch/jiaqu/test_ra0_decpm20_no_vel"
dec_min, dec_max = 0, 12.5     # degrees
ra_center = 140.0              # degrees
ra_width  = 40                 # degrees total width (RA 120-160)

# --- run ---
# Load catalog and filter to the RA/Dec region
df = pd.read_csv(catalog_file)
ra_min = ra_center - ra_width / 2
ra_max = ra_center + ra_width / 2
mask = (df['DEC'] >= dec_min) & (df['DEC'] <= dec_max) & (df['RA'] >= ra_min) & (df['RA'] <= ra_max)
df_crop = df[mask]

print(f"Total galaxies in catalog: {len(df)}")
print(f"Galaxies in RA [{ra_min}, {ra_max}], Dec [{dec_min}, {dec_max}]: {len(df_crop)}")

# Create figure with galaxy positions as black circles
fig, ax = plt.subplots(figsize=(16, 6))

# Plot galaxies as small filled circles - small size with moderate alpha
# so dense regions accumulate darker while sparse regions stay light
ax.scatter(df_crop['RA'], df_crop['DEC'], s=1, facecolors='black', edgecolors='none', alpha=0.5)

ax.set_xlabel('RA [deg]')
ax.set_ylabel('Dec [deg]')
ax.set_xlim(ra_max, ra_min)  # RA increases to the left
ax.set_ylim(dec_min, dec_max)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

os.makedirs(os.path.dirname(outbase), exist_ok=True)
plt.savefig(f"{outbase}.png", dpi=200, bbox_inches='tight')
plt.savefig(f"{outbase}.pdf",dpi=120, bbox_inches='tight')
plt.close()

print(f"Wrote {outbase}.png/.pdf with {len(df_crop)} galaxy positions")