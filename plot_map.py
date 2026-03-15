from pixell import enmap, enplot, utils
import numpy as np
import pandas as pd
import os
import argparse
#python3 plot_map.py --ticks 5 --downgrade 7
# --- argument parsing ---
parser = argparse.ArgumentParser(description="Plot enmap with customizable grid ticks")
parser.add_argument("-t", "--ticks", type=str, default="5",
                    help="The grid spacing in degrees. Either a single number to be used for both axes, or ty,tx.")
parser.add_argument("--tick-unit", "--tu", type=str, default=None,
                    help="Units for tick axis. Can be the unit size in degrees, or the word 'degree', 'arcmin' or 'arcsec' or the shorter 'd','m','s'.")
parser.add_argument("--font-size", type=int, default=20,
                    help="Font size for tick labels in pixels (default: 20)")
parser.add_argument("-d", "--downgrade", type=int, default=7,
                    help="Downgrade factor for resolution (default: 7, use 1 for full resolution)")
args = parser.parse_args()

# --- config ---
infile  = "/home/jiaqu/Thumbstack_DESI/output/z_all_new_mask/stage_template/template_car_nosrc_sub_mask.fits"
catalog_file = "/scratch/jiaqu/desi/output/zall_mask_no_src_with_cluster/filtered_catalog.csv"

outbase = "/scratch/jiaqu/test_ra0_decpm20"
dec_min, dec_max = 0, 12.5     # degrees
ra_min_deg, ra_max_deg = 120.0, 160.0

# --- helpers ---
def parse_tick_unit(tick_unit):
    """Parse tick_unit argument into a value usable by enplot."""
    if tick_unit is None:
        return 1.0
    try:
        return float(tick_unit)
    except ValueError:
        return tick_unit

# --- run ---
# Read only the submap directly from disk (much faster than reading full map)
box = np.deg2rad([[dec_min, ra_min_deg], [dec_max, ra_max_deg]])
m_crop = enmap.read_map(infile, box=box)

# Colorbar range
cbar_min = -0.002
cbar_max = 0.0025

os.makedirs(os.path.dirname(outbase), exist_ok=True)

# Plot template map
plot_kwargs = dict(
    downgrade=args.downgrade,
    colorbar=True,
    color="planck",
    grid=True,
    grid_width=1,
    ticks=args.ticks,
    font_size=args.font_size,
    mask=0,
    min=cbar_min,
    max=cbar_max,
)
if args.tick_unit is not None:
    plot_kwargs["tick_unit"] = parse_tick_unit(args.tick_unit)

p = enplot.plot(m_crop, **plot_kwargs)
enplot.write(outbase, p)
print(f"Wrote {outbase}.png/.pdf   crop shape={m_crop.shape}")
print(f"Color range: {cbar_min} to {cbar_max}")

# --- catalogue density map on same geometry ---
df = pd.read_csv(catalog_file)
mask_cat = (df['DEC'] >= dec_min) & (df['DEC'] <= dec_max) & (df['RA'] >= ra_min_deg) & (df['RA'] <= ra_max_deg)
df_crop = df[mask_cat]
print(f"Galaxies in region: {len(df_crop)}")

# Pixelize catalogue onto the same WCS as the cropped map
coords = np.deg2rad(np.array([df_crop['DEC'].values, df_crop['RA'].values]))
density = enmap.zeros(m_crop.shape, m_crop.wcs)
pix = density.sky2pix(coords).astype(int)
# Clip to valid pixel range
valid = (pix[0] >= 0) & (pix[0] < density.shape[0]) & (pix[1] >= 0) & (pix[1] < density.shape[1])
np.add.at(density, (pix[0][valid], pix[1][valid]), 1)

p_cat = enplot.plot(density, downgrade=args.downgrade, colorbar=True,
                    grid=True, grid_width=2, ticks=args.ticks,
                    font_size=args.font_size, color="gray")
enplot.write(outbase + "_catalog", p_cat)
print(f"Wrote {outbase}_catalog.png/.pdf with {len(df_crop)} galaxy positions")
