from pixell import enmap, enplot
import numpy as np
import os
import argparse

# --- argument parsing ---
parser = argparse.ArgumentParser(description="Plot enmap with customizable grid ticks")
parser.add_argument("-t", "--ticks", type=str, default="1",
                    help="The grid spacing in degrees. Either a single number to be used for both axes, or ty,tx.")
parser.add_argument("--tick-unit", "--tu", type=str, default=None,
                    help="Units for tick axis. Can be the unit size in degrees, or the word 'degree', 'arcmin' or 'arcsec' or the shorter 'd','m','s'.")
parser.add_argument("--font-size", type=int, default=20,
                    help="Font size for tick labels in pixels (default: 20)")
parser.add_argument("-d", "--downgrade", type=int, default=4,
                    help="Downgrade factor for resolution (default: 4, use 1 for full resolution)")
args = parser.parse_args()

# --- config ---
infile  = "/scratch/jiaqu/desi/catalogue/zall_template_car_nosrc_sub_mask.fits"
outbase = "/scratch/jiaqu/test_ra0_decpm20"
dec_min, dec_max = 0, 12.5     # degrees
ra_center = 140.0              # degrees
ra_width  = 40                 # degrees total width (RA 120-160)

# --- helpers ---
def parse_tick_unit(tick_unit):
    """Parse tick_unit argument into a value usable by enplot."""
    if tick_unit is None:
        return 1.0
    try:
        return float(tick_unit)
    except ValueError:
        return tick_unit

def crop_ra_dec_band_wcs_safe(m, ra_center_deg, ra_width_deg, dec_min_deg, dec_max_deg):
    """Crop a CAR enmap to RA in [ra_center - ra_width/2, ra_center + ra_width/2] (with wrap)
       and Dec in [dec_min, dec_max], using the actual WCS (no assumptions)."""
    pos = m.posmap()
    dec_map = pos[0]
    ra_map  = pos[1]

    dmin = np.deg2rad(dec_min_deg)
    dmax = np.deg2rad(dec_max_deg)
    y_mask = (dec_map[:, 0] >= min(dmin, dmax)) & (dec_map[:, 0] <= max(dmin, dmax))
    if not np.any(y_mask):
        raise ValueError("Dec band produced no rows; check dec_min/dec_max.")
    y0, y1 = np.where(y_mask)[0][[0, -1]]
    y1 += 1

    rac   = np.deg2rad(ra_center_deg)
    halfw = np.deg2rad(ra_width_deg / 2.0)
    y_mid = (y0 + y1) // 2
    ra_row = ra_map[y_mid]
    ra_unwrapped = np.unwrap(ra_row - rac) + rac

    ra_min_t = rac - halfw
    ra_max_t = rac + halfw
    x_mask = (ra_unwrapped >= ra_min_t) & (ra_unwrapped <= ra_max_t)
    if not np.any(x_mask):
        raise ValueError("RA window produced no columns; check ra_center/ra_width.")
    x0, x1 = np.where(x_mask)[0][[0, -1]]
    x1 += 1

    return m[y0:y1, x0:x1]

# --- run ---
m = enmap.read_map(infile)
m_crop = crop_ra_dec_band_wcs_safe(
    m, ra_center_deg=ra_center, ra_width_deg=ra_width,
    dec_min_deg=dec_min, dec_max_deg=dec_max
)

# Colorbar range (use nice round values)
cbar_min = -0.002
cbar_max = 0.0025

# Parse ticks
ticks_list = [float(x) for x in args.ticks.split(",")]
if len(ticks_list) == 1:
    tick_dec, tick_ra = ticks_list[0], ticks_list[0]
else:
    tick_dec, tick_ra = ticks_list[0], ticks_list[1]

os.makedirs(os.path.dirname(outbase), exist_ok=True)

# Use enplot with nice settings
plot_kwargs = dict(
    downgrade=args.downgrade,
    colorbar=True,
    color="planck",
    grid=True,
    grid_width=2,
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
