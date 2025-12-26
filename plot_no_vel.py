from pixell import enmap, enplot
import numpy as np
import os

# --- config ---
infile  = "/home/jiaqu/Thumbstack_DESI/output/z_all_new_mask/stage_template/../stage_template/template_no_vel.fits"
outbase = "/scratch/jiaqu/test_ra0_decpm20_no_vel"
dec_min, dec_max = 0, 20.0     # degrees
ra_center = 160.0              # degrees
ra_width  = 80                 # degrees total width
target_downgrade = 2  # Reduced downgrade to preserve fine-scale structure

# --- helpers ---
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

def safe_downgrade_factor(shape_hw, want):
    h, w = shape_hw
    d = max(1, int(want))
    while d > 1 and (h < d or w < d):
        d //= 2
    return d

# --- run ---
m = enmap.read_map(infile)
m_crop = crop_ra_dec_band_wcs_safe(
    m, ra_center_deg=ra_center, ra_width_deg=ra_width,
    dec_min_deg=dec_min, dec_max_deg=dec_max
)

dg = safe_downgrade_factor(m_crop.shape[-2:], target_downgrade)

# Calculate positive-only range for n(θ)/n̄ density data (non-negative by definition)
finite_data = m_crop[np.isfinite(m_crop)]
data_min = np.min(finite_data)
vmin_pct, vmax_pct = np.percentile(finite_data, [1, 99])
# Round max to nice value for cleaner colorbar display
mag = 10**np.floor(np.log10(max(1e-10, vmax_pct)))  # avoid log(0)
vmax_round = np.ceil(vmax_pct/mag) * mag
# Force min to 0 since density ratio is non-negative
vmin_round = 0

# Use the min:max range format for positive-only colorbar
# Changed: removed mask=0 so zero-density regions show at low end of colormap
# Changed: using min= parameter for better dynamic range display
p = enplot.plot(
    m_crop,
    downgrade=None if dg == 1 else dg,
    colorbar=True,
    color="gray",
    grid=True,
    grid_width=2,
    ticks=10,
    min=vmin_round,
    max=vmax_round,
)

os.makedirs(os.path.dirname(outbase), exist_ok=True)
enplot.write(outbase, p)
print(f"Wrote {outbase}.png/.pdf   crop shape={m_crop.shape}   downgrade={dg}")
print(f"Colorbar range: {vmin_round} to {vmax_round:.2f} (positive-only for n/nbar density)")
print(f"Note: Colorbar now reflects positive-only density values (data min={data_min:.2f})")