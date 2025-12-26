from pixell import enmap, enplot
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- config ---
infile  = "/scratch/jiaqu/desi/catalogue/zall_template_car_nosrc_sub_mask.fits"
outbase = "/scratch/jiaqu/test_ra0_decpm20"
dec_min, dec_max = 0, 12.5     # degrees
ra_center = 140.0              # degrees
ra_width  = 40                 # degrees total width (RA 120-160)
target_downgrade = 4

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

# --- run ---
m = enmap.read_map(infile)
m_crop = crop_ra_dec_band_wcs_safe(
    m, ra_center_deg=ra_center, ra_width_deg=ra_width,
    dec_min_deg=dec_min, dec_max_deg=dec_max
)

# Round colorbar range to nice value
vmax_round = 0.002

# Use enplot with nice settings
p = enplot.plot(
    m_crop,
    downgrade=target_downgrade,
    colorbar=True,
    color="planck",
    grid=True,
    grid_width=2,
    ticks=10,
    mask=0,
    range=f"{vmax_round}",
)

os.makedirs(os.path.dirname(outbase), exist_ok=True)

# Write temporary enplot output
temp_png = outbase + "_temp.png"
enplot.write(temp_png.replace(".png", ""), p)

# Load enplot image and add axis labels using matplotlib
img = Image.open(temp_png)
img_array = np.array(img)

# Create matplotlib figure matching plot_no_vel.py style
fig, ax = plt.subplots(figsize=(16, 6))

# Display the enplot image
# The enplot image has RA decreasing left-to-right (160 -> 120)
ra_min = ra_center - ra_width / 2  # 120
ra_max = ra_center + ra_width / 2  # 160
ax.imshow(img_array, extent=[ra_max, ra_min, dec_min, dec_max], aspect='auto')

# Add axis labels matching plot_no_vel.py
ax.set_xlabel('RA [deg]')
ax.set_ylabel('Dec [deg]')
ax.set_xlim(ra_max, ra_min)  # RA increases to the left
ax.set_ylim(dec_min, dec_max)

# Save with matplotlib
plt.savefig(f"{outbase}.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{outbase}.pdf", bbox_inches='tight')
plt.close()

# Clean up temporary file
os.remove(temp_png)

print(f"Wrote {outbase}.png/.pdf   crop shape={m_crop.shape}")
print(f"Color range: ±{vmax_round:.2e}")
