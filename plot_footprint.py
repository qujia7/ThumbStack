"""
plot_footprint.py  –  Fig. 1: DESI LRG DR2 + ACT DR6 survey footprint.

Uses pre-built HEALPix survey masks (not galaxy positions) for a clean
footprint, following the hp.projview pattern from C3CMB_plots/fig_footprint.py.
"""

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Mask paths
# ---------------------------------------------------------------------------
DESI_MASK_PATH = '/scratch/jiaqu/desi/catalogue/desi_survey_mask_healpix.fits'
ACT_MASK_PATH  = '/scratch/jiaqu/actxdesi33/masks/sz_mask_hp.fits'

# ---------------------------------------------------------------------------
# Settings  (match C3CMB fig_footprint.py)
# ---------------------------------------------------------------------------
NSIDE         = 64
SMOOTH_SIGMA  = 1.5

C_ACT  = '#F0E68C'             # yellow (top layer)
C_DESI = '#6495ED'             # cornflower blue (NGC blue)

ALPHA_FILL    = 0.55
ALPHA_CONTOUR = 0.85


# ---------------------------------------------------------------------------
# Helpers  (identical pattern to C3CMB fig_footprint.py)
# ---------------------------------------------------------------------------
def degrade(m, nside=NSIDE):
    return (hp.ud_grade(m.astype(float), nside) > 0).astype(float)


def project(binary_map):
    """Return (lon_1d, lat_1d, grid_2d) via hp.projview without rendering."""
    lon, lat, grid = hp.projview(
        binary_map, projection_type='mollweide', return_only_data=True)
    plt.close('all')
    return lon, lat, grid


def smooth(grid):
    g = grid.copy()
    g[np.isnan(g)] = 0
    return gaussian_filter(g, sigma=SMOOTH_SIGMA)


# ---------------------------------------------------------------------------
# Load masks
# ---------------------------------------------------------------------------
desi_mask = hp.read_map(DESI_MASK_PATH)
act_mask  = hp.read_map(ACT_MASK_PATH)
act_mask[act_mask != 0] = 1

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
lon, lat, grid_desi = project(degrade(desi_mask))
_,   _,   grid_act  = project(degrade(act_mask))

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.5),
                       subplot_kw={'projection': 'mollweide'})
ax.set_rasterization_zorder(0.5)

ax.contourf(lon, lat, smooth(grid_desi), levels=[0.5, 1.5], colors=[C_DESI], alpha=ALPHA_FILL,    zorder=0)
ax.contour( lon, lat, smooth(grid_desi), levels=[0.5],      colors=[C_DESI], linewidths=1.0, alpha=ALPHA_CONTOUR, zorder=1)
ax.contourf(lon, lat, smooth(grid_act),  levels=[0.5, 1.5], colors=[C_ACT],  alpha=ALPHA_FILL,    zorder=2)
ax.contour( lon, lat, smooth(grid_act),  levels=[0.5],      colors=[C_ACT],  linewidths=1.0, alpha=ALPHA_CONTOUR, zorder=3)

ax.grid(True, alpha=0.3)
ax.set_xticks([])
ax.set_yticks([])

# RA labels along equator (x: linear in lon; y: just below equator)
for x_ax, lbl in [(0.187, '-120°'), (0.343, '-60°'), (0.500, '0°'),
                   (0.657, '60°'),  (0.813, '120°')]:
    ax.text(x_ax, 0.47, lbl, ha='center', va='top', fontsize=8,
            transform=ax.transAxes, zorder=10)

# Dec labels just right of central meridian
for y_ax, lbl in [(0.13, '-60°'), (0.25, '-30°'), (0.75, '30°'), (0.87, '60°')]:
    ax.text(0.52, y_ax, lbl, ha='left', va='center', fontsize=8,
            transform=ax.transAxes, zorder=10)
ax.set_xlabel('RA [deg]', fontsize=9)
ax.set_ylabel('Dec [deg]', fontsize=9)
ax.legend(handles=[
    Patch(color=C_ACT,  alpha=ALPHA_FILL, label='ACT DR6'),
    Patch(color=C_DESI, alpha=ALPHA_FILL, label='DESI NGC'),
    Patch(color=C_DESI, alpha=ALPHA_FILL, label='DESI SGC'),
], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncols=3,
   fontsize=8, frameon=True, borderaxespad=0)

plt.tight_layout()

out = '/scratch/jiaqu/footprint_map_custom_alpha.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
