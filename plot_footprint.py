"""
plot_footprint.py  –  Fig. 1: DESI LRG DR2 + ACT DR6 survey footprint.

Closely follows the approach in C3CMB_plots/fig_footprint.py:
- hp.projview(..., return_only_data=True) → (lon_1d, lat_1d, grid)
- contourf for filled regions, contour for outlines
- DESI footprint (NGC + SGC combined) in a single dark blue
- ACT DR6 in red; overlap rendered naturally via alpha compositing
"""

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
NSIDE         = 64     # lower → fewer holes; raise for sharper boundaries
SMOOTH_SIGMA  = 1.5    # Gaussian smoothing (pixels) to suppress pixelation

C_ACT  = (0.84, 0.15, 0.16)   # red
C_DESI = (0.07, 0.27, 0.55)   # dark blue

ALPHA_FILL    = 0.35
ALPHA_CONTOUR = 0.70


# ---------------------------------------------------------------------------
# Helpers  (mirror C3CMB fig_footprint.py helpers)
# ---------------------------------------------------------------------------
def degrade(m, nside=NSIDE):
    """Downgrade map to lower nside and binarise."""
    return (hp.ud_grade(m.astype(float), nside) > 0).astype(float)


def project(binary_map):
    """Return (lon_1d, lat_1d, grid_2d) via hp.projview without rendering."""
    lon, lat, grid = hp.projview(
        binary_map, projection_type="mollweide", return_only_data=True)
    plt.close('all')
    return lon, lat, grid


def filled(grid):
    """Replace NaN (out-of-projection pixels) with 0."""
    g = grid.copy()
    g[np.isnan(g)] = 0
    return g


def smooth(grid):
    return gaussian_filter(filled(grid), sigma=SMOOTH_SIGMA)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def create_full_footprint_map_custom_alpha(
        ra_ngc, dec_ngc, ra_sgc, dec_sgc,
        act_mask=None, nside=NSIDE,
        alpha_act=ALPHA_FILL, alpha_desi=ALPHA_FILL):
    """
    Show the DESI footprint (NGC + SGC combined) in blue and ACT DR6 in red.
    The overlap defines the analysis region and is rendered naturally through
    alpha transparency.
    """
    npix = hp.nside2npix(nside)

    # ------------------------------------------------------------------
    # Build binary HEALPix masks
    # ------------------------------------------------------------------
    def radec_to_mask(ra, dec):
        m = np.zeros(npix, dtype=float)
        pix = hp.ang2pix(nside,
                         np.radians(90.0 - np.asarray(dec)),
                         np.radians(np.asarray(ra)))
        m[pix] = 1.0
        return m

    desi_hp = np.clip(radec_to_mask(ra_ngc, dec_ngc)
                      + radec_to_mask(ra_sgc, dec_sgc), 0, 1)

    if act_mask is not None:
        act_hp = degrade(act_mask, nside)
    else:
        act_hp = np.zeros(npix, dtype=float)

    # ------------------------------------------------------------------
    # Project with hp.projview  (handles RA convention correctly)
    # ------------------------------------------------------------------
    lon, lat, grid_act  = project(degrade(desi_hp * 0 + act_hp,  nside))
    _,   _,   grid_desi = project(degrade(desi_hp, nside))

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.5),
                           subplot_kw={'projection': 'mollweide'})
    ax.set_rasterization_zorder(0.5)

    # filled regions
    ax.contourf(lon, lat, smooth(grid_act),
                levels=[0.5, 1.5], colors=[C_ACT],  alpha=alpha_act,  zorder=0)
    ax.contourf(lon, lat, smooth(grid_desi),
                levels=[0.5, 1.5], colors=[C_DESI], alpha=alpha_desi, zorder=0)

    # outlines
    ax.contour(lon, lat, smooth(grid_act),
               levels=[0.5], colors=[C_ACT],  linewidths=1.0,
               alpha=ALPHA_CONTOUR, zorder=1)
    ax.contour(lon, lat, smooth(grid_desi),
               levels=[0.5], colors=[C_DESI], linewidths=1.0,
               alpha=ALPHA_CONTOUR, zorder=1)

    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("RA [deg]", fontsize=9)
    ax.set_ylabel("Dec [deg]", fontsize=9)

    ax.legend(
        handles=[
            Patch(color=C_ACT,  alpha=alpha_act,  label='ACT DR6'),
            Patch(color=C_DESI, alpha=alpha_desi, label='DESI LRG DR2'),
        ],
        loc='lower center', bbox_to_anchor=(0.5, 1.02),
        ncols=2, fontsize=8, frameon=True, borderaxespad=0,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Columns: RA  Dec  Z  velocity_LOS
    _NGC = np.loadtxt('/scratch/jiaqu/desi/catalogue/NGC_catalog_Y3.txt')
    _SGC = np.loadtxt('/scratch/jiaqu/desi/catalogue/SGC_catalog_Y3.txt')
    ra_ngc, dec_ngc = _NGC[:, 0], _NGC[:, 1]
    ra_sgc, dec_sgc = _SGC[:, 0], _SGC[:, 1]

    act_mask = hp.read_map('/scratch/jiaqu/actxdesi33/masks/sz_mask_hp.fits')
    act_mask[act_mask != 0] = 1

    fig = create_full_footprint_map_custom_alpha(
        ra_ngc, dec_ngc,
        ra_sgc, dec_sgc,
        act_mask=act_mask,
    )

    out = 'footprint_map_custom_alpha.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
