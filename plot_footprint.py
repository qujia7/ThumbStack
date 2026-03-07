"""
plot_footprint.py  –  Fig. 1: DESI LRG DR2 + ACT DR6 survey footprint.

Replaces the old create_full_footprint_map_custom_alpha that assigned
separate colors to NGC-only / SGC-only / ACT+NGC / ACT+SGC etc.

New design
----------
- DESI footprint (NGC + SGC combined) → single dark-blue fill
- ACT DR6 footprint                   → red fill
- Overlap rendered naturally via alpha transparency on contourf layers
- Contour outlines drawn on top for sharpness
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def create_full_footprint_map_custom_alpha(
        ra_ngc, dec_ngc, ra_sgc, dec_sgc,
        act_mask=None, nside=256,
        alpha_act=0.35, alpha_desi=0.45):
    """
    Show the DESI footprint (NGC + SGC combined) in blue and ACT DR6 in red.
    The overlap defines the analysis region and is rendered naturally through
    alpha transparency.

    Parameters
    ----------
    ra_ngc, dec_ngc : array-like  – RA/Dec of NGC galaxies (degrees)
    ra_sgc, dec_sgc : array-like  – RA/Dec of SGC galaxies (degrees)
    act_mask        : array-like or None – HEALPix ACT mask (any nside)
    nside           : int   – HEALPix nside for projection (default 256)
    alpha_act       : float – fill opacity for ACT layer
    alpha_desi      : float – fill opacity for DESI layer
    """
    SMOOTH = 1.5          # Gaussian σ in pixels – suppresses HEALPix pixelation

    C_ACT  = (0.84, 0.15, 0.16)   # red
    C_DESI = (0.07, 0.27, 0.55)   # dark blue

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

    desi_map = np.clip(radec_to_mask(ra_ngc, dec_ngc)
                       + radec_to_mask(ra_sgc, dec_sgc), 0, 1)

    if act_mask is not None:
        act_hp = hp.ud_grade(np.asarray(act_mask, dtype=float), nside)
        act_hp = (act_hp > 0).astype(float)
    else:
        act_hp = np.zeros(npix, dtype=float)

    # ------------------------------------------------------------------
    # Project HEALPix maps onto a regular lon/lat grid (Mollweide coords)
    # ------------------------------------------------------------------
    ny, nx = 400, 800
    lon_1d = np.linspace(-np.pi, np.pi, nx)
    lat_1d = np.linspace(-np.pi / 2, np.pi / 2, ny)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    def hp_to_grid(hpmap):
        theta = np.pi / 2 - lat2d
        # Flip sign: healpy phi=0 is RA=0 at centre, RA increases to the LEFT
        # matplotlib lon=0 is centre, lon increases to the RIGHT → phi = -lon
        phi   = (-lon2d) % (2 * np.pi)
        pix   = hp.ang2pix(nside, theta.ravel(), phi.ravel())
        g = hpmap[pix].reshape(ny, nx).astype(float)
        # mask pixels outside the Mollweide ellipse
        in_ellipse = (lon2d / np.pi) ** 2 + (lat2d / (np.pi / 2)) ** 2 <= 1.0
        g[~in_ellipse] = np.nan
        return g

    grid_desi = hp_to_grid(desi_map)
    grid_act  = hp_to_grid(act_hp)

    def smooth(g):
        tmp = g.copy()
        tmp[np.isnan(tmp)] = 0.0
        return gaussian_filter(tmp, sigma=SMOOTH)

    def make_facecolors(grid, color, alpha):
        """Per-cell RGBA array for pcolormesh (shape (ny-1)*(nx-1), 4)."""
        s = smooth(grid)
        filled = (s > 0.5) & ~np.isnan(grid)
        rgba = np.zeros((*grid.shape, 4))
        rgba[filled, 0] = color[0]
        rgba[filled, 1] = color[1]
        rgba[filled, 2] = color[2]
        rgba[filled, 3] = alpha
        # use upper-left corner of each cell (shading='flat')
        return rgba[:-1, :-1].reshape(-1, 4)

    def smooth_for_contour(g):
        """Smooth and break continuity at the ±π boundary to kill wrap lines."""
        s = smooth(g)
        s[:, 0] = np.nan
        s[:, -1] = np.nan
        return s

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5),
                           subplot_kw={'projection': 'mollweide'})
    ax.set_rasterization_zorder(0.5)

    # filled regions via pcolormesh (no wrap-around artefacts)
    for grid, color, alpha in [(grid_act, C_ACT, alpha_act),
                                (grid_desi, C_DESI, alpha_desi)]:
        pc = ax.pcolormesh(lon2d, lat2d,
                           np.zeros((ny - 1, nx - 1)),
                           shading='flat', zorder=0)
        pc.set_facecolors(make_facecolors(grid, color, alpha))
        pc.set_edgecolors('none')

    # outlines — boundary columns set to NaN to prevent cross-map lines
    for grid, color in [(grid_act, C_ACT), (grid_desi, C_DESI)]:
        try:
            ax.contour(lon2d, lat2d, smooth_for_contour(grid),
                       levels=[0.5], colors=[color],
                       linewidths=1.0, alpha=0.8, zorder=1)
        except Exception:
            pass

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("RA [deg]", fontsize=12)
    ax.set_ylabel("Dec [deg]", fontsize=12)
    ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
    ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))

    ax.legend(
        handles=[
            Patch(facecolor=C_ACT,  alpha=alpha_act,  edgecolor=C_ACT,
                  label='ACT DR6'),
            Patch(facecolor=C_DESI, alpha=alpha_desi, edgecolor=C_DESI,
                  label='DESI LRG DR2'),
        ],
        loc='lower center', bbox_to_anchor=(0.5, 1.02),
        ncols=2, fontsize=10, frameon=True,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point – mirrors what the notebook cell does
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
        nside=256,
        alpha_act=0.35,
        alpha_desi=0.45,
    )

    out = 'footprint_map_custom_alpha.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
    plt.show()
