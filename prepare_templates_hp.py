#!/usr/bin/env python3
import sys, os, argparse
import numpy as np
import healpy as hp
import pandas as pd
# ---- CLI ----
parser = argparse.ArgumentParser(description="Vectorized kSZ velocity template (HEALPix).")
parser.add_argument("--catalog", type=str,
                    default="/scratch/jiaqu/desi/output/y3_final_renorm/filtered_catalog.csv")
parser.add_argument("--nside", type=int, default=8192)
parser.add_argument("--lmax", type=int, default=10000)
parser.add_argument("--output", type=str, default="/scratch/jiaqu/desi/catalogue/")
parser.add_argument("--mask", type=str, default="./output/healpix_mask_map.fits",
                    help="Binary (0/1) footprint mask or any mask to be binarized.")
parser.add_argument("--mask_apo", type=str, default="", 
                    help="Optional apodized mask used only for harmonic analysis.")
args = parser.parse_args()
os.makedirs(args.output, exist_ok=True)

# ---- project imports ----
sys.path.append('/home/jiaqu/Thumbstack_DESI/')
from cosmoprimo.fiducial import DESI
from mass_conversion import MassConversionKravtsov14
from catalog import Catalog

# ---- helpers ----
def load_mask_as_binary(path, nside_target):
    m = hp.read_map(path, dtype=float)
    # upgrade/degrade if needed
    nside_m = hp.npix2nside(m.size)
    if nside_m != nside_target:
        m = hp.ud_grade(m, nside_out=nside_target, power=-2)  # preserve 0/1 on average
    # clean to 0/1 binary support (treat non-finite as masked)
    good = np.isfinite(m) & (m > 0)
    mb = np.zeros_like(m, dtype=bool)
    mb[good] = True
    return mb  # boolean footprint

def load_mask_optional(path, nside_target):
    if not path:
        return None
    m = hp.read_map(path, dtype=float)
    nside_m = hp.npix2nside(m.size)
    if nside_m != nside_target:
        m = hp.ud_grade(m, nside_out=nside_target, power=-2)
    m[~np.isfinite(m)] = 0.0
    return m.astype(float)

# ---- cosmology & catalog ----
u = DESI()
massConv = MassConversionKravtsov14()

galcat = pd.read_csv(args.catalog)

RA  = np.asarray(galcat.RA,  dtype=float)
DEC = np.asarray(galcat.DEC, dtype=float)
v   = -np.asarray(galcat.vR, dtype=float)/3e5  # vr/c
N_gal = v.size
print(f"[info] Galaxies: {N_gal}")

# ---- HEALPix setup ----
nside = args.nside
npix  = hp.nside2npix(nside)
lmax  = args.lmax
print(f"[info] HEALPix: nside={nside} npix={npix} lmax={lmax}")

# ---- masks ----
footprint = load_mask_as_binary(args.mask, nside)          # boolean (binary) footprint
footprint_gal=hp.read_map("/scratch/jiaqu/desi/catalogue/desi_survey_mask_healpix.fits")
footprint_gal_bool = (footprint_gal > 0.5)  # Convert to boolean

footprint=footprint&footprint_gal_bool
mask_apo  = load_mask_optional(args.mask_apo, nside)       # optional for alms/cls
n_unmasked = int(footprint.sum())
print(f"[info] Footprint pixels: {n_unmasked}/{npix}")

# ---- coordinates (vectorized) ----
phi   = np.deg2rad(RA)                    # RA -> phi in [0, 2pi)
theta = np.pi/2.0 - np.deg2rad(DEC)       # DEC -> co-latitude
valid_gal = np.isfinite(phi) & np.isfinite(theta) & np.isfinite(v)
phi, theta, v = phi[valid_gal], theta[valid_gal], v[valid_gal]

# ---- pixelize once (vectorized) ----
pix = hp.ang2pix(nside, theta, phi)       # RING ordering by default

# ---- counts & velocity sums per pixel (vectorized) ----
counts = np.bincount(pix, minlength=npix)
vsum   = np.bincount(pix, weights=v, minlength=npix)  # sum of vr/c per pixel

# ---- mean galaxies per unmasked pixel (binary footprint) ----
n_mean = counts[footprint].sum() / footprint.sum()
print(f"[info] mean galaxies / unmasked pixel = {n_mean:.6f}")

# ---- template map ----
template = np.zeros(npix, dtype=float)
template[footprint] = vsum[footprint] / max(n_mean, 1e-30)   # avoid div-by-0
template[~footprint] = 0.0

# Sanity check: mean in footprint should be ~0 (⟨vr⟩≈0)
print(f"[check] <template> over footprint = {template[footprint].mean():.4e}")

# ---- save map ----
map_path = os.path.join(args.output, "velocity_template_healpy_vectorized.fits")
hp.write_map(map_path, template, overwrite=True)
print(f"[out] wrote map: {map_path}")

# ---- alms / power spectrum ----
# For spectra, it helps to taper edges: use apodized mask if provided.
templ_for_alm = template if mask_apo is None else (template * mask_apo)
alm_path  = os.path.join(args.output, "velocity_template_alms.fits")
cls_path  = os.path.join(args.output, "velocity_template_cls.txt")

print("[info] computing alms...")
alms = hp.map2alm(templ_for_alm, lmax=lmax, iter=3)
hp.write_alm(alm_path, alms, overwrite=True)
print(f"[out] wrote alms: {alm_path}")

cls = hp.alm2cl(alms)
np.savetxt(cls_path, cls)
print(f"[out] wrote Cls: {cls_path}")

# ---- optional prints ----
Omega_pix = 4*np.pi/npix
nbar_2D = counts[footprint].sum() / (footprint.sum()*Omega_pix)
assert np.isclose(n_mean, nbar_2D*Omega_pix)
print(f"[check] Omega_pix = {Omega_pix:.3e} sr, nbar_2D = {nbar_2D:.6e} gal/sr")