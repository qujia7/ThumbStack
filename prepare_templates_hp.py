#!/usr/bin/env python3
import sys, os, argparse
import numpy as np
import healpy as hp
import pandas as pd

# ---- CLI ----
parser = argparse.ArgumentParser(description="kSZ velocity template (HEALPix) — previous-good")
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



# ---- catalog ----
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
footprint=hp.read_map("/home/jiaqu/Thumbstack_DESI/output/catalogue/lrg_mask_healpix.fits")
footprint = footprint > 0.5
n_unmasked = int(footprint.sum())
print(f"[info] Footprint pixels: {n_unmasked}/{npix}")

# ---- coordinates (vectorized) ----
phi   = np.deg2rad(RA)                    # RA -> phi
theta = np.pi/2.0 - np.deg2rad(DEC)       # DEC -> co-latitude
valid_gal = np.isfinite(phi) & np.isfinite(theta) & np.isfinite(v)
phi, theta, v = phi[valid_gal], theta[valid_gal], v[valid_gal]

# ---- pixelize once ----
pix = hp.ang2pix(nside, theta, phi)       # RING

# ---- counts & velocity sums ----
counts = np.bincount(pix, minlength=npix)
vsum   = np.bincount(pix, weights=v, minlength=npix)

# ---- mean galaxies per unmasked pixel (binary footprint) ----
print(f"[info] total galaxies in footprint: {counts[footprint].sum():d}")
n_mean = counts[footprint].sum() / footprint.sum()
print(f"[info] mean galaxies / unmasked pixel = {n_mean:.6f}")

# ---- template map ----
template = np.zeros(npix, dtype=float)
template[footprint] = vsum[footprint] / max(n_mean, 1e-30)
template[~footprint] = 0.0
print(f"[check] <template> over footprint = {template[footprint].mean():.4e}")

# ---- save map ----
map_path = os.path.join(args.output, "velocity_template_healpy_vectorized.fits")
hp.write_map(map_path, template, overwrite=True)
print(f"[out] wrote map: {map_path}")

# ---- alms / power spectrum ----
templ_for_alm = template 
alm_path  = os.path.join(args.output, "velocity_template_alms.fits")
cls_path  = os.path.join(args.output, "velocity_template_cls.txt")

print("[info] computing alms...")
alms = hp.map2alm(templ_for_alm, lmax=lmax, iter=3)
hp.write_alm(alm_path, alms, overwrite=True)
cls = hp.alm2cl(alms)
np.savetxt(cls_path, cls)
print(f"[out] wrote Cls: {cls_path}")

# ---- optional prints ----
Omega_pix = 4*np.pi/npix
nbar_2D = counts[footprint].sum() / (footprint.sum()*Omega_pix)
assert np.isclose(n_mean, nbar_2D*Omega_pix)
print(f"[check] Omega_pix = {Omega_pix:.3e} sr, nbar_2D = {nbar_2D:.6e} gal/sr")