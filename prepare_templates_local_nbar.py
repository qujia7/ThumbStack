#!/usr/bin/env python3
"""
kSZ velocity template on CAR (pixell) with LOCAL nbar normalization.

Based on prepare_templates.py but replaces global nbar_2D with local nbar mask
from HEALPix reprojection.

Outputs:
  - template_car_local_nbar.fits: velocity template with local normalization
  - template_car_local_nbar_binary_mask.fits: binary footprint mask
  - nbar_local_car.fits: reprojected local nbar mask (for weighted option)
"""
import sys, os, argparse, numpy as np
from pixell import enmap, curvedsky as cs, reproject
import pandas as pd
import healpy as hp

# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="kSZ velocity template on CAR with local nbar normalization")
p.add_argument("--catalogue", type=str,
               help="Path to the galaxy catalog file")
p.add_argument("--output-dir", type=str, default=None, help='Output directory.')
p.add_argument("--lmax", type=int, default=10000)
p.add_argument("--cmb_mask", type=str,
               default="/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits",
               help="CMB apodized mask")
p.add_argument("--local-nbar-healpix", type=str,
               default="/scratch/jiaqu/desi/catalogue/desi_survey_mask_healpix_local_nbar.fits",
               help="HEALPix local nbar mask")
p.add_argument("--dec-min", type=float, default=-20.0,
               help="Declination cut in degrees (keep dec >= dec_min)")
p.add_argument("--use_random", action='store_true',
               help="Use random catalog instead of real galaxy catalog")
args = p.parse_args()

# ---------------- Load catalog ----------------
galcat = pd.read_csv(args.catalogue)
RA   = np.asarray(galcat.RA,  dtype=float)
DEC  = np.asarray(galcat.DEC, dtype=float)
vrc  = -np.asarray(galcat.vR, dtype=float) / 3e5   # v_r / c

if args.use_random:
    np.random.seed(42)  # For reproducibility
    vrc = np.random.permutation(vrc)

print(f"[info] total catalog objects: {RA.size}")

# ---------------- Geometry & masks (CAR) ----------------
gal_mask = enmap.read_map("/scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits")
cmb_mask = enmap.read_map(args.cmb_mask)

cmb_mask_binary = cmb_mask.copy()
cmb_mask_binary[cmb_mask_binary != 1] = 0

# Hard intersection for selection + normalization
Mnorm = gal_mask * cmb_mask_binary
gal_mask = Mnorm.astype(float)

# Working template map on same geometry
template = enmap.zeros(gal_mask.shape, gal_mask.wcs, dtype=float)

# Pixel solid angle (sr) per CAR pixel
Omega = template.pixsizemap()

# ---------------- Load and reproject local nbar mask ----------------
print(f"[info] Loading HEALPix local nbar mask: {args.local_nbar_healpix}")
nbar_hp = hp.read_map(args.local_nbar_healpix)

# Get reference geometry from the existing mask
ref_shape = gal_mask.shape
ref_wcs = gal_mask.wcs

print("[info] Reprojecting HEALPix local nbar to CAR...")
nbar_local = reproject.healpix2map(nbar_hp, shape=ref_shape, wcs=ref_wcs, order=0, method='spline')

# ---------------- Sanity check after reprojection ----------------
print("\n" + "="*60)
print("SANITY CHECK: HEALPix vs CAR reprojection")
print("="*60)

# HEALPix stats (non-zero pixels)
hp_nonzero = nbar_hp[nbar_hp > 0]
print(f"[HEALPix] Non-zero pixels: {len(hp_nonzero)}")
print(f"[HEALPix] Sum:  {np.sum(hp_nonzero):.6e}")
print(f"[HEALPix] Min:  {np.min(hp_nonzero):.6e}")
print(f"[HEALPix] Max:  {np.max(hp_nonzero):.6e}")
print(f"[HEALPix] Mean: {np.mean(hp_nonzero):.6e}")

# CAR stats (non-zero pixels)
car_nonzero = nbar_local[nbar_local > 0]
print(f"\n[CAR] Non-zero pixels: {len(car_nonzero)}")
print(f"[CAR] Sum:  {np.sum(car_nonzero):.6e}")
print(f"[CAR] Min:  {np.min(car_nonzero):.6e}")
print(f"[CAR] Max:  {np.max(car_nonzero):.6e}")
print(f"[CAR] Mean: {np.mean(car_nonzero):.6e}")

# Check for negative values
n_negative = np.sum(nbar_local < 0)
print(f"\n[CHECK] Negative values in CAR: {n_negative}")
if n_negative > 0:
    print(f"[WARNING] Found {n_negative} negative values - setting to 0")
    nbar_local[nbar_local < 0] = 0

# Compare sums
ratio = np.sum(car_nonzero) / np.sum(hp_nonzero) if np.sum(hp_nonzero) > 0 else 0
print(f"\n[CHECK] CAR/HEALPix sum ratio: {ratio:.3f}")
print("[NOTE] Expected ~0.5 because CAR only covers ACT footprint")
print("="*60 + "\n")

# ---------------- Binary footprint mask ----------------
binary_mask = (nbar_local > 0).astype(float)
Mbin = binary_mask > 0.5

print(f"[info] footprint pixels: {int(np.sum(Mbin))}")
print(f"[info] Omega_survey [sr]: {float(np.sum(Omega[Mbin])):.6e}")

# ---------------- Bin galaxies to pixels ----------------
ra  = np.deg2rad(RA)
dec = np.deg2rad(DEC)
iy, ix = enmap.sky2pix(template.shape, template.wcs, [dec, ra], safe=True, corner=False)
iy = np.rint(iy).astype(int)
ix = np.rint(ix).astype(int)
inb = (iy >= 0) & (iy < template.shape[0]) & (ix >= 0) & (ix < template.shape[1])

vsum   = enmap.zeros(template.shape, template.wcs, dtype=float)
counts = enmap.zeros(template.shape, template.wcs, dtype=float)

np.add.at(vsum,   (iy[inb], ix[inb]), vrc[inb])
np.add.at(counts, (iy[inb], ix[inb]), 1.0)

N_gal_in = int(np.sum(counts[Mbin]))
print(f"[info] N_gal in footprint: {N_gal_in}")

# ---------------- Build template with LOCAL normalization ----------------
# Instead of: T[ok] = vsum[ok] / (nbar_2D * Omega[ok])
# We use:     T[ok] = vsum[ok] / nbar_local[ok]
# where nbar_local already contains nbar_2D_local * Omega (integrated surface density)

T = enmap.zeros(template.shape, template.wcs, dtype=float)
ok = Mbin & (nbar_local > 0)  # Only compute where nbar_local > 0
T[ok] = vsum[ok] / nbar_local[ok]

print(f"[check] <T> over footprint (should be ~0): {np.mean(T[Mbin]):.3e}")
print(f"[check] std(T) over footprint: {np.std(T[Mbin]):.3e}")

# ---------------- Save maps ----------------
stage_dir = os.path.join(args.output_dir, "../stage_template")

# Velocity template
map_path = os.path.join(stage_dir, "template_car_local_nbar.fits")
enmap.write_fits(map_path, T)
print(f"[out] wrote velocity template: {map_path}")

# Binary footprint mask
mask_path = os.path.join(stage_dir, "template_car_local_nbar_binary_mask.fits")
enmap.write_fits(mask_path, binary_mask)
print(f"[out] wrote binary mask: {mask_path}")

# Reprojected nbar_local (for weighted option)
nbar_path = os.path.join(stage_dir, "nbar_local_car.fits")
enmap.write_fits(nbar_path, nbar_local)
print(f"[out] wrote nbar_local: {nbar_path}")

# ---------------- Produce Cls ----------------
lmax = args.lmax
W = Mbin.astype(float)
print("[info] computing alms...")

alm = cs.map2alm(T * W, lmax=lmax, spin=0, tweak=True)

hp.write_alm(os.path.join(stage_dir, "template_alm_car_local_nbar.fits"), alm, overwrite=True)
cl  = cs.alm2cl(alm)
cls_path = os.path.join(stage_dir, "template_car_local_nbar_cls.txt")
np.savetxt(cls_path, cl)
print(f"[out] wrote Cls: {cls_path}")

# ---------------- Debugging ----------------
Omega_pix_mean = float(np.mean(Omega[Mbin]))
nbar_pix_mean  = N_gal_in / np.sum(Mbin)
print(f"[diag] <Omega_pix> [sr]: {Omega_pix_mean:.6e}")
print(f"[diag] mean galaxies per unmasked pixel: {nbar_pix_mean:.6f}")
