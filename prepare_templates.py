#!/usr/bin/env python3
import sys, os, argparse, numpy as np
from pixell import enmap, curvedsky as cs
import pandas as pd
import healpy as hp
# ---------------- CLI ----------------
p = argparse.ArgumentParser(description="kSZ velocity template on CAR (pixell)")
p.add_argument("--catalogue", type=str,





               help="Path to the galaxy catalog file")
p.add_argument("--output-dir", type=str,  default=None,help='Output directory.')
p.add_argument("--lmax", type=int, default=10000)
p.add_argument("--cmb_mask", type=str,
               default="/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits",
               help="CMB apodized mask")
p.add_argument("--dec-min", type=float, default=-20.0,
               help="Declination cut in degrees (keep dec >= dec_min)")
# Add this to your argument parser section
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
    vrc=np.random.permutation(vrc)




print(f"[info] total catalog objects: {RA.size}")

# ---------------- Geometry & masks (CAR) ----------------
#gal_mask = enmap.read_fits(args.mask)   # ndmap (apodized float)
gal_mask=enmap.read_map("/scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits")

#apodized CMB mask
cmb_mask = enmap.read_map(args.cmb_mask)


cmb_mask_binary=cmb_mask.copy()
cmb_mask_binary[cmb_mask_binary!=1]=0




# 2) Hard intersection for selection + normalization
Mnorm = gal_mask*cmb_mask_binary                        
gal_mask = Mnorm.astype(float) 
# working template map on same geometry
template = enmap.zeros(gal_mask.shape, gal_mask.wcs, dtype=float)




# Pixel solid angle (sr) per CAR pixel
Omega = template.pixsizemap()

# Binary footprint for area/normalization 
Mbin = (gal_mask > 0.5).astype(bool)

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

# ---------------- Compute nbar_2D (gal/sr) on the binary footprint ----------------
Omega_survey = float(np.sum(Omega[Mbin]))
N_gal_in     = int(np.sum(counts[Mbin]))
nbar_2D      = N_gal_in / Omega_survey

print(f"[info] footprint pixels: {int(np.sum(Mbin))}")
print(f"[info] Omega_survey [sr]: {Omega_survey:.6e}")
print(f"[info] N_gal in footprint: {N_gal_in}")
print(f"[info] nbar_2D [gal/sr]: {nbar_2D:.6e}")

# ---------------- Build template: T = sum(v/c) / (nbar_2D * Omega_i) ----------------
T = enmap.zeros(template.shape, template.wcs, dtype=float)
ok = Mbin & (Omega > 0)
T[ok] = vsum[ok] / (nbar_2D * Omega[ok])


# ---------------- Build template: T = 1/ (nbar_2D * Omega_i) ----------------
gal_map= enmap.zeros(template.shape, template.wcs, dtype=float)
gal_map[ok] = 1/ (nbar_2D * Omega[ok])


print(f"[check] <T> over footprint (should be ~0): {np.mean(T[Mbin]):.3e}")

# ---------------- Save map ----------------
# 
map_path = os.path.join(args.output_dir, "../stage_template/template_car_nosrc_sub_mask.fits")
enmap.write_fits(map_path, T)
print(f"[out] wrote map: {map_path}")

map_path = os.path.join(args.output_dir, "../stage_template/template_no_vel.fits")
enmap.write_fits(map_path, gal_map)
print(f"[out] wrote map: {map_path}")

# ---------------- Produce Cls ----------------
lmax = args.lmax
W = Mbin.astype(float)              
print("[info] computing alms...")


alm = cs.map2alm(T * W, lmax=lmax, spin=0, tweak=True)

hp.write_alm(os.path.join(args.output_dir, "../stage_template/template_alm_car_nosrc_sub_mask.fits"), alm, overwrite=True)
cl  = cs.alm2cl(alm)
cls_path = os.path.join(args.output_dir, "../stage_template/template_alm_car_nosrc_sub_cls.txt")
np.savetxt(cls_path, cl)
print(f"[out] wrote Cls: {cls_path}")

# ---------------- Debugging ----------------
Omega_pix_mean = float(np.mean(Omega[Mbin]))
nbar_pix_mean  = N_gal_in / np.sum(Mbin)
print(f"[diag] <Omega_pix> [sr]: {Omega_pix_mean:.6e}")
print(f"[diag] mean galaxies per unmasked pixel: {nbar_pix_mean:.6f}")