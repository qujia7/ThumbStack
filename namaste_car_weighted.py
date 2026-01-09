#!/usr/bin/env python3
"""
Harmonic-space cross-correlation with NaMaster using LOCAL nbar template.

Uses WEIGHTED mask option:
  - Template: template_car_local_nbar.fits
  - Weight: nbar_local_car.fits (local nbar as continuous weight, NOT binarized)

This affects the mode-coupling matrix: galaxies in high-density regions
contribute more to the final power spectrum estimate.

Based on namaste_car.py but uses local nbar as the galaxy mask weight.

Output: cl_master_weighted.txt
"""
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS
from pixell import enmap, reproject, curvedsky as cs
import numpy as np
import argparse

p = argparse.ArgumentParser(description="NaMaster cross-spectra with local nbar template (weighted mask)")

p.add_argument("--output-dir", type=str, default=None, help='Output directory.')
p.add_argument("--lmax", type=int, default=10000)
p.add_argument("--cmb-mask", type=str,
               default="/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits",
               help="CMB apodized mask")
p.add_argument("--gal-mask", type=str,
               default="/scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits",
               help="galactic mask from randoms")
p.add_argument("--dec-min", type=float, default=-20.0,
               help="Declination cut in degrees (keep dec >= dec_min)")
p.add_argument("--use_random", action='store_true',
               help="Use random catalog instead of real galaxy catalog")
args = p.parse_args()

print("="*60)
print("NaMaster cross-spectra: LOCAL NBAR + WEIGHTED MASK")
print("="*60)

print("loading maps...")

cmb_masked = enmap.read_map("./output/catalogue/cmb_map_car_nosrc_cluster_sub_mask.fits")
cmb_mask = enmap.read_map(args.cmb_mask)

# Make CMB mask binary
cmb_mask_binary = cmb_mask.copy()
cmb_mask_binary[cmb_mask_binary != 1] = 0

# Load LOCAL NBAR template and nbar_local weight mask
template = enmap.read_map(f'{args.output_dir}/../stage_template/template_car_local_nbar.fits')
nbar_local = enmap.read_map(f'{args.output_dir}/../stage_template/nbar_local_car.fits')

# Use nbar_local as the weight (NOT binarized)
# This affects the mode-coupling matrix
combined = cmb_mask_binary * nbar_local

# Normalize the weight to have max=1 for numerical stability
weight_max = np.max(combined)
if weight_max > 0:
    combined_normalized = combined / weight_max
else:
    combined_normalized = combined.copy()

wcs = combined.wcs

print(f"[info] Template: template_car_local_nbar.fits")
print(f"[info] Mask: nbar_local_car.fits (continuous weight)")
print(f"[info] Weight range: [{np.min(combined):.3e}, {np.max(combined):.3e}]")
print(f"[info] Normalized weight range: [{np.min(combined_normalized):.3e}, {np.max(combined_normalized):.3e}]")
print("Finished loading.")

lmax = args.lmax

# Create spin-0 fields with WCS for rectangular pixels
# For CMB field, use the original apodized mask
f_cmb = nmt.NmtField(cmb_mask, [cmb_masked], wcs=wcs, n_iter=0, masked_on_input=True, lmax=lmax)
# For template field, use nbar_local as the weight
f_template = nmt.NmtField(combined_normalized, [template], wcs=wcs, n_iter=0, masked_on_input=True, lmax=lmax)

# Binning: wide-fine-wide scheme
edges = list(np.arange(800, 2000, 400))
edges += list(np.arange(2000, 6500, 400))
edges += list(np.arange(6500, lmax + 1, 800))

if edges[-1] <= lmax:
    edges.append(lmax + 1)

edges = np.unique(edges).astype(int)

b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

print("Number of bins:", b.get_n_bands())
print("Bin centers:", b.get_effective_ells())

# Build binning matrix
nbins = len(edges) - 1
M = np.zeros((nbins, lmax + 1))

for i in range(nbins):
    lo, hi = edges[i], edges[i + 1]
    M[i, lo:hi] = 1.0 / (hi - lo)

# Compute mode-coupling matrix (this will use the weighted mask)
wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f_cmb, f_template, b)

# Save the workspace
wsp_file = f'{args.output_dir}/../stage_spectra/workspace_cmb_template_weighted.fits'
wsp.write_to(wsp_file)
print(f"Workspace saved to: {wsp_file}")

# Compute the pseudo cross-power spectrum
print("Computing pseudo-Cl...")
cl_pseudo = nmt.compute_coupled_cell(f_cmb, f_template)

# Apply the inverse mode-coupling matrix
print("Deconvolving window function...")
cl_deconvolved = wsp.decouple_cell(cl_pseudo)[0]

ell_eff = b.get_effective_ells()

# Save results
output_file = f'{args.output_dir}/../stage_spectra/cl_master_weighted.txt'
np.savetxt(output_file,
           np.column_stack([ell_eff, cl_deconvolved]),
           header="ell Cl_CMB_x_template (local nbar, weighted mask)")
print(f'Spectra saved to: {output_file}')

# Estimate error using alms
SNR = lambda auto, error: round(np.sqrt(np.sum(auto**2/error**2)), 0)

f_sky = 0.12032733727660157

def process_alms(a_alms, b_alms, f_sky, binning_matrix, pixwin=None, edges=None):
    acl = cs.alm2cl(a_alms) / 0.30340039899762616
    bcl = cs.alm2cl(b_alms) / f_sky
    kcl = cs.alm2cl(a_alms, b_alms) / f_sky

    if pixwin is not None:
        acl /= pixwin**2
        bcl /= pixwin**2
        kcl /= pixwin**2
    kclb = binning_matrix @ kcl[:10001]
    cl_e = binning_matrix @ (np.nan_to_num(acl * bcl + kcl**2))[:10001]
    ells = binning_matrix @ np.arange(10001)
    error = np.sqrt(1 / (2 * ells + 1) / np.diff(edges) / f_sky) * np.sqrt(cl_e)
    return ells, kclb, error

bin_matrix = lambda edges, lmax: np.array([
    [1.0/(min(edges[i+1], lmax+1) - max(edges[i], 0))
     if max(edges[i], 0) <= l < min(edges[i+1], lmax+1) else 0.0
     for l in range(lmax+1)]
    for i in range(len(edges)-1)
])

B = M

import healpy as hp

p_alms = hp.read_alm(f"{args.output_dir}/../stage_template/template_alm_car_local_nbar.fits")
cmb_alms = hp.read_alm("/scratch/jiaqu/desi/catalogue/cmb_alms_car_nosrc_sub_mask.fits")

ell, cl, error = process_alms(cmb_alms, p_alms, f_sky, B, edges=edges)

np.savetxt(f'{args.output_dir}/../stage_spectra/error_weighted.txt', error)
print(f"Error saved to: {args.output_dir}/../stage_spectra/error_weighted.txt")

print("\n" + "="*60)
print("DONE: cl_master_weighted.txt")
print("="*60)
