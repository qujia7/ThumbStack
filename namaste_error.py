#!/usr/bin/env python3
"""
namaste_error.py - NaMaster Gaussian Covariance Estimation for CMB x Template Cross-Spectra

PURPOSE:
    Computes the full Gaussian covariance matrix for the CMB × kSZ template cross-spectrum
    using NaMaster's analytic covariance formalism. This accounts for mode-coupling due to
    the survey mask and provides proper error bars including off-diagonal correlations.

USAGE:
    python namaste_error.py --output-dir /path/to/output [OPTIONS]

INPUTS REQUIRED:
    =============================================================================
    FROM PREVIOUS PIPELINE STAGES:
    =============================================================================
    1. Template map (from stage_template):
       - {output_dir}/../stage_template/template_car_nosrc_sub_mask.fits

    2. Template alms (from stage_template, for each redshift bin):
       - output/{z_bin}/stage_template/template_alm_car_nosrc_sub_mask.fits

    3. NaMaster workspace (from stage_spectra):
       - {output_dir}/../stage_spectra/workspace_cmb_template_no_scr_mask.fits
       - Contains the mode-coupling matrix M_ll' for the CMB x template fields

    =============================================================================
    STATIC / PRECOMPUTED FILES:
    =============================================================================
    4. Masked CMB map:
       - ./output/catalogue/cmb_map_car_nosrc_cluster_sub_mask.fits
       - ACT DR6 CMB map with point sources and clusters subtracted, masked

    5. CMB alms:
       - /scratch/jiaqu/desi/catalogue/cmb_alms_car_nosrc_sub_mask.fits
       - Spherical harmonic coefficients of the CMB map for C_aa computation

    6. Covariance workspace (NaMaster):
       - /scratch/jiaqu/desi/catalogue/cov_wp_test.fits
       - Precomputed coupling coefficients for covariance calculation
       - This is EXPENSIVE to compute; we read from disk

    7. Theory cross-spectrum (for off-diagonal terms):
       - /home/jiaqu/Thumbstack_DESI/binned_theory.txt
       - Binned C_l^{Tg} theory predictions for each redshift bin
       - Shape: (4, Nbins) for z1, z2, z3, z4

    =============================================================================
    MASKS (passed via command-line):
    =============================================================================
    8. CMB apodized mask (--cmb-mask):
       - Default: /home/jiaqu/.../wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits
       - Smooth apodization to reduce mode-coupling

    9. Galaxy survey mask (--gal-mask):
       - Default: /scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits
       - Binary footprint of the DESI survey

    =============================================================================
    THEORY BLOCK FOR OFF-DIAGONAL COVARIANCE:
    =============================================================================
    The Gaussian covariance of C_l^{ab} x C_l'^{ab} requires 4 input spectra:

        Cov(C_l^{ab}, C_l'^{ab}) ~ C_l^{aa} C_l'^{bb} + C_l^{ab} C_l'^{ab}

    Where:
    - C_aa = CMB auto-spectrum (computed from cmb_alms / f_sky^CMB)
    - C_bb = Template auto-spectrum (computed from template_alms / f_sky^gal)
    - C_ab = THEORY cross-spectrum (loaded from binned_theory.txt)

    WHY THEORY FOR C_ab?
    - The measured C_ab is noisy and would bias the covariance low
    - Theory provides the expected signal contribution to the variance
    - Off-diagonal terms capture correlations between bandpowers due to:
      (a) Mask-induced mode coupling
      (b) Sample variance from the cross-correlation signal itself

    The covariance workspace (cw) contains the coupling coefficients that
    translate the input full-sky spectra to the masked pseudo-Cl covariance.

OUTPUT:
    - output/{z_bin}/stage_spectra/namaste_covariance_{z_bin}.npy
    - Shape: (Nbins, Nbins) covariance matrix for the cross-spectrum bandpowers

DEPENDENCIES:
    - pymaster (NaMaster)
    - pixell
    - healpy
    - numpy
    - astropy

AUTHOR: Frank Qu 
DATE: 2024
"""
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS
from pixell import enmap, reproject, curvedsky as cs
import numpy as np
import argparse


p = argparse.ArgumentParser(
    description="Compute Gaussian covariance matrix for CMB x template cross-spectrum",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
EXAMPLE:
    python namaste_error.py --output-dir output/z1_new_mask/stage_error --lmax 10000

NOTES:
    - Requires precomputed covariance workspace (expensive, ~hours to compute)
    - Uses theory C_ab for off-diagonal terms to avoid noise bias
    - f_sky corrections: CMB=0.303, Galaxy=0.120 (hardcoded)
    """
)

# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================
p.add_argument("--output-dir", type=str, default=None,
               help='Output directory for covariance matrices')
p.add_argument("--lmax", type=int, default=10000,
               help='Maximum multipole for analysis (default: 10000)')
p.add_argument("--cmb-mask", type=str,
               default="/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits",
               help="Apodized CMB mask (smooth edges to reduce mode-coupling)")
p.add_argument("--gal-mask", type=str,
               default="/scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits",
               help="Galaxy survey footprint mask from randoms")
p.add_argument("--dec-min", type=float, default=-20.0,
               help="Declination cut in degrees (keep dec >= dec_min) [currently unused]")
p.add_argument("--use_random", action='store_true',
               help="Use random catalog instead of real galaxy catalog (for null tests) [currently unused]")

# Additional paths that could be made configurable
# Currently hardcoded in the script - see docstring for full list
args = p.parse_args()





# =============================================================================
# LOAD MAPS AND MASKS
# =============================================================================
print("Loading maps...")

# INPUT 1: Masked CMB map (point sources and clusters subtracted)
cmb_masked = enmap.read_map("./output/catalogue/cmb_map_car_nosrc_cluster_sub_mask.fits")

# INPUT 2: Apodized CMB mask (smooth taper at edges)
cmb_mask = enmap.read_map(args.cmb_mask)

# INPUT 3: Galaxy survey footprint mask
gal_mask = enmap.read_map(args.gal_mask)

# Create joint mask: binary CMB mask × galaxy mask
# This ensures we only use regions with both CMB and galaxy coverage
cmb_mask_binary = cmb_mask.copy()
cmb_mask_binary[cmb_mask_binary != 1] = 0
combined = cmb_mask_binary * gal_mask

wcs = combined.wcs

# INPUT 4: Template map from stage_template (just for field definition, not used in loop)
template = enmap.read_map(f'/home/jiaqu/Thumbstack_DESI/output/z_all_new_mask/stage_template/template_car_nosrc_sub_mask.fits')

print("Finished loading maps.")

# =============================================================================
# CREATE NAMASTER FIELDS
# =============================================================================
lmax = args.lmax  # Maximum multipole

# Create spin-0 NaMaster fields for CMB and template
# n_iter=0: no iterative purification (faster, ok for scalar fields)
# masked_on_input=True: maps are already multiplied by mask
f_cmb = nmt.NmtField(cmb_mask, [cmb_masked], wcs=wcs, n_iter=0, masked_on_input=True, lmax=lmax)
f_template = nmt.NmtField(combined, [template], wcs=wcs, n_iter=0, masked_on_input=True, lmax=lmax)




# =============================================================================
# DEFINE MULTIPOLE BINNING
# =============================================================================
# Binning scheme: wide-fine-wide
#   ℓ = 800-2000:  Δℓ = 400 (lower S/N, wider bins)
#   ℓ = 2000-6500: Δℓ = 400 (peak signal region)
#   ℓ = 6500-lmax: Δℓ = 800 (high-ℓ, wider bins)

edges = list(np.arange(800, 2000, 400))
edges += list(np.arange(2000, 6500, 400))
edges += list(np.arange(6500, lmax + 1, 800))

if edges[-1] <= lmax:
    edges.append(lmax + 1)

edges = np.unique(edges).astype(int)

b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

# =============================================================================
# LOAD PRECOMPUTED WORKSPACES
# =============================================================================
# INPUT 5: Mode-coupling workspace from stage_spectra
# Contains the binned coupling matrix M_bb' for CMB × template
wsp = nmt.NmtWorkspace()
wsp.read_from(f'/home/jiaqu/Thumbstack_DESI/output/z_all_new_mask/stage_spectra/workspace_cmb_template_no_scr_mask.fits')

# INPUT 6: Covariance coupling workspace (EXPENSIVE to compute, ~hours)
# This contains the 4-field coupling coefficients needed for Gaussian covariance
cw = nmt.NmtCovarianceWorkspace()

# To recompute (takes many hours):
# cw.compute_coupling_coefficients(f_cmb, f_template, f_cmb, f_template)
# cw.write_to("/scratch/jiaqu/desi/catalogue/cov_wp_test.fits")

cw.read_from("/scratch/jiaqu/desi/catalogue/cov_wp_test.fits")



# =============================================================================
# LOAD CMB AUTO-SPECTRUM (C_aa)
# =============================================================================
# INPUT 7: CMB alms for computing the CMB auto-spectrum
import healpy as hp
cmb_alms = hp.read_alm("/scratch/jiaqu/desi/catalogue/cmb_alms_car_nosrc_sub_mask.fits")
cmb_cls = cs.alm2cl(cmb_alms)

# f_sky correction for CMB mask
# This converts pseudo-Cl to full-sky Cl estimate
F_SKY_CMB = 0.30340039899762616
claa = cmb_cls / F_SKY_CMB

# =============================================================================
# BUILD BINNING MATRIX
# =============================================================================
# Recreate edges for binning matrix construction
edges = list(np.arange(800, 2000, 400))
edges += list(np.arange(2000, 6500, 400))
edges += list(np.arange(6500, lmax + 1, 800))

if edges[-1] <= lmax:
    edges.append(lmax + 1)

edges = np.unique(edges).astype(int)

b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])

print("Number of bins:", b.get_n_bands())
print("Bin centers:", b.get_effective_ells())

# Build binning matrix M: converts per-ℓ to bandpowers
# M[b, ℓ] = 1/Δℓ if ℓ in bin b, else 0
nbins = len(edges) - 1
M = np.zeros((nbins, lmax + 1))

for i in range(nbins):
    lo, hi = edges[i], edges[i + 1]
    M[i, lo:hi] = 1.0 / (hi - lo)

# Effective ℓ for each bin
ell = M @ np.arange(10001)

# =============================================================================
# LOAD THEORY CROSS-SPECTRUM (C_ab) FOR OFF-DIAGONAL COVARIANCE
# =============================================================================
# Redshift bin paths
paths = ["z1_new_mask", "z2_new_mask", "z3z4_merged_y3_template", "z4_new_mask", "z_all_new_mask"]

# INPUT 8: Theory C_l^{Tg} for each redshift bin
# Shape: (4, Nbins) - one row per z-bin (z1, z2, z3, z4)
Cl_tg_b_list = np.loadtxt(f"/home/jiaqu/Thumbstack_DESI/binned_theory.txt")


theory34=np.loadtxt(f"/home/jiaqu/Thumbstack_DESI/binned_theory_merged.txt")
Cl_tg_b_list[2]=theory34
# Theory for the coadded (z_all) cross-spectrum
# This is the weighted average of the 4 individual z-bin theories
Cl_tg_b = np.array([3.65535094e-10, 2.62909121e-10, 1.93385292e-10, 1.43442068e-10,
       1.07422532e-10, 8.17194783e-11, 6.19107857e-11, 4.79318681e-11,
       3.71981931e-11, 2.94734588e-11, 2.28942661e-11, 1.87451733e-11,
       1.46720688e-11, 1.17916278e-11, 1.05323159e-11, 8.21283532e-12,
       5.26152060e-12, 3.37831716e-12, 2.09036104e-12, 1.60718650e-12])

# Stack all theories: rows 0-3 = z1-z4, row 4 = z_all
Cl_tg_b_list = np.vstack([Cl_tg_b_list, Cl_tg_b])



# =============================================================================
# MAIN LOOP: COMPUTE COVARIANCE FOR EACH REDSHIFT BIN
# =============================================================================
# f_sky correction for galaxy mask
F_SKY_GAL = 0.12032733727660157

for i in range(len(paths)):
    print(f"\n{'='*60}")
    print(f"Processing redshift bin {i+1}/{len(paths)}: {paths[i]}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # INPUT 9: Template alms for this z-bin (for C_bb, template auto-spectrum)
    # -------------------------------------------------------------------------
    template_alms = hp.read_alm(f"/home/jiaqu/Thumbstack_DESI/output/{paths[i]}/stage_template/template_alm_car_nosrc_sub_mask.fits")
    template_cls = cs.alm2cl(template_alms)

    # f_sky correction for template auto-spectrum
    clbb = template_cls / F_SKY_GAL

    # Recompute claa (in case it was modified in previous iteration)
    claa = cmb_cls / F_SKY_CMB

    # -------------------------------------------------------------------------
    # INTERPOLATE THEORY C_ab TO PER-ℓ
    # -------------------------------------------------------------------------
    # Theory is binned; interpolate to get piecewise-constant per-ℓ values
    # This is what NaMaster expects for the covariance calculation
    Cl_tg_pc = np.interp(np.arange(10001), ell, Cl_tg_b_list[i])

    # Reshape for NaMaster: (Nspectra, ℓmax+1)
    # For spin-0 × spin-0, Nspectra = 1
    clab = Cl_tg_pc[None, :]  # (1, lmax+1) - THEORY CROSS for off-diagonals
    print(f"Theory C_ab shape: {clab.shape}")

    claa = claa[None, :]  # (1, lmax+1)
    clbb = clbb[None, :]  # (1, lmax+1)

    # -------------------------------------------------------------------------
    # COMPUTE GAUSSIAN COVARIANCE
    # -------------------------------------------------------------------------
    # The Gaussian covariance formula for C_l^{ab} is:
    #   Cov(C_l^{ab}, C_l'^{ab}) ∝ C_l^{aa} C_l'^{bb} + C_l^{ab} C_l'^{ba}
    #
    # Arguments to gaussian_covariance:
    #   cw: covariance workspace (coupling coefficients)
    #   0, 0, 0, 0: spins of the 4 fields (all scalar)
    #   claa, clab: C^{ac}, C^{ad} where a=CMB, c=CMB, d=template
    #   clab, clbb: C^{bc}, C^{bd} where b=template
    #   wa=wsp: workspace for decoupling
    #   coupled=False: return decoupled (deconvolved) covariance
    cov = nmt.gaussian_covariance(
        cw, 0, 0, 0, 0,
        claa, clab,  # C_aa, C_ab (CMB auto, theory cross)
        clab, clbb,  # C_ab, C_bb (theory cross, template auto)
        wa=wsp, coupled=False
    )

    # -------------------------------------------------------------------------
    # SAVE OUTPUT
    # -------------------------------------------------------------------------
    output_path = f'/home/jiaqu/Thumbstack_DESI/output/{paths[i]}/stage_spectra/namaste_covariance_{paths[i]}.npy'
    np.save(output_path, cov)
    print(f"[out] Covariance shape: {cov.shape}")
    print(f"[out] Saved to: {output_path}")