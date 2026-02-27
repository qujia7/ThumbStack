import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import fits
from astropy.wcs import WCS
from pixell import enmap, reproject, curvedsky as cs
import numpy as np
import argparse


p = argparse.ArgumentParser(description="create spectra")

p.add_argument("--output-dir", type=str,  default=None,help='Output directory.')

p.add_argument("--lmax", type=int, default=10000)
p.add_argument("--cmb-mask", type=str,
               default="/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits",
               help="CMB apodized mask")
p.add_argument("--gal-mask", type=str,
               default="/scratch/jiaqu/desi/catalogue/desi_survey_mask_car.fits",
               help="galactic mask from randoms")
p.add_argument("--dec-min", type=float, default=-20.0,
               help="Declination cut in degrees (keep dec >= dec_min)")
# Add this to your argument parser section
p.add_argument("--use_random", action='store_true',
               help="Use random catalog instead of real galaxy catalog")
args = p.parse_args()





print("loading maps...")

cmb_masked=enmap.read_map("./output/catalogue/cmb_map_car_nosrc_cluster_sub_mask.fits")
cmb_mask = enmap.read_map(args.cmb_mask)


#joint mask
gal_mask=enmap.read_map(args.gal_mask)
#make it binary
cmb_mask_binary=cmb_mask.copy()
cmb_mask_binary[cmb_mask_binary!=1]=0
combined=cmb_mask_binary*gal_mask

wcs = combined.wcs

template=enmap.read_map(f'{args.output_dir}/../stage_template/template_car_nosrc_sub_mask.fits')

print("Finished loading.")

lmax = args.lmax  # your true bandlimit


# # Create spin-0 field. Pass a WCS structure do define the rectangular pixels.
f_cmb = nmt.NmtField(cmb_mask, [cmb_masked], wcs=wcs, n_iter=0, masked_on_input=True,lmax=lmax)
# Create spin-2 field
f_template = nmt.NmtField(combined, [template], wcs=wcs, n_iter=0, masked_on_input=True,lmax=lmax)




# (ℓ_min, ℓ_max, Δℓ)   wide–fine–wide

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

# Test: recover ℓ centers
ells = np.arange(lmax + 1)
ell_centers = M @ ells






wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(f_cmb, f_template, b)

# Save the workspace
wsp_file = f'{args.output_dir}/../stage_spectra/workspace_cmb_template_no_scr_mask.fits'
wsp.write_to(wsp_file)
print(f"Workspace saved to: {wsp_file}")

# wsp= nmt.NmtWorkspace()
# wsp.read_from(f'{args.output_dir}/../stage_spectra/workspace_cmb_template_no_scr_mask.fits")


# Compute the pseudo cross-power spectrum
print("Computing pseudo-Cl...")
cl_pseudo = nmt.compute_coupled_cell(f_cmb, f_template)

# Apply the inverse mode-coupling matrix
print("Deconvolving window function...")
cl_deconvolved = wsp.decouple_cell(cl_pseudo)[0]

ell_eff = b.get_effective_ells()




np.savetxt(f'{args.output_dir}/../stage_spectra/cl_master.txt',
           np.column_stack([ell_eff, cl_deconvolved]), 
           header="ell Cl_CMB_x_template")
print(f'Spectra saved to: {args.output_dir}/../stage_spectra/cl_master.txt')


#get the covariance matrix

SNR = lambda auto, error : round(np.sqrt(np.sum(auto**2/error**2)),0)


f_sky=0.12032733727660157

def process_alms(a_alms, b_alms, f_sky, binning_matrix,pixwin=None,edges=None):
    
    acl = cs.alm2cl(a_alms)/0.30340039899762616
    bcl = cs.alm2cl(b_alms)/f_sky
    kcl = cs.alm2cl(a_alms, b_alms)/f_sky

    if pixwin is not None:
        acl/=pixwin**2
        bcl/=pixwin**2
        kcl/=pixwin**2
    kclb=binning_matrix@kcl[:10001]
    cl_e=binning_matrix@(np.nan_to_num(acl*bcl+kcl**2))[:10001]
    ells=binning_matrix@np.arange(10001)
    error = np.sqrt(1 / (2 * ells + 1) / np.diff(edges)/f_sky) * np.sqrt(cl_e)
    return ells,kclb,error

bin_matrix = lambda edges, lmax: np.array([[1.0/(min(edges[i+1], lmax+1) - max(edges[i], 0)) if max(edges[i], 0) <= l < min(edges[i+1], lmax+1) else 0.0 for l in range(lmax+1)] for i in range(len(edges)-1)])



B = M

import healpy as hp

p_alms=hp.read_alm(f"{args.output_dir}/../stage_template/template_alm_car_nosrc_sub_mask.fits")
cmb_alms=hp.read_alm("/scratch/jiaqu/desi/catalogue/cmb_alms_car_nosrc_sub_mask.fits")

ell,cl,error=process_alms(cmb_alms, p_alms, f_sky, B, edges=edges)

np.savetxt(f'{args.output_dir}/../stage_spectra/error.txt',error)



print("estimate the covmat")

#use the beam deconvolved cls to estimate the covariance

# import healpy as hp
# cmb_alms=hp.read_alm("./output/catalogue/cmb_alms_car_nosrc_sub_mask.fits")
# cmb_cls=cs.alm2cl(cmb_alms)
# claa=cmb_cls/0.31
# template_alms=hp.read_alm("/scratch/jiaqu/desi/catalogue/template_alm_car_nosrc_sub_mask.fits")
# template_cls=cs.alm2cl(template_alms)
# clbb=template_cls/0.12



# r = 0.65
# sigma_v  = 314/3e5
# sigma_vr = 233/3e5
# prefactor = r * sigma_v * sigma_vr / 3.0
# T_CMB = 2.7255e6  # μK

# el, oh, th, op2h = np.loadtxt(
#     "/home/jiaqu/Thumbstack_DESI/cleg_l_1h_2h_1p2h_1bin.txt",
#     unpack=True
# )

# el = np.asarray(el, dtype=float)
# oh = np.asarray(oh, dtype=float)
# th = np.asarray(th, dtype=float)

# Cl_tg_model = (oh + th) * float(prefactor) * float(T_CMB)  # shape = (len(el),)

# # Interpolate the model onto 0..lmax (outside range → 0)
# ell_grid = np.arange(lmax + 1, dtype=float)
# Cl_tg_theory = np.interp(ell_grid, el, Cl_tg_model, left=0.0, right=0.0).astype(np.float64)

# # Bin to your bandpowers
# Cl_tg_b = b.bin_cell(Cl_tg_theory[None, :])[0]  # (Nbin,)

# # Unbin back to per-ℓ piecewise-constant (NaMaster will re-bin internally)
# Cl_tg_pc = np.zeros(lmax + 1, dtype=np.float64)
# for ib, (l1, l2) in enumerate(zip(edges[:-1], edges[1:])):
#     l1 = max(int(l1), 0)
#     l2 = min(int(l2), lmax + 1)
#     Cl_tg_pc[l1:l2] = Cl_tg_b[ib]

# # Shape for NaMaster (Nspectra, ℓmax+1)
# clab = Cl_tg_pc[None, :]  # (1, lmax+1)  <-- THEORY CROSS USED IN COVARIANCE


# # Shape for NaMaster (Nspectra, ℓmax+1)
# claa = claa[None, :]
# clbb = clbb[None, :]

# # ---------- 6) Covariance workspace with  masks ----------
# cw = nmt.NmtCovarianceWorkspace()
# cw.compute_coupling_coefficients(f_cmb, f_template, f_cmb, f_template)

# # ---------- 7) Gaussian covariance of binned, decoupled (CMB×Template) ----------
# # Inputs (claa, clbb, clab) are FULL-SKY *theory-like per-ℓ* arrays (piecewise-constant).
# cov = nmt.gaussian_covariance(
#     cw, 0, 0, 0, 0,     # scalar fields
#     claa, clab,         # C^{ac}, C^{bd}  (autos)
#     clab, clbb,         # C^{ad}, C^{bc}  (cross; use your *theory* twice)
#     wa=wsp, wb=wsp
# )

# np.save("{args.output_dir}/../stage_spectra/namaste_error.txt", cov)
# print("[out] covariance shape:", cov.shape)