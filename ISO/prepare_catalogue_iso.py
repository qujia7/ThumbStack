"""
Self-contained script to prepare the Y1 LRG catalogue for the ISO (SO) analysis.

- Reads the Y1 iron v1.2 unblinded LSS catalogues.
- Uses the ISO footprint mask to select objects overlapping the CMB map.
- No NGC/SGC split saved; only the merged full catalogue is written.
- Produces:
    full_catalog_Y1_renorm_ISO.txt -> RA, DEC, Z, VEL_LOS_RENORM
"""

import os
import numpy as np
import pandas as pd
from astropy.table import Table
from cosmoprimo.fiducial import DESI
from pixell import enmap


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
main_directory     = '/project/rrg-rbond-ac/jiaqu/DESI/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/'
post_rec_directory = 'desipipe/baseline_2pt/recon_recsym/'

# ISO footprint mask used to extract the catalogue footprint.
mask_path          = '/home/r/rbond/jiaqu/projects/SO/ISO/mask_i1_20250704.fits'

output_dir         = '/home/jiaqu/Thumbstack_DESI/output/catalogue/'

# Threshold above which an object is considered inside the CMB footprint.
overlap_thresh     = 0.95

# Effective redshift for the Y1 LRG sample (used for the renorm growth factor).
z_eff              = 0.780


# ---------------------------------------------------------------------------
# Load catalogues
# ---------------------------------------------------------------------------
print("Loading Y1 LRG catalogues...")

dat_pre_rec_NGC  = Table.read(main_directory + 'LRG_NGC_clustering.dat.fits', format='fits')
pre_rec_NGC      = dat_pre_rec_NGC.to_pandas()

dat_post_rec_NGC = Table.read(main_directory + post_rec_directory + 'LRG_NGC_clustering.dat.fits', format='fits')
post_rec_NGC     = dat_post_rec_NGC.to_pandas()

dat_pre_rec_SGC  = Table.read(main_directory + 'LRG_SGC_clustering.dat.fits', format='fits')
pre_rec_SGC      = dat_pre_rec_SGC.to_pandas()

dat_post_rec_SGC = Table.read(main_directory + post_rec_directory + 'LRG_SGC_clustering.dat.fits', format='fits')
post_rec_SGC     = dat_post_rec_SGC.to_pandas()

print(f"  NGC pre-rec : {len(pre_rec_NGC)}")
print(f"  NGC post-rec: {len(post_rec_NGC)}")
print(f"  SGC pre-rec : {len(pre_rec_SGC)}")
print(f"  SGC post-rec: {len(post_rec_SGC)}")

pre_rec_NGC["GC"] = "NGC"
pre_rec_SGC["GC"] = "SGC"


# ---------------------------------------------------------------------------
# Compute LOS displacements and velocities
# ---------------------------------------------------------------------------
cosmo = DESI()
f_eff = cosmo.growth_rate(z_eff)

# Comoving distances (Mpc/h)
chi_pre_NGC  = cosmo.comoving_radial_distance(pre_rec_NGC['Z'])
chi_post_NGC = cosmo.comoving_radial_distance(post_rec_NGC['Z'])
chi_pre_SGC  = cosmo.comoving_radial_distance(pre_rec_SGC['Z'])
chi_post_SGC = cosmo.comoving_radial_distance(post_rec_SGC['Z'])

# LOS displacements in Mpc (sign convention from DA2 script).
disp_NGC_LOS = -(chi_post_NGC - chi_pre_NGC) / cosmo.h
disp_SGC_LOS = -(chi_post_SGC - chi_pre_SGC) / cosmo.h

# RSD correction.
disp_NGC_LOS /= (1 + f_eff)
disp_SGC_LOS /= (1 + f_eff)

pre_rec_NGC["DISP_LOS"] = disp_NGC_LOS
pre_rec_SGC["DISP_LOS"] = disp_SGC_LOS

# LOS velocities (km/s).
f_NGC   = cosmo.growth_rate(post_rec_NGC['Z'])
H_z_NGC = cosmo.hubble_function(post_rec_NGC['Z'])
a_NGC   = 1 / (1 + post_rec_NGC['Z'])

f_SGC   = cosmo.growth_rate(post_rec_SGC['Z'])
H_z_SGC = cosmo.hubble_function(post_rec_SGC['Z'])
a_SGC   = 1 / (1 + post_rec_SGC['Z'])

vel_NGC_LOS = a_NGC * H_z_NGC * f_NGC * disp_NGC_LOS
vel_SGC_LOS = a_SGC * H_z_SGC * f_SGC * disp_SGC_LOS

pre_rec_NGC["VEL_LOS"] = vel_NGC_LOS
pre_rec_SGC["VEL_LOS"] = vel_SGC_LOS

# Renormalised velocity (rescaled by growth factor relative to z_eff).
renorm_NGC = cosmo.growth_factor(pre_rec_NGC["Z"]) / cosmo.growth_factor(z_eff)
renorm_SGC = cosmo.growth_factor(pre_rec_SGC["Z"]) / cosmo.growth_factor(z_eff)

pre_rec_NGC["VEL_LOS_RENORM"] = vel_NGC_LOS * renorm_NGC
pre_rec_SGC["VEL_LOS_RENORM"] = vel_SGC_LOS * renorm_SGC


# ---------------------------------------------------------------------------
# Merge NGC + SGC (no split saved for ISO) and sort by redshift
# ---------------------------------------------------------------------------
pre_rec      = pd.concat([pre_rec_NGC, pre_rec_SGC])
pre_rec_sort = pd.DataFrame(pre_rec).sort_values("Z").reset_index(drop=True)

print(f"\nMerged Y1 LRG catalogue: {len(pre_rec_sort)} objects")


# ---------------------------------------------------------------------------
# Apply ISO footprint filtering
# ---------------------------------------------------------------------------
def sky2map(ra, dec, cmb_map):
    """Return CMB map values at (ra, dec) in degrees using nearest neighbour.
    Returns 0 outside the map.
    """
    sourcecoord = np.array([dec, ra]) * (np.pi / 180)
    return cmb_map.at(sourcecoord, order=0)


def apply_overlap_filter(catalog_df, cmb_mask, thresh=0.95, name="catalog"):
    ra  = np.array(catalog_df["RA"])
    dec = np.array(catalog_df["DEC"])
    n_in = len(ra)
    print(f"Applying ISO overlap filter to {name}: {n_in} objects")

    hit          = sky2map(ra, dec, cmb_mask)
    overlap_flag = (np.array(hit) > thresh)

    filtered = catalog_df[overlap_flag]
    print(f"  -> {len(filtered)} objects retained ({len(filtered)/n_in*100:.1f}%)")
    return filtered


print(f"\nLoading ISO mask: {mask_path}")
cmb_mask = enmap.read_fits(mask_path)

pre_rec_sort_ISO = apply_overlap_filter(pre_rec_sort, cmb_mask,
                                        thresh=overlap_thresh,
                                        name="Full Y1 LRG")


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)

out_vel_renorm = os.path.join(output_dir, "full_catalog_Y1_renorm_ISO.txt")

print(f"\nSaving catalogue to {output_dir}")
np.savetxt(out_vel_renorm,
           np.array(pre_rec_sort_ISO[["RA", "DEC", "Z", "VEL_LOS_RENORM"]]))
print(f"  {out_vel_renorm} ({len(pre_rec_sort_ISO)} objects)")

print("\nDone.")
