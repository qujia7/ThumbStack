
import numpy as np
from astropy.table import Table
from pixell import enmap
import os

# Paths
catalog_path = '/project/rrg-rbond-ac/jiaqu/DESI/catalogs/BGS/recon/catalog_BGS_BRIGHT-20.2_R12.50_nmesh512_recsym_MG_masked.fits'
mask_path    = '/home/jiaqu/Thumbstack_DESI/output/wide_mask_GAL070_apod_1.50_deg_wExtended_no_src_with_cluster.fits'
output_dir   = '/home/jiaqu/Thumbstack_DESI/bgs/output/catalogue/'


def sky2map(ra, dec, cmbMap):
    sourcecoord = np.array([dec, ra]) * (np.pi / 180)
    return cmbMap.at(sourcecoord, order=0)


def apply_act_overlap_filter(catalog_df, cmbMask, thresh=0.95, name="catalog"):
    ra   = np.array(catalog_df["RA"])
    dec  = np.array(catalog_df["DEC"])
    nObj = len(ra)
    print(f"Applying ACT overlap filter to {name}: {nObj} objects")
    hit         = sky2map(ra, dec, cmbMask)
    overlapFlag = np.array(hit > thresh) * 1
    filtered    = catalog_df[overlapFlag == 1]
    print(f"After ACT filtering - {name}: {len(filtered)} objects ({len(filtered)/nObj*100:.1f}%)")
    return filtered


# Load catalog
print("Loading BGS catalog...")
cat = Table.read(catalog_path).to_pandas()
print(f"Total objects: {len(cat)}")

# Load ACT mask and filter
print("Loading ACT mask...")
cmbMask = enmap.read_fits(mask_path)
cat_ACT = apply_act_overlap_filter(cat, cmbMask, name="BGS_BRIGHT-20.2")

# Sort by redshift
cat_ACT = cat_ACT.sort_values("Z")

# Save
os.makedirs(output_dir, exist_ok=True)
output_file = output_dir + 'BGS_BRIGHT-20.2_full_no_src_with_cluster_mask.txt'
np.savetxt(output_file, np.array(cat_ACT[["RA", "DEC", "Z", "vR"]]))
print(f"Saved {len(cat_ACT)} objects -> {output_file}")
