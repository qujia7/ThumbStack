import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cosmoprimo.fiducial import DESI # tienes que tener el environment de cosmodesi
from astropy.table import Table
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from pixell import enmap
import copy
import matplotlib
from scipy import special, optimize, integrate, stats
# import classy module
from classy import Class
from scipy import integrate
from cosmoprimo.fiducial import DESI # tienes que tener el environment de cosmodesi
import pandas as pd
import pyclass


# Importing catalogs

main_directory = '//project/rrg-rbond-ac/jiaqu/DESI/catalogs/DA2/'
# pre_rec_directory = 'analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/recon_sm15_IFFT_recsym/'
# post_rec_directory = "LSS/loa-v1/LSScats/v1.1/nonKP/"
post_rec_directory = 'analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/recon_sm15_IFFT_recsym/'
pre_rec_directory = "LSS/loa-v1/LSScats/v1.1/nonKP/"


# North Galactic Cap:
dat_pre_rec_NGC = Table.read(main_directory+pre_rec_directory+'LRG_NGC_clustering.dat.fits', format='fits')
pre_rec_NGC = dat_pre_rec_NGC.to_pandas()

dat_post_rec_NGC = Table.read(main_directory+post_rec_directory+'LRG_NGC_clustering.dat.fits', format='fits')
post_rec_NGC = dat_post_rec_NGC.to_pandas()

# South Galactic Cap:
dat_pre_rec_SGC = Table.read(main_directory+pre_rec_directory+'LRG_SGC_clustering.dat.fits', format='fits')
pre_rec_SGC = dat_pre_rec_SGC.to_pandas()

dat_post_rec_SGC = Table.read(main_directory+post_rec_directory+'LRG_SGC_clustering.dat.fits', format='fits')
post_rec_SGC = dat_post_rec_SGC.to_pandas()

# All (for visualization purposes)
pre_rec = pd.concat([pre_rec_NGC, pre_rec_SGC])

# DESI Legacy survey:
bins_directory = "/project/rrg-rbond-ac/jiaqu/DESI_LRG_legacy/dr9_lrg_pzbins.fits"
dat_bins = Table.read(bins_directory, format='fits')
bins = dat_bins.to_pandas()

# Let's now obtain the displacements and velocities
cosmo = DESI()

# From Y1:
z_eff =  0.780
f_eff = cosmo.growth_rate(0.780) # No units

pre_rec_NGC["GC"] = "NGC"
pre_rec_SGC["GC"] = "SGC"

######################################################################

# Displacements along the LOS:
chi_pre_NGC = cosmo.comoving_radial_distance(pre_rec_NGC['Z']) # Mpc/h
chi_post_NGC = cosmo.comoving_radial_distance(post_rec_NGC['Z']) # Mpc/h

chi_pre_SGC = cosmo.comoving_radial_distance(pre_rec_SGC['Z']) # Mpc/h
chi_post_SGC = cosmo.comoving_radial_distance(post_rec_SGC['Z']) # Mpc/h

#why is the negative sign make sense here
disp_NGC_LOS = - (chi_post_NGC - chi_pre_NGC) / cosmo.h # Mpc
disp_SGC_LOS = - (chi_post_SGC - chi_pre_SGC) / cosmo.h # Mpc

disp_NGC_LOS /= (1 + f_eff) # Mpc, RSD correction
disp_SGC_LOS /= (1 + f_eff) # Mpc, RSD correction

pre_rec_NGC["DISP_LOS"] = disp_NGC_LOS # Mpc
pre_rec_SGC["DISP_LOS"] = disp_SGC_LOS # Mpc

######################################################################

# Velocities along the LOS: 
f_NGC = cosmo.growth_rate(post_rec_NGC['Z']) # No units
H_z_NGC = cosmo.hubble_function(post_rec_NGC['Z']) # km / (s Mpc)
a_NGC = 1/(1+post_rec_NGC['Z']) # No units

f_SGC = cosmo.growth_rate(post_rec_SGC['Z']) # No units
H_z_SGC = cosmo.hubble_function(post_rec_SGC['Z']) # km / (s Mpc)
a_SGC = 1/(1+post_rec_SGC['Z']) # No units

vel_NGC_LOS = a_NGC * H_z_NGC * f_NGC * disp_NGC_LOS # km/s
pre_rec_NGC["VEL_LOS"] = vel_NGC_LOS

vel_SGC_LOS = a_SGC * H_z_SGC * f_SGC * disp_SGC_LOS # km/s
pre_rec_SGC["VEL_LOS"] = vel_SGC_LOS

# ######################################################################

renorm_NGC = cosmo.growth_factor(pre_rec_NGC["Z"])/cosmo.growth_factor(z_eff)
renorm_SGC = cosmo.growth_factor(pre_rec_SGC["Z"])/cosmo.growth_factor(z_eff)

pre_rec_NGC["VEL_LOS_RENORM"] = vel_NGC_LOS*renorm_NGC
pre_rec_SGC["VEL_LOS_RENORM"] = vel_SGC_LOS*renorm_SGC

# Merge the NGC and SGC samples
pre_rec = pd.concat([pre_rec_NGC, pre_rec_SGC])

# We sort them by redshift
pre_rec_sort = pd.DataFrame(pre_rec).sort_values("Z")

#also construct the non merged catalogues
pre_rec_NGC_sort = pd.DataFrame(pre_rec_NGC).sort_values("Z")
pre_rec_SGC_sort = pd.DataFrame(pre_rec_SGC).sort_values("Z")


# Select the ones overlapping with ACT

def sky2map(ra, dec, cmbMap):
    '''Gives the map value at coordinates (ra, dec).
    ra, dec in degrees.
    Uses nearest neighbor, no interpolation.
    Will return 0 if the coordinates requested are outside the map
    '''
    # interpolate the map to the given sky coordinates
    sourcecoord = np.array([dec, ra]) * (np.pi / 180)   # convert from degrees to radians
    # use nearest neighbor interpolation
    return cmbMap.at(sourcecoord, order=0)

def apply_act_overlap_filter(catalog_df, cmbMask, thresh=0.95, name="catalog"):
    """
    Apply ACT overlap filtering to a catalog DataFrame
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Input catalog with RA, DEC columns
    cmbMask : enmap
        CMB mask map
    thresh : float
        Threshold for overlap (default 0.95)
    name : str
        Name for logging purposes
        
    Returns:
    --------
    pandas.DataFrame : Filtered catalog
    """
    ra = np.array(catalog_df["RA"])
    dec = np.array(catalog_df["DEC"])
    nObj = len(ra)
    
    print(f"Applying ACT overlap filter to {name}: {nObj} objects")
    
    # Vectorized approach for better performance
    hit = sky2map(ra, dec, cmbMask)
    overlapFlag = np.array(hit>thresh)*1
    
    filtered_catalog = catalog_df[overlapFlag==1]
    print(f"After ACT filtering - {name}: {len(filtered_catalog)} objects ({len(filtered_catalog)/nObj*100:.1f}%)")
    
    return filtered_catalog

def create_redshift_bins(catalog_df, z_bins=None, name="catalog"):
    """
    Create redshift bins from a catalog
    
    Parameters:
    -----------
    catalog_df : pandas.DataFrame
        Input catalog with Z column
    z_bins : list of tuples
        List of (z_min, z_max) tuples. If None, uses default bins
    name : str
        Name for logging purposes
        
    Returns:
    --------
    dict : Dictionary with bin names as keys and DataFrames as values
    """
    if z_bins is None:
        z_bins = [
            (0.4, 0.6),
            (0.6, 0.8), 
            (0.8, 0.95),
            (0.95, 1.1)
        ]
    
    binned_catalogs = {}
    
    for i, (z_min, z_max) in enumerate(z_bins, 1):
        bin_catalog = catalog_df[(catalog_df['Z'] > z_min) & (catalog_df['Z'] <= z_max)]
        bin_name = f"{name}_zbin{i}"
        binned_catalogs[bin_name] = bin_catalog
        print(f"Z-bin {i} ({z_min}-{z_max}): {len(bin_catalog)} objects")
    
    return binned_catalogs

# Load CMB maps
print("Loading CMB maps...")
cmbMap = enmap.read_fits("/project/rrg-rbond-ac/msyriac/ilc_dr6v3/20230606/hilc_fullRes_TT_17000.fits")
cmbMask = enmap.read_fits("/home/jiaqu/Thumbstack_DESI/wide_mask_GAL070_apod_1.50_deg_wExtended_srcfree_Will.fits")

# Apply ACT overlap filtering to all catalogs
print("\n=== Applying ACT Overlap Filtering ===")
pre_rec_sort_ACT = apply_act_overlap_filter(pre_rec_sort, cmbMask, name="Full catalog")
pre_rec_NGC_sort_ACT = apply_act_overlap_filter(pre_rec_NGC_sort, cmbMask, name="NGC catalog") 
pre_rec_SGC_sort_ACT = apply_act_overlap_filter(pre_rec_SGC_sort, cmbMask, name="SGC catalog")

# Create redshift bins for all filtered catalogs
print("\n=== Creating Redshift Bins ===")
print("Full catalog bins:")
full_zbins = create_redshift_bins(pre_rec_sort_ACT, name="full")

print("\nNGC catalog bins:")
ngc_zbins = create_redshift_bins(pre_rec_NGC_sort_ACT, name="NGC")

print("\nSGC catalog bins:")
sgc_zbins = create_redshift_bins(pre_rec_SGC_sort_ACT, name="SGC")

# For backward compatibility, keep the original variable names
df = pre_rec_sort_ACT
df1 = full_zbins['full_zbin1_0.4_0.6']
df2 = full_zbins['full_zbin2_0.6_0.8']
df3 = full_zbins['full_zbin3_0.8_0.95']
df4 = full_zbins['full_zbin4_0.95_1.1']

# Optional: Save all catalogs
def save_catalogs(output_dir="/home/jiaqu/Thumbstack_DESI/output/catalogue/", save_format="txt"):
    """
    Save all processed catalogs to files
    
    Parameters:
    -----------
    output_dir : str
        Directory to save catalogs
    save_format : str
        Format to save ('txt', 'csv', 'fits')
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main catalogs
    catalogs_to_save = {
        'full_catalog_Y3': pre_rec_sort_ACT,
        'NGC_catalog_Y3': pre_rec_NGC_sort_ACT,
        'SGC_catalog_Y3': pre_rec_SGC_sort_ACT
    }
    
    # Add redshift bins
    catalogs_to_save.update(full_zbins)
    catalogs_to_save.update(ngc_zbins)
    catalogs_to_save.update(sgc_zbins)
    
    print(f"\n=== Saving Catalogs to {output_dir} ===")
    for name, catalog in catalogs_to_save.items():
        if len(catalog) > 0:
            if save_format == "txt":
                # Save in format compatible with ThumbStack (RA, DEC, Z, VEL_LOS_RENORM)
                output_file = f"{output_dir}/{name}.txt"
                # Use the same format as your original method
                np.savetxt(output_file, np.array(catalog[["RA", "DEC", "Z", "VEL_LOS_RENORM"]]))
                print(f"Saved {name}: {len(catalog)} objects -> {output_file}")
            elif save_format == "csv":
                output_file = f"{output_dir}/{name}.csv"
                catalog.to_csv(output_file, index=False)
                print(f"Saved {name}: {len(catalog)} objects -> {output_file}")
        else:
            print(f"Skipping {name}: empty catalog")

# Uncomment to save catalogs
save_catalogs( save_format="txt")

print(f"\n=== Summary ===")
print(f"Full catalog (ACT filtered): {len(pre_rec_sort_ACT)} objects")
print(f"NGC catalog (ACT filtered): {len(pre_rec_NGC_sort_ACT)} objects") 
print(f"SGC catalog (ACT filtered): {len(pre_rec_SGC_sort_ACT)} objects")
print(f"Total in z-bins: {sum(len(cat) for cat in full_zbins.values())} objects")