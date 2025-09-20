import numpy as np
import healpy as hp
from astropy.table import Table, vstack
import os

def create_survey_mask_from_randoms(random_files, nside=2048, dec_limit=-20):
    """
    Create a HEALPix survey mask from DESI random catalogs
    
    Parameters:
    -----------
    random_files : list
        List of random catalog file paths
    nside : int
        HEALPix resolution parameter
    dec_limit : float
        Declination limit in degrees (default: -20 for DESI)
    
    Returns:
    --------
    mask : array
        HEALPix mask (1 where survey coverage exists, 0 elsewhere)
    """
    
    # Read and combine all random catalogs
    print(f"Reading {len(random_files)} random catalog files...")
    all_randoms = []
    
    for i, file_path in enumerate(random_files):
        if i % 2 == 0:  # Print more frequently since we have fewer files
            print(f"Processing file {i+1}/{len(random_files)}: {os.path.basename(file_path)}")
        
        try:
            ran_table = Table.read(file_path, format='fits')
            all_randoms.append(ran_table)
            print(f"  Loaded {len(ran_table)} objects from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            continue
    
    if not all_randoms:
        raise ValueError("No random catalogs could be read")
    
    # Combine all random catalogs
    combined_randoms = vstack(all_randoms)
    print(f"Total random objects: {len(combined_randoms)}")
    
    # Extract coordinates
    ra = combined_randoms['RA']
    dec = combined_randoms['DEC']
    
    # Apply declination cut
    dec_mask = dec >= dec_limit
    ra_filtered = ra[dec_mask]
    dec_filtered = dec[dec_mask]
    
    print(f"Objects after dec > {dec_limit} cut: {len(ra_filtered)}")
    
    # Convert to HEALPix
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix)
    
    # Convert coordinates to HEALPix pixels
    theta = np.deg2rad(90 - dec_filtered)  # Convert dec to colatitude
    phi = np.deg2rad(ra_filtered)
    
    pix_indices = hp.ang2pix(nside, theta, phi)
    
    # Mark pixels with randoms as part of the survey
    mask[pix_indices] = 1
    
    # Apply additional declination mask to HEALPix map
    theta_all, phi_all = hp.pix2ang(nside, np.arange(npix))
    dec_all = 90 - np.rad2deg(theta_all)
    healpix_dec_mask = dec_all >= dec_limit
    
    mask *= healpix_dec_mask
    
    print(f"Fraction of sky covered: {np.sum(mask)/len(mask):.4f}")
    
    return mask

def get_all_random_files(directory):
    """Get all LRG random catalog files from directory (both NGC and SGC)"""
    files = []
    for filename in os.listdir(directory):
        # Look for the specific DESI LSS pattern: LRG_[NGC/SGC]_[0-9]_clustering.ran.fits
        if (('LRG_NGC_' in filename or 'LRG_SGC_' in filename) and 
            'clustering.ran.fits' in filename):
            files.append(os.path.join(directory, filename))
    return sorted(files)

def analyze_random_files(directory):
    """Analyze what random files are available"""
    files = get_all_random_files(directory)
    
    ngc_files = [f for f in files if 'NGC' in f]
    sgc_files = [f for f in files if 'SGC' in f]
    
    print(f"Found {len(files)} total random catalog files:")
    print(f"  NGC files: {len(ngc_files)}")
    print(f"  SGC files: {len(sgc_files)}")
    
    if ngc_files:
        print(f"  NGC files: {[os.path.basename(f) for f in ngc_files]}")
    if sgc_files:
        print(f"  SGC files: {[os.path.basename(f) for f in sgc_files]}")
    
    return files, ngc_files, sgc_files

# Main execution
main_directory = "/project/rrg-rbond-ac/jiaqu/DESI/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/"
output_directory = "/scratch/jiaqu/desi/catalogue/"  # For output files

os.makedirs(output_directory, exist_ok=True)

print("Analyzing available random catalog files...")
print(f"Looking in directory: {main_directory}")

try:
    random_files, ngc_files, sgc_files = analyze_random_files(main_directory)
except FileNotFoundError:
    print(f"Directory not found: {main_directory}")
    print("Please check the path and permissions.")
    exit(1)

if not random_files:
    print("No random catalog files found. Listing all files in directory:")
    try:
        all_files = os.listdir(main_directory)
        lrg_files = [f for f in all_files if 'LRG' in f]
        ran_files = [f for f in all_files if '.ran.' in f or 'random' in f.lower()]
        print(f"Files containing 'LRG': {lrg_files[:10]}...")
        print(f"Files containing 'ran' or 'random': {ran_files[:10]}...")
    except Exception as e:
        print(f"Error listing directory: {e}")
else:
    # Create mask using both NGC and SGC
    nside = 8192  # High resolution, adjust as needed
    print(f"\nCreating survey mask with nside={nside}...")
    
    survey_mask = create_survey_mask_from_randoms(random_files, nside=nside)

    # Save the mask
    output_file = output_directory + "desi_survey_mask_healpix.fits"
    hp.write_map(output_file, survey_mask, overwrite=True)
    print(f"Survey mask saved to: {output_file}")

    # Optional: Create a plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        hp.mollview(survey_mask, title="DESI Survey Mask (NGC + SGC)", cmap='RdYlBu_r')
        plt.savefig(output_directory + "desi_survey_mask_plot.png", 
                    dpi=300, bbox_inches='tight')
        print("Survey mask plot saved")
        plt.close()
    except Exception as e:
        print(f"Could not create plot: {e}")

    if ngc_files and sgc_files:
        print("\nCreating separate NGC and SGC masks...")
        
        # NGC mask
        print("Creating NGC mask...")
        ngc_mask = create_survey_mask_from_randoms(ngc_files, nside=nside)
        ngc_output = output_directory + "desi_ngc_mask_healpix.fits"
        hp.write_map(ngc_output, ngc_mask, overwrite=True)
        print(f"NGC mask saved to: {ngc_output}")
        
        # SGC mask
        print("Creating SGC mask...")
        sgc_mask = create_survey_mask_from_randoms(sgc_files, nside=nside)
        sgc_output = output_directory + "desi_sgc_mask_healpix.fits"
        hp.write_map(sgc_output, sgc_mask, overwrite=True)
        print(f"SGC mask saved to: {sgc_output}")
        
        # Plot separate masks
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            hp.mollview(ngc_mask, title="DESI NGC Survey Mask", cmap='RdYlBu_r', hold=True)
            
            plt.subplot(2, 1, 2)
            hp.mollview(sgc_mask, title="DESI SGC Survey Mask", cmap='RdYlBu_r', hold=True)
            
            plt.tight_layout()
            plt.savefig(output_directory + "desi_ngc_sgc_masks.png", 
                        dpi=300, bbox_inches='tight')
            print("NGC and SGC mask plots saved")
            plt.close()
        except Exception as e:
            print(f"Could not create separate plots: {e}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total files processed: {len(random_files)}")
    print(f"NGC files: {len(ngc_files)}")
    print(f"SGC files: {len(sgc_files)}")
    print(f"Survey mask nside: {nside}")
    print(f"Survey mask saved to: {output_file}")

print("Survey mask creation complete")