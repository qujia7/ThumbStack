# Standard libraries
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import healpy as hp
from scipy.stats import chi2, binned_statistic as binnedstat

# Set up argument parser
parser = argparse.ArgumentParser(description='Process galaxy catalog and create velocity template with healpy.')
parser.add_argument('--catalog', type=str, 
                    default="/scratch/jiaqu/desi/catalogue/full_catalog_Y3_test_swap.txt",
                    help='Path to the galaxy catalog file')
parser.add_argument('--nside', type=int, default=8192,
                    help='HEALPix nside parameter (power of 2, default: 2048)')
parser.add_argument('--lmax', type=int, default=10000,
                    help='Maximum multipole for alms calculation')
parser.add_argument('--output', type=str, 
                    default="/scratch/jiaqu/desi/catalogue/",
                    help='Output directory for maps and other products')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output, exist_ok=True)

# Add custom path
sys.path.append('/home/jiaqu/Thumbstack_DESI/')

# External libraries
from cosmoprimo.fiducial import DESI

# Custom modules
import flat_map
import universe
import mass_conversion
from mass_conversion import *
import catalog
from catalog import *
import thumbstack
from thumbstack import *
import cmb
from cmb import *

# Initialize cosmology
u = DESI()

# Catalog configuration
catpath = "./"
catname = args.catalog  # Use the catalog path from command line arguments
massConv = MassConversionKravtsov14()
nObj = None

# Initialize galaxy catalog
galcat = Catalog(u,
                massConv,
                # catType='radec',                     
                # catType='stellar_mass',
                catType='no_mass',
                name=catname,
                nObj=nObj,
                pathInCatalog=catpath+catname,
                workDir=catpath,
                rV=0.65)

# Extract galaxy properties
RA = galcat.RA
DEC = galcat.DEC
v = -galcat.vR/3e5  # Velocity in units of c
N_gal=len(v)
print(f"number of galaxies {N_gal}")

# HEALPix settings
nside = args.nside
npix = hp.nside2npix(nside)
lmax = args.lmax
print(f"Using HEALPix with nside={nside}, npix={npix}, lmax={lmax}")

# Create empty template
template = np.zeros(npix)

# Create mask for declination cut (declination > -20 degrees)
dec_limit = -20  # in degrees
dec_limit_rad = np.deg2rad(dec_limit)
theta_limit = np.pi/2 - dec_limit_rad  # convert to co-latitude

# Create a mask for pixels above our declination limit
theta, phi = hp.pix2ang(nside, np.arange(npix))
mask = theta <= theta_limit  # Only include pixels with co-latitude <= theta_limit (dec >= dec_limit)



# Convert galaxy coordinates from degrees to radians
ra_rad = np.deg2rad(RA)
dec_rad = np.deg2rad(DEC)

# Convert to theta (co-latitude)
theta_rad = np.pi/2 - dec_rad  # Convert declination to co-latitude

# Apply declination cut to galaxies
valid_gal = dec_rad >= dec_limit_rad
ra_rad_valid = ra_rad[valid_gal]
theta_rad_valid = theta_rad[valid_gal]
v_valid = v[valid_gal]

print(f"Using {np.sum(valid_gal)} galaxies after declination cut")

# Find pixel indices for each galaxy
count_map = np.zeros(npix)  # To track number of galaxies per pixel

pix_indices = hp.ang2pix(nside, theta_rad_valid, ra_rad_valid)
# First pass: Count galaxies in each pixel
print("First pass: Counting galaxies in each pixel")
for idx in pix_indices:
    count_map[idx] += 1
print("number of galaxies pre masking")
print(np.sum(count_map))


# Apply mask to count map
count_map *= mask

print("number of galaxies after masking")
print(np.sum(count_map))
# Calculate mean galaxy count in masked region
n_mean = np.sum(count_map) / np.sum(mask)
print(f"Mean galaxy count per pixel: {n_mean:.6f}")

# Add velocity contributions to each pixel
for i, (idx, vel) in enumerate(zip(pix_indices, v_valid)):
    if i % 10000 == 0:
        print(f"Processing galaxy {i}/{len(pix_indices)}")
    
    template[idx] += vel 

# Apply mask to template (zero out regions below declination limit)
template *= mask
template/=n_mean
print("number of galaxies after masking and normalizing")
print(n_mean)
# Print the mean of the template
mean_value = np.mean(template[mask])
print(f"Template mean (in region above dec={dec_limit}): {mean_value}")

# Build output paths
map_filename = os.path.join(args.output, "velocity_template_healpy_all_test1.fits")
alm_filename = os.path.join(args.output, "velocity_template_alms_all_test1.fits")
cls_filename = os.path.join(args.output, "velocity_template_cls_test1.txt")
map_plot_filename = os.path.join(args.output, "velocity_template_map.png")
spectrum_plot_filename = os.path.join(args.output, "velocity_template_power_spectrum.png")

# Save the HEALPix map
hp.write_map(map_filename, template, overwrite=True)
print(f"Saved map to {map_filename}")

# Compute alms
print("Computing alms...")
alms = hp.map2alm(template, lmax=lmax, iter=3)  # iter=3 for better accuracy

# Save alms
hp.write_alm(alm_filename, alms, overwrite=True)
print(f"Saved alms to {alm_filename}")

# Compute power spectrum
cls = hp.alm2cl(alms)
np.savetxt(cls_filename, cls)
print(f"Saved power spectrum to {cls_filename}")


print("\nFor comparison with pixell map:")
print("1. Convert pixell map to HEALPix:")
print("   - Create a healpy map with matching resolution")
print("   - Interpolate values from pixell grid to HEALPix grid")
print("   - Compute alms and cls for the converted map")
print("2. Compare the power spectra to verify equivalence")