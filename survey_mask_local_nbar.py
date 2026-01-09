import numpy as np
import healpy as hp
from astropy.table import Table, vstack
import os
import argparse


def create_local_nbar_mask(random_files, catalog_path, nside=2048, dec_limit=-20):
    """
    Create a HEALPix local number density mask from DESI random catalogs

    Implements equation 8: mask = alpha_r * counts_per_pixel
    where alpha_r = N_data / N_randoms

    Parameters:
    -----------
    random_files : list
        List of random catalog file paths
    catalog_path : str
        Path to galaxy catalog (CSV with RA, DEC columns)
    nside : int
        HEALPix resolution parameter
    dec_limit : float
        Declination limit in degrees (default: -20 for DESI)

    Returns:
    --------
    mask : array
        HEALPix local nbar mask (alpha_r * random counts per pixel)
    diagnostics : dict
        Dictionary with N_data, N_randoms, alpha_r, and mask statistics
    """

    # Read galaxy catalog to get N_data
    print(f"Reading galaxy catalog: {catalog_path}")
    if catalog_path.endswith('.csv'):
        import pandas as pd
        catalog = pd.read_csv(catalog_path)
        n_data = len(catalog)
    elif catalog_path.endswith('.fits'):
        catalog = Table.read(catalog_path, format='fits')
        n_data = len(catalog)
    elif catalog_path.endswith('.txt'):
        catalog = np.loadtxt(catalog_path, skiprows=1)
        n_data = len(catalog)
    else:
        raise ValueError(f"Unsupported catalog format: {catalog_path}")

    print(f"N_data (galaxy catalog): {n_data}")

    # Read and combine all random catalogs
    print(f"\nReading {len(random_files)} random catalog files...")
    all_randoms = []

    for i, file_path in enumerate(random_files):
        if i % 2 == 0:
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
    n_randoms = len(combined_randoms)
    print(f"\nN_randoms (total random objects): {n_randoms}")

    # Extract coordinates
    ra = combined_randoms['RA']
    dec = combined_randoms['DEC']

    # Apply declination cut
    dec_mask = dec >= dec_limit
    ra_filtered = np.array(ra[dec_mask])
    dec_filtered = np.array(dec[dec_mask])
    n_randoms_filtered = len(ra_filtered)

    print(f"N_randoms after dec > {dec_limit} cut: {n_randoms_filtered}")

    # Convert to HEALPix
    npix = hp.nside2npix(nside)

    # Convert coordinates to HEALPix pixels
    theta = np.deg2rad(90 - dec_filtered)  # Convert dec to colatitude
    phi = np.deg2rad(ra_filtered)

    pix_indices = hp.ang2pix(nside, theta, phi)

    # Compute counts per pixel using bincount (key difference from binary mask)
    counts = np.bincount(pix_indices, minlength=npix).astype(float)

    # Compute alpha_r = N_data / N_randoms
    alpha_r = n_data / n_randoms_filtered
    print(f"\nalpha_r = N_data / N_randoms = {n_data} / {n_randoms_filtered} = {alpha_r:.6e}")

    # Create mask = alpha_r * counts
    mask = alpha_r * counts

    # Apply additional declination mask to HEALPix map
    theta_all, phi_all = hp.pix2ang(nside, np.arange(npix))
    dec_all = 90 - np.rad2deg(theta_all)
    healpix_dec_mask = dec_all >= dec_limit

    mask *= healpix_dec_mask

    # Compute mask statistics for non-zero pixels
    nonzero_mask = mask[mask > 0]
    mask_stats = {
        'min': np.min(nonzero_mask) if len(nonzero_mask) > 0 else 0,
        'max': np.max(nonzero_mask) if len(nonzero_mask) > 0 else 0,
        'mean': np.mean(nonzero_mask) if len(nonzero_mask) > 0 else 0,
        'std': np.std(nonzero_mask) if len(nonzero_mask) > 0 else 0,
        'n_nonzero': len(nonzero_mask)
    }

    print(f"\nMask statistics (non-zero pixels only):")
    print(f"  N_nonzero pixels: {mask_stats['n_nonzero']}")
    print(f"  Min: {mask_stats['min']:.6e}")
    print(f"  Max: {mask_stats['max']:.6e}")
    print(f"  Mean: {mask_stats['mean']:.6e}")
    print(f"  Std: {mask_stats['std']:.6e}")

    fsky = np.sum(mask > 0) / len(mask)
    print(f"  Fraction of sky covered: {fsky:.4f}")

    diagnostics = {
        'n_data': n_data,
        'n_randoms': n_randoms,
        'n_randoms_filtered': n_randoms_filtered,
        'alpha_r': alpha_r,
        'mask_stats': mask_stats,
        'fsky': fsky
    }

    return mask, diagnostics


def get_all_random_files(directory):
    """Get all LRG random catalog files from directory (both NGC and SGC)"""
    files = []
    for filename in os.listdir(directory):
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


def main():
    parser = argparse.ArgumentParser(
        description='Create local number density HEALPix mask from DESI randoms'
    )
    parser.add_argument('--catalog', type=str, required=True,
                        help='Path to galaxy catalog (CSV/FITS/TXT with RA, DEC)')
    parser.add_argument('--nside', type=int, default=8192,
                        help='HEALPix nside parameter (default: 8192)')
    parser.add_argument('--dec-limit', type=float, default=-20,
                        help='Declination limit in degrees (default: -20)')
    parser.add_argument('--output-dir', type=str,
                        default='/scratch/jiaqu/desi/catalogue/',
                        help='Output directory for mask files')
    parser.add_argument('--random-dir', type=str,
                        default='/project/rrg-rbond-ac/jiaqu/DESI/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/',
                        help='Directory containing random catalogs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Creating Local Number Density Mask (Equation 8)")
    print("=" * 60)
    print(f"\nGalaxy catalog: {args.catalog}")
    print(f"Random catalog directory: {args.random_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"nside: {args.nside}")
    print(f"dec_limit: {args.dec_limit}")
    print("=" * 60)

    print("\nAnalyzing available random catalog files...")

    try:
        random_files, ngc_files, sgc_files = analyze_random_files(args.random_dir)
    except FileNotFoundError:
        print(f"Directory not found: {args.random_dir}")
        print("Please check the path and permissions.")
        return 1

    if not random_files:
        print("No random catalog files found.")
        return 1

    # Create local nbar mask
    print(f"\nCreating local nbar mask with nside={args.nside}...")

    mask, diagnostics = create_local_nbar_mask(
        random_files,
        args.catalog,
        nside=args.nside,
        dec_limit=args.dec_limit
    )

    # Save the mask
    output_file = os.path.join(args.output_dir, "desi_survey_mask_healpix_local_nbar.fits")
    hp.write_map(output_file, mask, overwrite=True)
    print(f"\nLocal nbar mask saved to: {output_file}")

    # Create a plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        hp.mollview(mask, title="DESI Local Number Density Mask",
                    cmap='viridis', unit=r'$\alpha_r \times n_{rand}$')
        plot_file = os.path.join(args.output_dir, "desi_survey_mask_local_nbar_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Mask plot saved to: {plot_file}")
        plt.close()
    except Exception as e:
        print(f"Could not create plot: {e}")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N_data (galaxies):     {diagnostics['n_data']:,}")
    print(f"N_randoms (total):     {diagnostics['n_randoms']:,}")
    print(f"N_randoms (filtered):  {diagnostics['n_randoms_filtered']:,}")
    print(f"alpha_r ratio:         {diagnostics['alpha_r']:.6e}")
    print(f"Sky fraction:          {diagnostics['fsky']:.4f}")
    print(f"Output file:           {output_file}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
