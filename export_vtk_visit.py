#!/usr/bin/env python
"""
Export DESI Y1 and Y3 LRG catalogs to VTK format for VisIt visualization.

Generates wedge diagram visualization showing galaxy positions colored by
LOS velocity, with Y1 (greyed) and Y3 overlay capability.

Usage:
    python export_vtk_visit.py --dataset Y1    # Export Y1 only
    python export_vtk_visit.py --dataset Y3    # Export Y3 only
    python export_vtk_visit.py --dataset both  # Export both (default)
"""

import argparse
import os
import numpy as np
import pandas as pd
from astropy.table import Table
from cosmoprimo.fiducial import DESI
from tqdm import tqdm


# ============================================================================
# Catalog paths
# ============================================================================

# Y1 paths
Y1_MAIN = '/project/rrg-rbond-ac/jiaqu/DESI/catalogs/Y1/LSS/iron/LSScats/v1.2/unblinded/'
Y1_POST_REC = 'desipipe/baseline_2pt/recon_recsym/'

# Y3 (DA2) paths
Y3_MAIN = '/project/rrg-rbond-ac/jiaqu/DESI/catalogs/DA2/'
Y3_PRE_REC = 'LSS/loa-v1/LSScats/v1.1/nonKP/'
Y3_POST_REC = 'analysis/loa-v1/LSScats/v1.1/BAO/unblinded/desipipe/2pt/recon_sm15_IFFT_recsym/'

# Output directory
OUTPUT_DIR = '/home/jiaqu/Thumbstack_DESI/output/vtk/'


def load_catalog(filepath):
    """Load a FITS catalog and convert to pandas DataFrame."""
    dat = Table.read(filepath, format='fits')
    return dat.to_pandas()


def compute_los_velocity(pre_rec_df, post_rec_df, cosmo, z_eff=0.780):
    """
    Compute LOS velocity from pre- and post-reconstruction catalogs.

    Parameters
    ----------
    pre_rec_df : pd.DataFrame
        Pre-reconstruction catalog with 'Z' column
    post_rec_df : pd.DataFrame
        Post-reconstruction catalog with 'Z' column
    cosmo : cosmoprimo cosmology object
        DESI fiducial cosmology
    z_eff : float
        Effective redshift for growth rate calculation

    Returns
    -------
    np.ndarray
        LOS velocity in km/s
    """
    f_eff = cosmo.growth_rate(z_eff)

    # Comoving distances in Mpc/h
    chi_pre = cosmo.comoving_radial_distance(pre_rec_df['Z'])
    chi_post = cosmo.comoving_radial_distance(post_rec_df['Z'])

    # LOS displacement in Mpc (note negative sign convention)
    disp_LOS = -(chi_post - chi_pre) / cosmo.h
    disp_LOS /= (1 + f_eff)  # RSD correction

    # Convert displacement to velocity
    f = cosmo.growth_rate(post_rec_df['Z'])
    H_z = cosmo.hubble_function(post_rec_df['Z'])
    a = 1 / (1 + post_rec_df['Z'])

    vel_LOS = a * H_z * f * disp_LOS  # km/s

    return np.array(vel_LOS)


def compute_cartesian_coords(df, cosmo):
    """
    Convert RA/DEC/Z to Cartesian coordinates.

    Parameters
    ----------
    df : pd.DataFrame
        Catalog with 'RA', 'DEC', 'Z' columns
    cosmo : cosmoprimo cosmology object
        DESI fiducial cosmology

    Returns
    -------
    tuple of np.ndarray
        (coordX, coordY, coordZ) in Mpc
    """
    chi = cosmo.comoving_radial_distance(df['Z']) / cosmo.h  # Mpc
    ra_rad = np.radians(df['RA'])
    dec_rad = np.radians(df['DEC'])

    coordX = chi * np.cos(dec_rad) * np.cos(ra_rad)
    coordY = chi * np.cos(dec_rad) * np.sin(ra_rad)
    coordZ = chi * np.sin(dec_rad)

    return np.array(coordX), np.array(coordY), np.array(coordZ)


def write_vtk(filepath, coordX, coordY, coordZ, vel_LOS, redshift, gc_flag, title="DESI LRG Catalog"):
    """
    Write catalog to VTK format.

    Parameters
    ----------
    filepath : str
        Output VTK file path
    coordX, coordY, coordZ : np.ndarray
        Cartesian coordinates in Mpc
    vel_LOS : np.ndarray
        LOS velocity in km/s
    redshift : np.ndarray
        Redshift values
    gc_flag : np.ndarray
        Galactic cap flag (1 for NGC, 0 for SGC)
    title : str
        VTK file title
    """
    nObj = len(coordX)

    print(f"  Writing VTK file: {filepath}")
    print(f"  Number of objects: {nObj:,}")

    with open(filepath, 'w') as f:
        # Header
        f.write('# vtk DataFile Version 3.0\n')
        f.write(f'{title}\n')
        f.write('ASCII\n')
        f.write('\n')

        # Points
        f.write('DATASET POLYDATA\n')
        f.write(f'POINTS {nObj} DOUBLE\n')
        for i in tqdm(range(nObj), desc="  Writing coordinates", leave=False):
            f.write(f'{coordX[i]:.10f} {coordY[i]:.10f} {coordZ[i]:.10f}\n')
        f.write('\n')

        # Point data header
        f.write(f'POINT_DATA {nObj}\n')
        f.write('\n')

        # Vel_LOS scalar
        f.write('SCALARS Vel_LOS DOUBLE\n')
        f.write('LOOKUP_TABLE default\n')
        for i in tqdm(range(nObj), desc="  Writing velocities", leave=False):
            f.write(f'{vel_LOS[i]:.10e}\n')
        f.write('\n')

        # Redshift scalar
        f.write('SCALARS Redshift DOUBLE\n')
        f.write('LOOKUP_TABLE default\n')
        for i in tqdm(range(nObj), desc="  Writing redshifts", leave=False):
            f.write(f'{redshift[i]:.10e}\n')
        f.write('\n')

        # GC flag scalar
        f.write('SCALARS GC DOUBLE\n')
        f.write('LOOKUP_TABLE default\n')
        for i in tqdm(range(nObj), desc="  Writing GC flags", leave=False):
            f.write(f'{gc_flag[i]:.1f}\n')
        f.write('\n')

    print(f"  VTK file written successfully.")


def process_catalog(pre_rec_path, post_rec_path, gc_name, dataset_name, cosmo, output_dir, vel_clip=(-2000, 2000)):
    """
    Process a single galactic cap catalog and export to VTK.

    Parameters
    ----------
    pre_rec_path : str
        Path to pre-reconstruction FITS file
    post_rec_path : str
        Path to post-reconstruction FITS file
    gc_name : str
        Galactic cap name ('NGC' or 'SGC')
    dataset_name : str
        Dataset name ('Y1' or 'Y3')
    cosmo : cosmoprimo cosmology object
        DESI fiducial cosmology
    output_dir : str
        Output directory for VTK files
    vel_clip : tuple
        (min, max) velocity clipping bounds in km/s

    Returns
    -------
    dict
        Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} {gc_name}")
    print(f"{'='*60}")

    # Load catalogs
    print(f"Loading pre-reconstruction catalog: {pre_rec_path}")
    pre_rec = load_catalog(pre_rec_path)
    print(f"  Loaded {len(pre_rec):,} objects")

    print(f"Loading post-reconstruction catalog: {post_rec_path}")
    post_rec = load_catalog(post_rec_path)
    print(f"  Loaded {len(post_rec):,} objects")

    # Verify catalog alignment
    if len(pre_rec) != len(post_rec):
        raise ValueError(f"Catalog size mismatch: pre_rec={len(pre_rec)}, post_rec={len(post_rec)}")

    # Compute LOS velocity
    print("Computing LOS velocities...")
    vel_LOS = compute_los_velocity(pre_rec, post_rec, cosmo)

    # Clip velocities
    vel_LOS_clipped = np.clip(vel_LOS, vel_clip[0], vel_clip[1])
    n_clipped = np.sum((vel_LOS < vel_clip[0]) | (vel_LOS > vel_clip[1]))
    if n_clipped > 0:
        print(f"  Clipped {n_clipped:,} velocities to [{vel_clip[0]}, {vel_clip[1]}] km/s")

    # Compute Cartesian coordinates
    print("Computing Cartesian coordinates...")
    coordX, coordY, coordZ = compute_cartesian_coords(pre_rec, cosmo)

    # Get redshift and GC flag
    redshift = np.array(pre_rec['Z'])
    gc_flag = np.ones(len(pre_rec)) if gc_name == 'NGC' else np.zeros(len(pre_rec))

    # Write VTK file
    output_path = os.path.join(output_dir, f'DESI_{dataset_name}_{gc_name}.vtk')
    write_vtk(
        output_path,
        coordX, coordY, coordZ,
        vel_LOS_clipped,
        redshift,
        gc_flag,
        title=f'DESI LRG {dataset_name} {gc_name} Catalog'
    )

    # Summary statistics
    stats = {
        'dataset': dataset_name,
        'gc': gc_name,
        'n_galaxies': len(pre_rec),
        'vel_mean': np.mean(vel_LOS),
        'vel_std': np.std(vel_LOS),
        'vel_min': np.min(vel_LOS),
        'vel_max': np.max(vel_LOS),
        'z_mean': np.mean(redshift),
        'z_min': np.min(redshift),
        'z_max': np.max(redshift),
    }

    print(f"\n  Summary Statistics:")
    print(f"    N galaxies:    {stats['n_galaxies']:,}")
    print(f"    Velocity mean: {stats['vel_mean']:.2f} km/s")
    print(f"    Velocity std:  {stats['vel_std']:.2f} km/s")
    print(f"    Velocity range:[{stats['vel_min']:.2f}, {stats['vel_max']:.2f}] km/s")
    print(f"    Redshift mean: {stats['z_mean']:.4f}")
    print(f"    Redshift range:[{stats['z_min']:.4f}, {stats['z_max']:.4f}]")

    return stats


def export_y1(cosmo, output_dir):
    """Export Y1 catalogs."""
    stats_list = []

    # Y1 NGC
    pre_rec_path = os.path.join(Y1_MAIN, 'LRG_NGC_clustering.dat.fits')
    post_rec_path = os.path.join(Y1_MAIN, Y1_POST_REC, 'LRG_NGC_clustering.dat.fits')
    stats = process_catalog(pre_rec_path, post_rec_path, 'NGC', 'Y1', cosmo, output_dir)
    stats_list.append(stats)

    # Y1 SGC
    pre_rec_path = os.path.join(Y1_MAIN, 'LRG_SGC_clustering.dat.fits')
    post_rec_path = os.path.join(Y1_MAIN, Y1_POST_REC, 'LRG_SGC_clustering.dat.fits')
    stats = process_catalog(pre_rec_path, post_rec_path, 'SGC', 'Y1', cosmo, output_dir)
    stats_list.append(stats)

    return stats_list


def export_y3(cosmo, output_dir):
    """Export Y3 catalogs."""
    stats_list = []

    # Y3 NGC
    pre_rec_path = os.path.join(Y3_MAIN, Y3_PRE_REC, 'LRG_NGC_clustering.dat.fits')
    post_rec_path = os.path.join(Y3_MAIN, Y3_POST_REC, 'LRG_NGC_clustering.dat.fits')
    stats = process_catalog(pre_rec_path, post_rec_path, 'NGC', 'Y3', cosmo, output_dir)
    stats_list.append(stats)

    # Y3 SGC
    pre_rec_path = os.path.join(Y3_MAIN, Y3_PRE_REC, 'LRG_SGC_clustering.dat.fits')
    post_rec_path = os.path.join(Y3_MAIN, Y3_POST_REC, 'LRG_SGC_clustering.dat.fits')
    stats = process_catalog(pre_rec_path, post_rec_path, 'SGC', 'Y3', cosmo, output_dir)
    stats_list.append(stats)

    return stats_list


def main():
    parser = argparse.ArgumentParser(
        description='Export DESI LRG catalogs to VTK format for VisIt visualization.'
    )
    parser.add_argument(
        '--dataset',
        choices=['Y1', 'Y3', 'both'],
        default='both',
        help='Which dataset to export (default: both)'
    )
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory for VTK files (default: {OUTPUT_DIR})'
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize cosmology
    print("Initializing DESI fiducial cosmology...")
    cosmo = DESI()

    all_stats = []

    # Export requested datasets
    if args.dataset in ['Y1', 'both']:
        print("\n" + "="*60)
        print("Exporting Y1 catalogs")
        print("="*60)
        stats = export_y1(cosmo, args.output_dir)
        all_stats.extend(stats)

    if args.dataset in ['Y3', 'both']:
        print("\n" + "="*60)
        print("Exporting Y3 catalogs")
        print("="*60)
        stats = export_y3(cosmo, args.output_dir)
        all_stats.extend(stats)

    # Final summary
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nGenerated files:")
    for stat in all_stats:
        filename = f"DESI_{stat['dataset']}_{stat['gc']}.vtk"
        print(f"  {filename}: {stat['n_galaxies']:,} galaxies")

    total_galaxies = sum(s['n_galaxies'] for s in all_stats)
    print(f"\nTotal galaxies exported: {total_galaxies:,}")


if __name__ == '__main__':
    main()
