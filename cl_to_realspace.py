#!/usr/bin/env python
"""
cl_to_realspace.py

Convert harmonic-space C_ell to real-space stacked profile using the CAP window function.

Implements the Fourier transform for NaMaster-binned bandpowers:
    ΔT(θ_d) = [1/(r σ_rec)] × Σ_b Δℓ_b [(2ℓ_b+1)/(4π)] W_ℓ(θ_d) C̃_b × A_disk

where:
    - Δℓ_b is the bin width (NaMaster windows are normalized to average, not sum)
    - 1/π accounts for the FT convention difference between diskring_window and true FT
    - W_ℓ(θ_d) = [4 J_1(ℓθ_d) - 2√2 J_1(ℓ√2 θ_d)] / (ℓθ_d) is the harmonic window
    - A_disk = πθ_d² is the disk area (thumbstack convention)

Usage:
    python cl_to_realspace.py [--plot] [--fit] [--output OUTPUT_FILE]

"""

import numpy as np
from scipy.special import j1  # Bessel function J_1
import argparse

# =============================================================================
# Unit conversion constant
# =============================================================================

# Unit conversion: sr to arcmin^2
SR_TO_ARCMIN2 = (180.0 * 60.0 / np.pi) ** 2


# =============================================================================
# Window function for diskring filter
# =============================================================================

def diskring_window(ell, theta_d_rad):
    """
    Compute the harmonic-space window function W_ℓ(θ_d) for the diskring filter.

    The diskring is a compensated aperture photometry filter:
    - Disk of radius θ_d with weight +1/(π θ_d²)
    - Ring from θ_d to √2 θ_d with weight -1/(π θ_d²)  [equal area]

    The Fourier transform gives:
        W_ℓ(θ_d) = [4 J_1(ℓθ_d) - 2√2 J_1(ℓ√2 θ_d)] / (ℓθ_d)

    Parameters
    ----------
    ell : array_like
        Multipole moments
    theta_d_rad : float
        Aperture radius in radians

    Returns
    -------
    W_ell : array_like
        Window function values at each ell
    """
    ell = np.atleast_1d(ell).astype(float)
    x = ell * theta_d_rad

    # Handle ell=0 case (limit as x->0 gives W_ℓ -> 0)
    W_ell = np.zeros_like(x)
    nonzero = x > 0

    x_nz = x[nonzero]
    sqrt2 = np.sqrt(2.0)

    # W_ℓ = [4 J_1(x) - 2√2 J_1(√2 x)] / x
    W_ell[nonzero] = (4.0 * j1(x_nz) - 2.0 * sqrt2 * j1(sqrt2 * x_nz)) / x_nz

    return W_ell


def cl_to_realspace_profile(ell, cl, theta_d_arcmin, area_normalized=False,
                            r=None, sigma_rec=None, delta_ell=None):
    """
    Convert C_ℓ to real-space stacked profile at given aperture radii.

    Implements the Fourier transform of NaMaster-binned C_ℓ to real space:
        ΔT(θ_d) = [3/(π r σ_rec)] Σ_b Δℓ_b [(2ℓ_b+1)/(4π)] W_ℓ(θ_d) C̃_b × A_disk

    Factors:
    - 3/(r σ_rec): converts from harmonic (factor 1/3 r σ_true σ_rec) to stacking
    - 1/π: FT convention correction (diskring_window differs from true FT by A_disk)
    - Δℓ_b: bin width correction (NaMaster windows average, not sum)

    Parameters
    ----------
    ell : array_like
        Multipole moments (bin centers for binned data)
    cl : array_like
        Power spectrum C_ℓ (not D_ℓ!)
    theta_d_arcmin : array_like
        Aperture radii in arcmin
    area_normalized : bool
        If False (default), multiply by disk area to match thumbstack convention.
        If True, return area-normalized CAP values.
    r : float, optional
        Velocity correlation coefficient (default: 0.65)
    sigma_rec : float, optional
        Reconstructed velocity dispersion in v/c units (default: 233/3e5)
    delta_ell : array_like, optional
        Bin widths Δℓ for each ell bin. If None, computed from ell spacing.
        Required for NaMaster-binned data where windows are normalized to 1.

    Returns
    -------
    delta_T : array_like
        Real-space stacked profile ΔT(θ_d) in μK
    """
    ell = np.atleast_1d(ell)
    cl = np.atleast_1d(cl)
    theta_d_arcmin = np.atleast_1d(theta_d_arcmin)

    # Default values for velocity correlation and dispersion
    if r is None:
        r = 0.65  # Hadzhiyska et al. 2024
    if sigma_rec is None:
        sigma_rec = 233 / 3e5  # v/c units

    # Compute bin widths if not provided
    if delta_ell is None:
        delta_ell = np.zeros_like(ell)
        if len(ell) > 1:
            delta_ell[:-1] = np.diff(ell)
            delta_ell[-1] = delta_ell[-2]  # last bin same width as previous
        else:
            delta_ell[:] = 1.0  # single bin, assume width 1

    delta_ell = np.atleast_1d(delta_ell)

    # Prefactor to convert harmonic C_ell to stacking convention
    # -  1/(r σ_rec): harmonic has factor  r σ_true σ_rec; invert r σ_rec part
    # - 1/π: FT convention correction (diskring_window normalized differently)
    prefactor = 1.0 / (r * sigma_rec)

    # Convert arcmin to radians
    theta_d_rad = theta_d_arcmin * np.pi / (180.0 * 60.0)

    # Compute (2ℓ+1)/(4π) factor
    ell_weight = (2.0 * ell + 1.0) / (4.0 * np.pi)

    delta_T = np.zeros(len(theta_d_rad))

    for i, theta_rad in enumerate(theta_d_rad):
        W_ell = diskring_window(ell, theta_rad)
        # Sum over bins: ΔT = prefactor × Σ_b Δℓ_b [(2ℓ+1)/(4π)] W_ℓ C_ℓ
        delta_T[i] = prefactor * np.sum(delta_ell * ell_weight * W_ell * cl)

        # Multiply by disk area if not area-normalized i.e weights are +-1
        if not area_normalized:
            A_disk_sr = np.pi * theta_rad**2  # disk area in sr
            delta_T[i] *= A_disk_sr

    return delta_T


# =============================================================================
# Data loading functions
# =============================================================================

def load_theory_cl(theory_file, ell_file=None):
    """
    Load best-fit theory C_ℓ.

    Parameters
    ----------
    theory_file : str
        Path to theory C_ℓ file (single column, assumed to be C_ℓ not D_ℓ)
    ell_file : str, optional
        Path to file with ell values. If None, assumes theory_file has ell in first column.

    Returns
    -------
    ell : array
        Multipole moments
    cl : array
        Power spectrum C_ℓ
    """
    if ell_file is not None:
        # Load ell from separate file
        data = np.loadtxt(ell_file, unpack=True)
        ell = data[0]
        cl = np.loadtxt(theory_file)
    else:
        # Try loading as single column (just C_ℓ values)
        try:
            cl = np.loadtxt(theory_file)
            # Need ell values - raise error
            raise ValueError("theory_file has no ell values. Provide ell_file or use 2-column format.")
        except:
            # Try 2-column format
            data = np.loadtxt(theory_file, unpack=True)
            if len(data) >= 2:
                ell = data[0]
                cl = data[1]
            else:
                raise ValueError("Cannot parse theory file format")

    return ell, cl


def load_stack(catalog_fname, factor=SR_TO_ARCMIN2, random=False,
               base_path="/home/jiaqu/Thumbstack_DESI/output/output"):
    """
    Load the measurement and covariance data for a given catalog directory.

    Parameters
    ----------
    catalog_fname : str
        The folder name (e.g., "zall_mask_no_src_with_cluster")
    factor : float
        Scaling factor (sr to arcmin^2 conversion)
    random : bool
        If True, load velocity-shuffled null test
    base_path : str
        Base path to output directories

    Returns
    -------
    x : array
        Aperture radii in arcmin
    y : array
        Stacked profile (scaled)
    yerr : array
        Uncertainties from covariance diagonal
    cov : array
        Full covariance matrix (scaled)
    """
    if random:
        measured_file = f"{base_path}/{catalog_fname}/diskring_ksz_uniformweight_vshufflemean.txt"
        cov_file = f"{base_path}/{catalog_fname}/cov_diskring_ksz_uniformweight_vshuffle.txt"
    else:
        measured_file = f"{base_path}/{catalog_fname}/diskring_ksz_uniformweight_measured.txt"
        cov_file = f"{base_path}/{catalog_fname}/cov_diskring_ksz_uniformweight_bootstrap.txt"

    measurements = np.loadtxt(measured_file, unpack=True)
    x = measurements[0]
    y = factor * measurements[1]

    cov = np.loadtxt(cov_file)
    cov = factor**2 * cov
    yerr = np.sqrt(np.diag(cov))

    return x, y, yerr, cov


# =============================================================================
# Fitting functions
# =============================================================================

def monte_carlo_error_bands(ell, cl_mean, cl_cov, theta_d_arcmin,
                           area_normalized=False, r=None, sigma_rec=None,
                           delta_ell=None, n_samples=1000, random_seed=42):
    """
    Generate 1σ and 2σ error bands for real-space profile via Monte Carlo sampling.

    Samples from Cl distribution and transforms each to real space, then computes
    percentiles to define error bands.

    Parameters
    ----------
    ell : array_like
        Multipole moments (bin centers)
    cl_mean : array_like
        Mean measured C_ℓ values
    cl_cov : array_like
        Covariance matrix of C_ℓ (N_ell × N_ell)
    theta_d_arcmin : array_like
        Aperture radii in arcmin
    area_normalized : bool
        If False (default), multiply by disk area to match thumbstack convention
    r : float, optional
        Velocity correlation coefficient (default: 0.65)
    sigma_rec : float, optional
        Reconstructed velocity dispersion in v/c units (default: 233/3e5)
    delta_ell : array_like, optional
        Bin widths Δℓ for each ell bin
    n_samples : int
        Number of Monte Carlo samples (default: 1000)
    random_seed : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    delta_T_mean : array_like
        Mean real-space profile
    delta_T_std : array_like
        Standard deviation at each theta
    delta_T_1sigma_low : array_like
        16th percentile (1σ lower)
    delta_T_1sigma_high : array_like
        84th percentile (1σ upper)
    delta_T_2sigma_low : array_like
        2.5th percentile (2σ lower)
    delta_T_2sigma_high : array_like
        97.5th percentile (2σ upper)
    """
    np.random.seed(random_seed)

    ell = np.atleast_1d(ell)
    cl_mean = np.atleast_1d(cl_mean)
    theta_d_arcmin = np.atleast_1d(theta_d_arcmin)

    # Generate samples from multivariate normal: Cl_samples ~ N(cl_mean, cl_cov)
    cl_samples = np.random.multivariate_normal(cl_mean, cl_cov, size=n_samples)

    # Transform each sample to real space
    delta_T_samples = np.zeros((n_samples, len(theta_d_arcmin)))

    for i in range(n_samples):
        delta_T_samples[i, :] = cl_to_realspace_profile(
            ell, cl_samples[i, :], theta_d_arcmin,
            area_normalized=area_normalized,
            r=r, sigma_rec=sigma_rec,
            delta_ell=delta_ell
        )

    # Compute statistics
    delta_T_mean = np.mean(delta_T_samples, axis=0)
    delta_T_std = np.std(delta_T_samples, axis=0)
    delta_T_1sigma_low = np.percentile(delta_T_samples, 16, axis=0)
    delta_T_1sigma_high = np.percentile(delta_T_samples, 84, axis=0)
    delta_T_2sigma_low = np.percentile(delta_T_samples, 2.5, axis=0)
    delta_T_2sigma_high = np.percentile(delta_T_samples, 97.5, axis=0)

    return (delta_T_mean, delta_T_std,
            delta_T_1sigma_low, delta_T_1sigma_high,
            delta_T_2sigma_low, delta_T_2sigma_high)


def fit_amplitude(model_at_data, data, cov):
    """
    Fit amplitude A to rescale model to match data.

    Minimizes χ² = (data - A×model)ᵀ C⁻¹ (data - A×model)

    Parameters
    ----------
    model_at_data : array
        Model profile values evaluated at data points
    data : array
        Measured profile
    cov : array
        Covariance matrix

    Returns
    -------
    A_best : float
        Best-fit amplitude
    A_err : float
        Uncertainty on amplitude
    chi2 : float
        Minimum chi-squared
    dof : int
        Degrees of freedom
    """
    # Inverse covariance
    cov_inv = np.linalg.inv(cov)

    # Analytic solution for linear fit: A = (m^T C^{-1} d) / (m^T C^{-1} m)
    m = model_at_data
    d = data

    mCm = m @ cov_inv @ m
    mCd = m @ cov_inv @ d

    A_best = mCd / mCm
    A_err = 1.0 / np.sqrt(mCm)

    # Chi-squared at best fit
    residual = d - A_best * m
    chi2 = residual @ cov_inv @ residual
    dof = len(data) - 1  # 1 free parameter

    return A_best, A_err, chi2, dof


# =============================================================================
# Main code
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert C_ell to real-space profile')
    parser.add_argument('--theory', type=str,
                        default='/home/jiaqu/ksz_mcmc/theory_bf/theory_windowed_zbin_all.txt',
                        help='Path to theory C_ell file')
    parser.add_argument('--ell-file', type=str,
                        default='/home/jiaqu/Thumbstack_DESI/output/z1_new_mask/stage_spectra/cl_master.txt',
                        help='Path to file with ell values (first column)')
    parser.add_argument('--catalog', type=str, default='zall_mask_no_src_with_cluster',
                        help='Catalog folder name for measured data')
    parser.add_argument('--theta-min', type=float, default=1.0,
                        help='Minimum aperture radius (arcmin)')
    parser.add_argument('--theta-max', type=float, default=6.0,
                        help='Maximum aperture radius (arcmin)')
    parser.add_argument('--n-theta', type=int, default=100,
                        help='Number of theta points for model')
    parser.add_argument('--fit', action='store_true',
                        help='Fit amplitude A to match data')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plot')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for real-space profile')
    parser.add_argument('--area-normalized', action='store_true',
                        help='Use area-normalized CAP (no disk area multiplication)')
    parser.add_argument('--r', type=float, default=0.65,
                        help='Velocity correlation coefficient (default: 0.65)')
    parser.add_argument('--sigma-rec', type=float, default=233/3e5,
                        help='Reconstructed velocity dispersion in v/c (default: 233/3e5)')
    parser.add_argument('--measured-cl', type=str, default=None,
                        help='Path to measured Cl file (2-column format: ell, Cl)')
    parser.add_argument('--measured-cov', type=str, default=None,
                        help='Path to covariance matrix .npy file for measured Cl')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of Monte Carlo samples for error propagation (default: 1000)')

    args = parser.parse_args()

    # Load theory C_ell
    print(f"Loading theory from: {args.theory}")
    print(f"Loading ell from: {args.ell_file}")

    # Load ell values from measurement file
    ell_data = np.loadtxt(args.ell_file, unpack=True)
    ell = ell_data[0]

    # Load theory (single column, windowed C_ell)
    cl_theory = np.loadtxt(args.theory)

    print(f"  ell range: {ell.min():.0f} - {ell.max():.0f}")
    print(f"  Number of ell bins: {len(ell)}")

    # Load measured Cl and covariance if provided
    has_measured = False
    if args.measured_cl is not None:
        print(f"\nLoading measured Cl from: {args.measured_cl}")
        measured_data = np.loadtxt(args.measured_cl, unpack=True)
        ell_measured = measured_data[0]
        cl_measured = measured_data[1]

        # Check ell values match
        if not np.allclose(ell, ell_measured):
            print("  Warning: ell values don't match between theory and measured files!")
            print(f"  Using ell from measured file.")
            ell = ell_measured

        if args.measured_cov is not None:
            print(f"Loading covariance from: {args.measured_cov}")
            cl_cov = np.load(args.measured_cov)
            print(f"  Covariance shape: {cl_cov.shape}")
            print(f"  Cl errors (sqrt(diag)): {np.sqrt(np.diag(cl_cov))}")
            has_measured = True
        else:
            print("  Warning: --measured-cl provided but no --measured-cov. Cannot compute errors.")

    # Generate theta array for model
    theta_model = np.linspace(args.theta_min, args.theta_max, args.n_theta)

    # Compute bin widths for NaMaster-binned data
    delta_ell = np.zeros_like(ell)
    if len(ell) > 1:
        delta_ell[:-1] = np.diff(ell)
        delta_ell[-1] = delta_ell[-2]
    else:
        delta_ell[:] = 1.0

    # Convert C_ell to real-space
    print("\nComputing real-space profile from theory...")
    print(f"  r = {args.r}, sigma_rec = {args.sigma_rec:.6e}")
    print(f"  prefactor = 1/(r*sigma_rec) = {1/( args.r * args.sigma_rec):.1f}")
    print(f"  mean(Δℓ) = {np.mean(delta_ell):.1f}")
    print(f"  area_normalized = {args.area_normalized}")

    delta_T_theory = cl_to_realspace_profile(ell, cl_theory, theta_model,
                                             area_normalized=args.area_normalized,
                                             r=args.r, sigma_rec=args.sigma_rec,
                                             delta_ell=delta_ell)

    # Convert to same units as data (μK × arcmin²)
    delta_T_theory_scaled = delta_T_theory * SR_TO_ARCMIN2

    print(f"  Theory range: {delta_T_theory_scaled.min():.4f} to {delta_T_theory_scaled.max():.4f} μK·arcmin²")

    # Compute measured FT profile with error bands if available
    measured_results = None
    if has_measured:
        print(f"\nComputing real-space profile from measured Cl with {args.n_samples} MC samples...")
        measured_results = monte_carlo_error_bands(
            ell, cl_measured, cl_cov, theta_model,
            area_normalized=args.area_normalized,
            r=args.r, sigma_rec=args.sigma_rec,
            delta_ell=delta_ell,
            n_samples=args.n_samples
        )
        # Unpack and scale to μK·arcmin²
        (delta_T_meas_mean, delta_T_meas_std,
         delta_T_meas_1sig_low, delta_T_meas_1sig_high,
         delta_T_meas_2sig_low, delta_T_meas_2sig_high) = measured_results

        # Scale all to μK·arcmin²
        delta_T_meas_mean *= SR_TO_ARCMIN2
        delta_T_meas_std *= SR_TO_ARCMIN2
        delta_T_meas_1sig_low *= SR_TO_ARCMIN2
        delta_T_meas_1sig_high *= SR_TO_ARCMIN2
        delta_T_meas_2sig_low *= SR_TO_ARCMIN2
        delta_T_meas_2sig_high *= SR_TO_ARCMIN2

        measured_results = (delta_T_meas_mean, delta_T_meas_std,
                           delta_T_meas_1sig_low, delta_T_meas_1sig_high,
                           delta_T_meas_2sig_low, delta_T_meas_2sig_high)

        print(f"  Measured FT range: {delta_T_meas_mean.min():.4f} to {delta_T_meas_mean.max():.4f} μK·arcmin²")
        print(f"  Mean uncertainty: {np.mean(delta_T_meas_std):.4f} μK·arcmin²")

    # Load measured data for comparison
    print(f"\nLoading measured data from catalog: {args.catalog}")
    try:
        x_data, y_data, yerr_data, cov_data = load_stack(args.catalog)
        print(f"  Aperture radii: {x_data}")
        print(f"  Measurements: {y_data}")
        has_data = True
    except Exception as e:
        print(f"  Warning: Could not load data: {e}")
        has_data = False

    # Fit amplitude if requested
    A_best = 1.0
    model_at_data = None
    if args.fit and has_data:
        print("\nFitting amplitude to stacking data...")
        # Compute theory model directly at data points (no interpolation needed)
        model_at_data = cl_to_realspace_profile(ell, cl_theory, x_data,
                                                 area_normalized=args.area_normalized,
                                                 r=args.r, sigma_rec=args.sigma_rec,
                                                 delta_ell=delta_ell) * SR_TO_ARCMIN2
        A_best, A_err, chi2, dof = fit_amplitude(model_at_data, y_data, cov_data)
        print(f"  Best-fit A = {A_best:.3f} ± {A_err:.3f}")
        print(f"  χ² = {chi2:.2f} (dof = {dof})")
        print(f"  χ²/dof = {chi2/dof:.2f}")
        print(f"  PTE = {1 - __import__('scipy').stats.chi2.cdf(chi2, dof):.4f}")

    # Save output at data aperture radii (same as real-space stacking)
    if args.output:
        # Use data aperture radii, or default if no data loaded
        if has_data:
            theta_out = x_data
            data_out = y_data
        else:
            # Default aperture radii from real-space stacking
            theta_out = np.array([1.0, 1.625, 2.25, 2.875, 3.5, 4.125, 4.75, 5.375, 6.0])
            data_out = np.full_like(theta_out, np.nan)

        # Compute FT from theory at output theta values
        FT_theory_at_theta = cl_to_realspace_profile(ell, cl_theory, theta_out,
                                                      area_normalized=args.area_normalized,
                                                      r=args.r, sigma_rec=args.sigma_rec,
                                                      delta_ell=delta_ell) * SR_TO_ARCMIN2
        FT_theory_fitted = A_best * FT_theory_at_theta

        # Build output columns
        columns = [theta_out]
        header_parts = ["theta_arcmin"]

        # Add measured FT if available
        if has_measured:
            # Compute measured FT at output theta values
            meas_results_at_theta = monte_carlo_error_bands(
                ell, cl_measured, cl_cov, theta_out,
                area_normalized=args.area_normalized,
                r=args.r, sigma_rec=args.sigma_rec,
                delta_ell=delta_ell,
                n_samples=args.n_samples
            )
            FT_meas_mean = meas_results_at_theta[0] * SR_TO_ARCMIN2
            FT_meas_err = meas_results_at_theta[1] * SR_TO_ARCMIN2

            columns.extend([FT_meas_mean, FT_meas_err])
            header_parts.extend(["FT_measured_uK_arcmin2", "FT_measured_err_uK_arcmin2"])

        # Add theory FT
        columns.extend([FT_theory_at_theta, FT_theory_fitted])
        header_parts.extend(["FT_theory_uK_arcmin2", f"FT_theory_fitted(A={A_best:.3f})_uK_arcmin2"])

        # Add stacking data
        columns.append(data_out)
        header_parts.append("data_stacking_uK_arcmin2")

        output_data = np.column_stack(columns)
        header = "  ".join(header_parts)
        np.savetxt(args.output, output_data, header=header, fmt='%.6e')
        print(f"\nSaved to: {args.output}")

    # Plot if requested
    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot measured FT with error bands if available
        if measured_results is not None:
            (delta_T_meas_mean, delta_T_meas_std,
             delta_T_meas_1sig_low, delta_T_meas_1sig_high,
             delta_T_meas_2sig_low, delta_T_meas_2sig_high) = measured_results

            # 2σ band
            ax.fill_between(theta_model, delta_T_meas_2sig_low, delta_T_meas_2sig_high,
                           alpha=0.2, color='blue', label='Measured FT (2σ)')
            # 1σ band
            ax.fill_between(theta_model, delta_T_meas_1sig_low, delta_T_meas_1sig_high,
                           alpha=0.4, color='blue', label='Measured FT (1σ)')
            # Mean
            ax.plot(theta_model, delta_T_meas_mean, 'b-', lw=2, label='Measured FT (mean)')

        # Plot theory FT
        ax.plot(theta_model, delta_T_theory_scaled, 'r-', lw=2, label='Theory FT (A=1)')

        if A_best != 1.0:
            ax.plot(theta_model, A_best * delta_T_theory_scaled, 'r--', lw=2,
                    label=f'Theory FT (A={A_best:.2f})')

        # Plot stacking data if available
        if has_data:
            ax.errorbar(x_data, y_data, yerr=yerr_data, fmt='ko', capsize=3,
                        markersize=5, label='Stacking data', zorder=10)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Aperture radius [arcmin]', fontsize=13)
        ax.set_ylabel(r'$\Delta T$ [$\mu$K $\cdot$ arcmin$^2$]', fontsize=13)
        ax.set_title('Harmonic → Real Space Conversion', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plot_file = args.output.replace('.txt', '.pdf') if args.output else 'cl_to_realspace.pdf'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {plot_file}")
        plt.show()

    return theta_model, delta_T_theory_scaled, A_best


if __name__ == '__main__':
    main()
