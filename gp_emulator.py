import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from scipy.stats import qmc
import matplotlib.pyplot as plt
import pickle
import os
from classy_sz import Class
from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration

# ============================================================================
# Setup Colossus cosmology for mass conversions
# ============================================================================
params_cosmo = {'flat': True, 'H0': 67.66, 'Om0': 0.3111, 'Ob0': 0.049, 
                'sigma8': 0.8102, 'ns': 0.9665}
cosmology.addCosmology('myCosmo', params_cosmo)
cosmology.setCosmology('myCosmo')

z_eff =  0.725

# ============================================================================
# CLASS-SZ parameters
# ============================================================================
cosmo_params = {
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'H0': 67.66,
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.047,
    'n_s': 0.9665,
}

common_params = {
    'z_min': 0.005,
    'z_max': 3.0,
    'M_min': 1.0e10,
    'M_max': 3.5e15,
    'mass function': 'T08M200c',
    'concentration parameter': 'B13',
    'redshift_epsabs': 1.0e-40,
    'redshift_epsrel': 0.0005,
    'mass_epsabs': 1.0e-40,
    'mass_epsrel': 0.0005,
    'ell_max': 1600.0,
    'ell_min': 2.0,
    'dell': 10,
    'non_linear': 'hmcode',
    'hm_consistency': 1,
}

ksz_params = {
    'output': 'tau_gal_1h,tau_gal_2h',
    "ell_min": 2,
    "ell_max": 20000,
    'dell': 0,
    'dlogell': 0.2,
    'M_min': 1.0e10,
    'M_max': 5e15,
    'gas profile': 'B16',
    'gas profile mode': 'custom',
    'use_xout_in_density_profile_from_enclosed_mass': 1,
    'n_z_m_to_xout': 30,
    'n_mass_m_to_xout': 30,
    'n_m_density_profile': 30,
    'n_z_density_profile': 30,
    'k_min_samp_fftw': 1e-3,
    'k_max_samp_fftw': 1e3,
    'N_samp_fftw': 1024,
    'hm_consistency': 1,
    'use_fft_for_profiles_transform': 1,
    'x_min_gas_density_fftw': 1e-6,
    'x_max_gas_density_fftw': 1e5,
}

def l_to_dl(lp):
    return lp * (lp + 1.) / 2. / np.pi

# Fixed gas parameters
A_ALPHA_FIXED =0.88 

# ============================================================================
# Load HOD chains and get bounds
# ============================================================================
chain_file = "/scratch/jiaqu/HOD/LRG/z1/v1.1/LRG_z1_all_chains.txt"
z0, z1, z2 = [0.5, 0.725, 0.950]

print(f"Reading chain file: {chain_file}")
data = np.loadtxt(chain_file)

n_chain_params = data.shape[1] - 3
samples_all = data[:, :n_chain_params]
weights = data[:, -3]
logl = data[:, -2]
logp = data[:, -1]

print(f"Chain shape: {data.shape}")
print(f"Number of chain parameters: {n_chain_params}")

# Extract only the HOD parameters we care about (skip alpha at index 3)
hod_indices = [0, 1, 2, 4, 5, 6]  # Skip index 3 (alpha)
samples = samples_all[:, hod_indices]

hod_param_names = [
    'log_Mmin',
    'log_Msat', 
    'sigma_logM',
    'kappa',
    'alpha_c',
    'alpha_s'
]

n_hod_params = len(hod_param_names)
print(f"Using {n_hod_params} HOD parameters (excluding alpha)")
print(f"HOD parameters: {hod_param_names}")

# ============================================================================
# Convert masses from virial to M200c
# ============================================================================
print("\nConverting masses from virial to M200c...")

samples_m200c = samples.copy()

for i in range(len(samples)):
    if i % 1000 == 0:
        print(f"  Converting sample {i}/{len(samples)}")
    
    log_Mmin_vir = samples[i, 0]
    log_Msat_vir = samples[i, 1]
    
    M_min_vir = 10**log_Mmin_vir
    M_sat_vir = 10**log_Msat_vir
    
    c_min_vir = concentration.concentration(M_min_vir, 'vir', z_eff, model='duffy08')
    c_sat_vir = concentration.concentration(M_sat_vir, 'vir', z_eff, model='duffy08')
    
    M_min_200c, _, _ = mass_defs.changeMassDefinition(M_min_vir, c_min_vir, z_eff, 'vir', '200c', profile='nfw')
    M_sat_200c, _, _ = mass_defs.changeMassDefinition(M_sat_vir, c_sat_vir, z_eff, 'vir', '200c', profile='nfw')
    
    samples_m200c[i, 0] = np.log10(M_min_200c)
    samples_m200c[i, 1] = np.log10(M_sat_200c)

samples = samples_m200c
print("Mass conversion complete!")

# ============================================================================
# Define parameter bounds
# ============================================================================

def get_parameter_bounds(samples, weights, percentiles=[5, 95]):
    """Extract parameter bounds from weighted samples"""
    bounds = []
    for i in range(samples.shape[1]):
        sorted_idx = np.argsort(samples[:, i])
        cumsum = np.cumsum(weights[sorted_idx])
        cumsum /= cumsum[-1]
        
        lower = samples[sorted_idx][np.searchsorted(cumsum, percentiles[0]/100), i]
        upper = samples[sorted_idx][np.searchsorted(cumsum, percentiles[1]/100), i]
        bounds.append([lower, upper])
    
    return np.array(bounds)

# HOD bounds from chains
hod_bounds = get_parameter_bounds(samples, weights)

print("\nHOD Parameter Bounds (5-95 percentile, M200c):")
print("=" * 60)
for name, (lower, upper) in zip(hod_param_names, hod_bounds):
    print(f"{name:15s}: [{lower:.6f}, {upper:.6f}]")

# Gas parameter bounds - REMOVED A_alpha
gas_param_names = [
    'log10_A_rho0',
    'xc_B16',
    'A_beta',
    'a_k2h'
]

# Define gas parameter bounds based on your MCMC priors
gas_bounds = np.array([
    [0.8, 5.2],      # log10_A_rho0
    [0.05, 1.05],      # xc_B16
    [0.8, 8.0],      # A_beta
    [0.0, 5.0]       # a_k2h
])

print("\nGas Parameter Bounds:")
print("=" * 60)
for name, (lower, upper) in zip(gas_param_names, gas_bounds):
    print(f"{name:15s}: [{lower:.6f}, {upper:.6f}]")
print(f"A_alpha (fixed): {A_ALPHA_FIXED}")

# Combine all bounds
all_param_names = hod_param_names + gas_param_names
all_bounds = np.vstack([hod_bounds, gas_bounds])
n_total_params = len(all_param_names)

print(f"\nTotal parameters: {n_total_params} ({n_hod_params} HOD + {len(gas_param_names)} gas)")
print(f"Fixed parameters: A_alpha = {A_ALPHA_FIXED}")

# ============================================================================
# Generate Latin Hypercube samples
# ============================================================================

n_training_samples = 600  # Reduced from 500 since we have 10D instead of 11D
n_test_samples = 60

print(f"\nGenerating {n_training_samples} training samples in {n_total_params}D space...")

sampler = qmc.LatinHypercube(d=n_total_params, seed=42)

lhs_samples_raw = sampler.random(n_training_samples)
training_samples = qmc.scale(lhs_samples_raw, all_bounds[:, 0], all_bounds[:, 1])

lhs_test_raw = sampler.random(n_test_samples)
test_samples = qmc.scale(lhs_test_raw, all_bounds[:, 0], all_bounds[:, 1])

print(f"Generated {n_training_samples} training samples")
print(f"Generated {n_test_samples} test samples")

# ============================================================================
# CLASS-SZ wrapper function for combined HOD + gas parameters
# ============================================================================

def run_class_sz_combined(combined_params, verbose=False):
    """
    Run CLASS-SZ with both HOD and gas parameters
    
    Parameters:
    combined_params: array of [HOD params (6), gas params (4)]
                    = [log_Mmin, log_Msat, sigma_logM, kappa, alpha_c, alpha_s,
                       log10_A_rho0, xc_B16, A_beta, a_k2h]
    
    Returns:
    dict with 'ell', 'cl_1h', 'cl_2h', 'cl_total'
    """
    # Split parameters
    hod_params = combined_params[:n_hod_params]
    gas_params = combined_params[n_hod_params:]
    
    log_Mmin, log_Msat, sigma_logM, kappa, alpha_c, alpha_s = hod_params
    log10_A_rho0, xc_B16, A_beta, a_k2h = gas_params
    
    # Convert log10_A_rho0 to A_rho0
    A_rho0 = 10**log10_A_rho0
    
    # Build HOD dictionary
    HOD = {
        'sigma_log10M_HOD': sigma_logM,
        'alpha_s_HOD': alpha_s,
        'M1_prime_HOD': 10**log_Msat,
        'M_min_HOD': 10**log_Mmin,
        'M0_HOD': 10**log_Mmin * kappa,
        'x_out_truncated_nfw_profile_satellite_galaxies': 1.0,
        'f_cen_HOD': alpha_c,
        'UNWISE_dndz_file': "/home/jiaqu/Thumbstack_DESI/HOD/dndz_bin2_interpolated_new.txt"
    }
    
    # Build gas parameters dictionary with FIXED A_alpha
    variable_ksz_params = {
        'A_rho0': A_rho0,
        'A_alpha': A_ALPHA_FIXED,  # FIXED
        'A_beta': A_beta,
        'alpha_m_rho0': 0.29,      # Fixed mass slopes
        'alpha_m_alpha': -0.03,
        'alpha_m_beta': 0.04,
        'alpha_z_rho0': -0.66,     # Fixed redshift slopes
        'alpha_z_alpha': 0.19,
        'alpha_z_beta': -0.025,
        'gamma_B16': -0.2,         # Fixed gamma
        'xc_B16': xc_B16,
    }
    
    if verbose:
        print("HOD parameters:")
        for key, val in HOD.items():
            if 'file' not in key:
                print(f"  {key}: {val}")
        print("Gas parameters:")
        for key, val in variable_ksz_params.items():
            print(f"  {key}: {val}")
    
    # Initialize and run CLASS-SZ
    M = Class()
    M.set(common_params)
    M.set(cosmo_params)
    M.set(ksz_params)
    M.set(HOD)
    M.set(variable_ksz_params)
    M.set({'use_fft_for_profiles_transform': 1, 'ndim_redshifts': 30})
    M.compute_class_szfast()
    
    # Extract results
    l = np.asarray(M.cl_sz()['ell'])
    cl_eg_1h = np.asarray(M.cl_eg()['1h']) / l_to_dl(l)
    cl_eg_2h = np.asarray(M.cl_eg()['2h']) / l_to_dl(l)
    
    # Apply a_k2h scaling to 2-halo term
    cl_eg_total = cl_eg_1h + a_k2h * cl_eg_2h
    
    # Clean up
    M.struct_cleanup()
    M.empty()
    
    return {
        'ell': l,
        'cl_1h': cl_eg_1h,
        'cl_2h': cl_eg_2h,
        'cl_total': cl_eg_total
    }

# ============================================================================
# Run CLASS-SZ for all training samples
# ============================================================================

print("\n" + "="*70)
print("RUNNING CLASS-SZ FOR TRAINING SAMPLES")
print("="*70)

training_predictions = []
training_ell = None

import time
start_time = time.time()

for i, params in enumerate(training_samples):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            remaining = avg_time * (n_training_samples - i)
            print(f"\nProgress: {i}/{n_training_samples} ({100*i/n_training_samples:.1f}%)")
            print(f"  Elapsed: {elapsed/60:.1f} min, Remaining: {remaining/60:.1f} min")
        else:
            print(f"\nStarting training sample {i+1}/{n_training_samples}")
    
    if i == 0:
        print(f"Sample 0 parameters:")
        for name, val in zip(all_param_names, params):
            print(f"  {name}: {val:.6f}")
        print(f"  A_alpha (fixed): {A_ALPHA_FIXED}")
    
    result = run_class_sz_combined(params, verbose=(i==0))
    
    if training_ell is None:
        training_ell = result['ell']
        print(f"\nell array length: {len(training_ell)}")
        print(f"ell range: {training_ell[0]:.1f} - {training_ell[-1]:.1f}")
    
    training_predictions.append(result['cl_total'])

training_predictions = np.array(training_predictions)
total_time = time.time() - start_time

print(f"\nTraining complete!")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Average time per sample: {total_time/n_training_samples:.2f} seconds")
print(f"Training predictions shape: {training_predictions.shape}")

# ============================================================================
# Run CLASS-SZ for test samples
# ============================================================================

print("\n" + "="*70)
print("RUNNING CLASS-SZ FOR TEST SAMPLES")
print("="*70)

test_predictions = []

for i, params in enumerate(test_samples):
    if i % 10 == 0:
        print(f"Test sample {i+1}/{n_test_samples}")
    
    result = run_class_sz_combined(params)
    test_predictions.append(result['cl_total'])

test_predictions = np.array(test_predictions)
print(f"\nTest predictions shape: {test_predictions.shape}")

# ============================================================================
# Train GP Emulator
# ============================================================================

print("\n" + "="*70)
print("TRAINING GP EMULATOR")
print("="*70)

# Normalize inputs
params_mean = np.mean(training_samples, axis=0)
params_std = np.std(training_samples, axis=0)
training_normalized = (training_samples - params_mean) / params_std
test_normalized = (test_samples - params_mean) / params_std

print(f"Parameter normalization:")
for name, mean, std in zip(all_param_names, params_mean, params_std):
    print(f"  {name:15s}: mean={mean:.6f}, std={std:.6f}")

# Train separate GP for each ell
n_ells = len(training_ell)
gp_emulators = []

print(f"\nTraining {n_ells} GPs (one per ell bin)...")

for i_ell in range(n_ells):
    if i_ell % 10 == 0:
        print(f"  Training GP {i_ell+1}/{n_ells} (ell={training_ell[i_ell]:.1f})")
    
    # Extract predictions at this ell
    y_train = training_predictions[:, i_ell]
    
    # Normalize predictions
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    
    # Define kernel
    length_scales = np.ones(n_total_params)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=length_scales,
        length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    
    # Train GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        normalize_y=False,
        alpha=1e-10
    )
    gp.fit(training_normalized, y_train_norm)
    
    gp_emulators.append({
        'gp': gp,
        'y_mean': y_mean,
        'y_std': y_std,
        'ell': training_ell[i_ell]
    })

print(f"\nTrained {len(gp_emulators)} GP emulators")

# ============================================================================
# Validate GP Emulator
# ============================================================================

print("\n" + "="*70)
print("VALIDATING GP EMULATOR")
print("="*70)

def predict_with_emulator(combined_params):
    """Predict C_ell using trained GP emulator"""
    params_norm = (combined_params - params_mean) / params_std
    
    predictions = []
    uncertainties = []
    
    for emulator in gp_emulators:
        pred_norm, std_norm = emulator['gp'].predict([params_norm], return_std=True)
        pred = pred_norm[0] * emulator['y_std'] + emulator['y_mean']
        std = std_norm[0] * emulator['y_std']
        
        predictions.append(pred)
        uncertainties.append(std)
    
    return np.array(predictions), np.array(uncertainties)

# Test on validation set
test_pred_gp = []
test_pred_std = []

for i, params in enumerate(test_samples):
    if i % 10 == 0:
        print(f"Validating sample {i+1}/{n_test_samples}")
    pred, std = predict_with_emulator(params)
    test_pred_gp.append(pred)
    test_pred_std.append(std)

test_pred_gp = np.array(test_pred_gp)
test_pred_std = np.array(test_pred_std)

# Compute errors
errors = np.abs(test_predictions - test_pred_gp) / np.abs(test_predictions)

print(f"\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)
print(f"Mean relative error:       {np.mean(errors):.4%}")
print(f"Median relative error:     {np.median(errors):.4%}")
print(f"95th percentile error:     {np.percentile(errors, 95):.4%}")
print(f"Max relative error:        {np.max(errors):.4%}")

print(f"\nError statistics by ell:")
ell_indices = [0, len(training_ell)//4, len(training_ell)//2, 3*len(training_ell)//4, -1]
for idx in ell_indices:
    ell = training_ell[idx]
    err_median = np.median(errors[:, idx])
    err_95 = np.percentile(errors[:, idx], 95)
    print(f"  ell={ell:6.1f}: median={err_median:.4%}, 95th={err_95:.4%}")

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating validation plots...")

fig = plt.figure(figsize=(18, 12))

# Plot 1: Example predictions
ax1 = plt.subplot(3, 3, 1)
idx_plot = 0
ax1.loglog(training_ell, test_predictions[idx_plot], 'o-', label='CLASS-SZ', alpha=0.7, markersize=3)
ax1.loglog(training_ell, test_pred_gp[idx_plot], 's-', label='GP Emulator', alpha=0.7, markersize=3)
ax1.fill_between(training_ell,
                 test_pred_gp[idx_plot] - test_pred_std[idx_plot],
                 test_pred_gp[idx_plot] + test_pred_std[idx_plot],
                 alpha=0.3, label='GP uncertainty')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$C_\ell^{eg}$')
ax1.set_title(f'Example Prediction (Test {idx_plot})')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Another example
ax2 = plt.subplot(3, 3, 2)
idx_plot = n_test_samples//2
ax2.loglog(training_ell, test_predictions[idx_plot], 'o-', label='CLASS-SZ', alpha=0.7, markersize=3)
ax2.loglog(training_ell, test_pred_gp[idx_plot], 's-', label='GP Emulator', alpha=0.7, markersize=3)
ax2.fill_between(training_ell,
                 test_pred_gp[idx_plot] - test_pred_std[idx_plot],
                 test_pred_gp[idx_plot] + test_pred_std[idx_plot],
                 alpha=0.3)
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$C_\ell^{eg}$')
ax2.set_title(f'Example Prediction (Test {idx_plot})')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Relative errors vs ell
ax3 = plt.subplot(3, 3, 3)
median_errors = np.median(errors, axis=0)
p16 = np.percentile(errors, 16, axis=0)
p84 = np.percentile(errors, 84, axis=0)
ax3.semilogx(training_ell, median_errors * 100, 'k-', linewidth=2, label='Median')
ax3.fill_between(training_ell, p16*100, p84*100, alpha=0.3, label='16-84%ile')
ax3.axhline(1, color='r', linestyle='--', alpha=0.5, label='1% error')
ax3.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% error')
ax3.set_xlabel(r'$\ell$')
ax3.set_ylabel('Relative Error (%)')
ax3.set_title('Emulator Accuracy vs Scale')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Error histogram
ax4 = plt.subplot(3, 3, 4)
ax4.hist(errors.flatten() * 100, bins=50, alpha=0.7, edgecolor='black')
ax4.axvline(np.median(errors)*100, color='r', linestyle='--', linewidth=2,
            label=f'Median: {np.median(errors):.4%}')
ax4.axvline(np.percentile(errors, 95)*100, color='orange', linestyle='--', linewidth=2,
            label=f'95th: {np.percentile(errors, 95):.4%}')
ax4.set_xlabel('Relative Error (%)')
ax4.set_ylabel('Count')
ax4.set_title('Distribution of Relative Errors')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, min(10, np.percentile(errors, 99)*100))

# Plot 5: Predicted vs True
ax5 = plt.subplot(3, 3, 5)
ell_indices = [0, len(training_ell)//3, 2*len(training_ell)//3, -1]
colors = ['blue', 'green', 'orange', 'red']
for idx, color in zip(ell_indices, colors):
    y_true = test_predictions[:, idx]
    y_pred = test_pred_gp[:, idx]
    ax5.scatter(y_true, y_pred, alpha=0.6, s=30, color=color,
                label=f'ℓ={training_ell[idx]:.0f}')
lims = [min(ax5.get_xlim()[0], ax5.get_ylim()[0]),
        max(ax5.get_xlim()[1], ax5.get_ylim()[1])]
ax5.plot(lims, lims, 'k--', lw=2, alpha=0.5)
ax5.set_xlabel('True $C_\\ell$')
ax5.set_ylabel('Predicted $C_\\ell$')
ax5.set_title('Predicted vs True')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Plot 6: Uncertainty calibration
ax6 = plt.subplot(3, 3, 6)
errors_flat = np.abs(test_predictions - test_pred_gp).flatten()
std_flat = test_pred_std.flatten()
ax6.scatter(std_flat, errors_flat, alpha=0.3, s=10)
ax6.plot([0, std_flat.max()], [0, std_flat.max()], 'r--', lw=2, label='Perfect')
ax6.set_xlabel('GP Uncertainty')
ax6.set_ylabel('Actual Error')
ax6.set_title('Uncertainty Calibration')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Plot 7: Multiple test predictions
ax7 = plt.subplot(3, 3, 7)
for idx in range(min(5, n_test_samples)):
    alpha_val = 0.3 + 0.1 * idx
    ax7.loglog(training_ell, test_predictions[idx], '-', alpha=alpha_val,
              color='blue', linewidth=1, label='CLASS-SZ' if idx == 0 else '')
    ax7.loglog(training_ell, test_pred_gp[idx], '--', alpha=alpha_val,
              color='red', linewidth=1, label='GP' if idx == 0 else '')
ax7.set_xlabel(r'$\ell$')
ax7.set_ylabel(r'$C_\ell^{eg}$')
ax7.set_title('Multiple Test Predictions')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Error vs HOD parameter
ax8 = plt.subplot(3, 3, 8)
hod_param_idx = 0  # log_Mmin
hod_vals = test_samples[:, hod_param_idx]
mean_errors_per_sample = np.mean(errors, axis=1)
ax8.scatter(hod_vals, mean_errors_per_sample * 100, alpha=0.6, s=30)
ax8.set_xlabel(f'{hod_param_names[hod_param_idx]}')
ax8.set_ylabel('Mean Relative Error (%)')
ax8.set_title(f'Error vs {hod_param_names[hod_param_idx]}')
ax8.grid(True, alpha=0.3)

# Plot 9: Error vs Gas parameter
ax9 = plt.subplot(3, 3, 9)
gas_param_idx = 0  # log10_A_rho0
gas_vals = test_samples[:, n_hod_params + gas_param_idx]
ax9.scatter(gas_vals, mean_errors_per_sample * 100, alpha=0.6, s=30, color='orange')
ax9.set_xlabel(f'{gas_param_names[gas_param_idx]}')
ax9.set_ylabel('Mean Relative Error (%)')
ax9.set_title(f'Error vs {gas_param_names[gas_param_idx]}')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
output_plot = f'/scratch/jiaqu/HOD/gp_emulator_2d_validation_z{z_eff:.3f}.png'
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"Saved validation plot: {output_plot}")

# ============================================================================
# Save emulator
# ============================================================================

print("\n" + "="*70)
print("SAVING EMULATOR")
print("="*70)

emulator_data = {
    'gp_emulators': gp_emulators,
    'params_mean': params_mean,
    'params_std': params_std,
    'all_param_names': all_param_names,
    'hod_param_names': hod_param_names,
    'gas_param_names': gas_param_names,
    'n_hod_params': n_hod_params,
    'all_bounds': all_bounds,
    'hod_bounds': hod_bounds,
    'gas_bounds': gas_bounds,
    'ell': training_ell,
    'z_eff': z_eff,
    'z_range': [z0, z1, z2],
    'n_training_samples': n_training_samples,
    'A_alpha_fixed': A_ALPHA_FIXED,  # Store the fixed value
    'validation_errors': {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'p95': np.percentile(errors, 95),
        'max': np.max(errors)
    },
    'training_time_minutes': total_time / 60
}

output_file = f'/scratch/jiaqu/HOD/gp_emulator_2d_z{z_eff:.3f}_.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(emulator_data, f)

print(f"Saved emulator to: {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*70)
print("2D GP EMULATOR TRAINING COMPLETE!")
print("="*70)
print(f"Redshift range: z={z0:.3f} - {z1:.3f}")
print(f"Training samples: {n_training_samples}")
print(f"Test samples: {n_test_samples}")
print(f"Total parameters: {n_total_params} ({n_hod_params} HOD + {len(gas_param_names)} gas)")
print(f"Fixed parameters: A_alpha = {A_ALPHA_FIXED}")
print(f"Number of ell bins: {len(training_ell)}")
print(f"ell range: {training_ell[0]:.1f} - {training_ell[-1]:.1f}")
print(f"Training time: {total_time/60:.1f} minutes")
print(f"\nValidation Performance:")
print(f"  Median error: {np.median(errors):.4%}")
print(f"  95th percentile: {np.percentile(errors, 95):.4%}")
print(f"\nParameter ranges:")
print("HOD (6 parameters):")
for name, bound in zip(hod_param_names, hod_bounds):
    print(f"  {name:15s}: [{bound[0]:.6f}, {bound[1]:.6f}]")
print("Gas (4 parameters):")
for name, bound in zip(gas_param_names, gas_bounds):
    print(f"  {name:15s}: [{bound[0]:.6f}, {bound[1]:.6f}]")
print("="*70)