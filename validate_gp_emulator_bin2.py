"""
Validate GP emulators for bin1 and bin2: diagnose error vs ell.

Uses analytical leave-one-out (LOO) cross-validation on the stored
training data to compute actual prediction errors per ell bin, without
needing to re-run CLASS-SZ.

LOO formulas for GP regression:
    LOO_mean_i  = y_i - alpha_i / K_inv_ii
    LOO_var_i   = 1 / K_inv_ii
where K_inv = K^{-1}, alpha = K^{-1} y.
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import qmc

# ============================================================================
# Load emulators
# ============================================================================
emulator_paths = {
    'bin1': '/scratch/jiaqu/HOD/gp_emulator_2d_z0.500_0.725.pkl',
    'bin2': '/scratch/jiaqu/HOD/gp_emulator_2d_z0.725_.pkl',
}

emulators = {}
for label, path in emulator_paths.items():
    with open(path, 'rb') as f:
        emulators[label] = pickle.load(f)
    d = emulators[label]
    print(f"Loaded {label}: z_eff={d['z_eff']}, "
          f"n_train={d['n_training_samples']}, "
          f"n_ell={len(d['ell'])}, "
          f"stored median err={d['validation_errors']['median']:.4%}")


# ============================================================================
# Analytical LOO cross-validation
# ============================================================================
def compute_loo_errors(emulator_data):
    """
    For each ell bin's GP, compute leave-one-out errors analytically.

    Returns
    -------
    ell : array (n_ell,)
    loo_rel_errors : array (n_train, n_ell) — relative errors per sample per ell
    """
    gps = emulator_data['gp_emulators']
    n_ell = len(gps)
    n_train = emulator_data['n_training_samples']

    ell = np.array([g['ell'] for g in gps])
    loo_rel_errors = np.zeros((n_train, n_ell))

    for i_ell, gp_entry in enumerate(gps):
        gp = gp_entry['gp']
        y_mean = gp_entry['y_mean']
        y_std = gp_entry['y_std']

        # Normalized training targets
        y_norm = gp.y_train_.copy()

        # Compute K_inv via Cholesky factor stored in gp.L_
        L = gp.L_
        n = L.shape[0]
        I = np.eye(n)
        # L is lower-triangular: K = L L^T, so K_inv = L^{-T} L^{-1}
        L_inv = np.linalg.solve(L, I)
        K_inv = L_inv.T @ L_inv

        # alpha = K^{-1} y  (stored as gp.alpha_)
        alpha = gp.alpha_.ravel()

        K_inv_diag = np.diag(K_inv)

        # LOO prediction (normalized space)
        loo_mean_norm = y_norm - alpha / K_inv_diag

        # De-normalize to physical units
        loo_pred = loo_mean_norm * y_std + y_mean
        y_true = y_norm * y_std + y_mean

        # Relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.abs(loo_pred - y_true) / np.abs(y_true)
        # Cap extreme values where signal ~ 0
        rel_err = np.clip(rel_err, 0, 100)

        loo_rel_errors[:, i_ell] = rel_err

    return ell, loo_rel_errors


# ============================================================================
# Also generate LHS test samples and get GP uncertainty
# ============================================================================
def compute_test_uncertainty(emulator_data):
    """
    Regenerate LHS test samples (same seed=42) and evaluate GP uncertainty.
    Returns relative uncertainty = std / |mean| at each test point and ell.
    """
    n_train = emulator_data['n_training_samples']
    bounds = emulator_data['all_bounds']
    n_params = bounds.shape[0]

    # Regenerate the same LHS sequence
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    _ = sampler.random(n_train)  # skip training samples
    n_test = 60
    lhs_test = sampler.random(n_test)
    test_samples = qmc.scale(lhs_test, bounds[:, 0], bounds[:, 1])

    params_mean = emulator_data['params_mean']
    params_std = emulator_data['params_std']
    test_norm = (test_samples - params_mean) / params_std

    gps = emulator_data['gp_emulators']
    n_ell = len(gps)
    ell = np.array([g['ell'] for g in gps])

    pred = np.zeros((n_test, n_ell))
    pred_std = np.zeros((n_test, n_ell))

    for i_ell, gp_entry in enumerate(gps):
        gp = gp_entry['gp']
        y_mean = gp_entry['y_mean']
        y_std_val = gp_entry['y_std']

        mu_norm, std_norm = gp.predict(test_norm, return_std=True)
        pred[:, i_ell] = mu_norm * y_std_val + y_mean
        pred_std[:, i_ell] = std_norm * y_std_val

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_unc = np.abs(pred_std) / np.abs(pred)
    rel_unc = np.clip(rel_unc, 0, 100)

    return ell, rel_unc, pred, pred_std


# ============================================================================
# Run diagnostics
# ============================================================================
results = {}
for label in ['bin1', 'bin2']:
    print(f"\n{'='*60}")
    print(f"Processing {label}...")
    print(f"{'='*60}")

    ell_loo, loo_errors = compute_loo_errors(emulators[label])
    ell_test, test_unc, test_pred, test_std = compute_test_uncertainty(emulators[label])

    results[label] = {
        'ell': ell_loo,
        'loo_errors': loo_errors,
        'test_unc': test_unc,
        'test_pred': test_pred,
        'test_std': test_std,
    }

    # Summary stats
    median_per_ell = np.median(loo_errors, axis=0)
    in_range = (ell_loo >= 1000) & (ell_loo <= 7000)
    out_range = ~in_range

    print(f"  LOO median error (all ell):    {np.median(loo_errors):.4%}")
    if np.any(in_range):
        print(f"  LOO median error (1000<ell<7000): {np.median(loo_errors[:, in_range]):.4%}")
    if np.any(out_range):
        print(f"  LOO median error (outside):    {np.median(loo_errors[:, out_range]):.4%}")


# ============================================================================
# Summary table
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE: LOO Median Relative Error")
print("=" * 70)
print(f"{'Bin':<8} {'All ell':<14} {'1000<ell<7000':<16} {'Outside':<14}")
print("-" * 52)
for label in ['bin1', 'bin2']:
    ell = results[label]['ell']
    err = results[label]['loo_errors']
    in_range = (ell >= 1000) & (ell <= 7000)
    out_range = ~in_range

    all_med = np.median(err)
    in_med = np.median(err[:, in_range]) if np.any(in_range) else float('nan')
    out_med = np.median(err[:, out_range]) if np.any(out_range) else float('nan')

    print(f"{label:<8} {all_med:<14.4%} {in_med:<16.4%} {out_med:<14.4%}")

# Also print GP uncertainty on test samples
print("\n" + "=" * 70)
print("GP Predicted Uncertainty on Test Samples (median |std/pred|)")
print("=" * 70)
print(f"{'Bin':<8} {'All ell':<14} {'1000<ell<7000':<16} {'Outside':<14}")
print("-" * 52)
for label in ['bin1', 'bin2']:
    ell = results[label]['ell']
    unc = results[label]['test_unc']
    in_range = (ell >= 1000) & (ell <= 7000)
    out_range = ~in_range

    all_med = np.median(unc)
    in_med = np.median(unc[:, in_range]) if np.any(in_range) else float('nan')
    out_med = np.median(unc[:, out_range]) if np.any(out_range) else float('nan')

    print(f"{label:<8} {all_med:<14.4%} {in_med:<16.4%} {out_med:<14.4%}")


# ============================================================================
# Plot: median relative error vs ell for both bins
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- Panel 1: LOO cross-validation errors ---
ax = axes[0]
for label, color, marker in [('bin1', 'C0', 'o'), ('bin2', 'C1', 's')]:
    ell = results[label]['ell']
    err = results[label]['loo_errors']
    median_err = np.median(err, axis=0) * 100
    p16 = np.percentile(err, 16, axis=0) * 100
    p84 = np.percentile(err, 84, axis=0) * 100

    ax.plot(ell, median_err, f'-{marker}', color=color, markersize=4,
            linewidth=1.5, label=f'{label} (z={emulators[label]["z_eff"]:.3f})')
    ax.fill_between(ell, p16, p84, alpha=0.15, color=color)

ax.axvline(1000, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(7000, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(5, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='5% target')
ax.axhspan(0, 5, alpha=0.05, color='green')

ax.set_ylabel('LOO Relative Error (%)')
ax.set_title('GP Emulator Validation: LOO Cross-Validation Error vs $\\ell$')
ax.set_xscale('log')
ax.set_ylim(0, min(80, ax.get_ylim()[1]))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Label the fit range
ax.text(2500, ax.get_ylim()[1] * 0.92, 'fit range', ha='center',
        fontsize=9, color='gray', style='italic')

# --- Panel 2: GP predicted uncertainty on test samples ---
ax = axes[1]
for label, color, marker in [('bin1', 'C0', 'o'), ('bin2', 'C1', 's')]:
    ell = results[label]['ell']
    unc = results[label]['test_unc']
    median_unc = np.median(unc, axis=0) * 100
    p16 = np.percentile(unc, 16, axis=0) * 100
    p84 = np.percentile(unc, 84, axis=0) * 100

    ax.plot(ell, median_unc, f'-{marker}', color=color, markersize=4,
            linewidth=1.5, label=f'{label} (z={emulators[label]["z_eff"]:.3f})')
    ax.fill_between(ell, p16, p84, alpha=0.15, color=color)

ax.axvline(1000, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(7000, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(5, color='red', linestyle=':', linewidth=0.8, alpha=0.5, label='5% target')
ax.axhspan(0, 5, alpha=0.05, color='green')

ax.set_xlabel(r'$\ell$')
ax.set_ylabel('GP Uncertainty / |Prediction| (%)')
ax.set_title('GP Predicted Uncertainty on Test Samples vs $\\ell$')
ax.set_xscale('log')
ax.set_ylim(0, min(80, ax.get_ylim()[1]))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(2500, ax.get_ylim()[1] * 0.92, 'fit range', ha='center',
        fontsize=9, color='gray', style='italic')

plt.tight_layout()
outpath = 'figures/gp_emulator_validation_bin1_bin2.pdf'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved plot: {outpath}")

# Also save a PNG for quick viewing
plt.savefig(outpath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"Saved plot: {outpath.replace('.pdf', '.png')}")

plt.close()
print("\nDone.")
