# Project Context: Thumbstack_DESI

## Environment Setup

### Terminal Environment (HPC/Compute Canada)
```bash
export DISABLE_MPI=false
module load StdEnv/2023
module load aocl-lapack/5.1
module load openblas
module load gsl
module load openmpi
module load fftw
module load cfitsio
module load python
source /home/jiaqu/.bashrc
```

### Notebook Environment
- Virtualenv: `~/.virtualenvs/trillium/bin/python`

## Paper Overview
**Title:** "Tracing Gas Around Luminous Red Galaxies: New precision Kinematic Sunyaev-Zel'dovich measurements with ACT DR6 and DESI DR2 galaxies"

**Authors:** Frank J. Qu, Bernardita Ried Guachalla, Emmanuel Schaan, Boryana Hadzhiyska, Simone Ferraro

**Key Results:**
- 18-sigma detection of kSZ signal using 2,376,871 DESI LRG Y3 galaxies cross-correlated with ACT DR6
- Novel harmonic-space methodology constructing momentum-weighted kSZ templates
- Configuration-space CAP (Compensated Aperture Photometry) measurements at SNR=17.5
- GNFW gas profile constraints with HOD marginalization via GP emulator
- Redshift and stellar mass dependence analysis

## Datasets
- **Galaxy catalog:** DESI Year 3 LRGs (spectroscopic), ~2.4 million objects after masking
- **CMB maps:** ACT DR6 hILC temperature maps (night-time, dr6.01)
- **Redshift range:** 0.4 < z < 1.1, divided into bins (z1: 0.4-0.6, z2: 0.6-0.8, z3: 0.8-1.1)
- **Stellar mass bins:** mass1-mass4 in log10(M*/Msun): (10.5-11.2), (11.2-11.4), (11.4-11.6), (11.6-12.5)

## Methodology

### Harmonic Space Analysis
1. Build kSZ template by projecting galaxy LOS velocities onto HEALPix grid (Nside=8192)
2. Cross-correlate template with CMB using NaMaster pseudo-Cl estimator
3. Model with class-sz halo model (1-halo + 2-halo terms)
4. Fit GNFW gas profile parameters: rho0, xc, beta, Ak2h (fixing gamma=-0.2, alpha=1)

### Configuration Space Analysis
1. Extract CMB cutouts around galaxy positions
2. Apply CAP filter (disk minus ring of equal area)
3. Stack with velocity weighting: T_kSZ = weighted sum of T_i * v_i
4. Covariance from block bootstrap (2x2 deg patches, 10,000 realizations)

### Key Equations
- kSZ template: p(theta) = (1/Omega_pix * n_bar) * sum(v_r,i/c)
- Cross-spectrum: C_l^{p x T} = -(1/3) * r * sigma_true * sigma_rec * C_l^{tau g}
- Velocity correlation coefficient: r = 0.65 (from Hadzhiyska et al. 2024 light cone)

## Repository Structure
- **Core pipeline:** `thumbstack.py` (ThumbStack class for stacking)
- **Profile computation:** `test.py` (compute stacked profiles)
- **Covariance:** `test_cov.py` (bootstrap covariance estimation)
- **Harmonic analysis:** `namaste_car.py`, `namaste_error.py` (NaMaster-based)
- **Template preparation:** `prepare_templates_hp.py` (HEALPix), `prepare_templates.py` (CAR)
- **Theory:** `theory_pred.py`, `theory_pred_new_hod.py`, `class_sz_ksz.py`
- **GP emulator:** `gp_emulator.py` (for HOD marginalization)
- **Catalog prep:** `prepare_catalogue.py`, `catalogue_merged.py`
- **VTK export:** `export_vtk_visit.py` (export Y1/Y3 catalogs to VTK for VisIt visualization)

## Current Analysis Status

### Completed
- Y3 catalog preparation with redshift/mass binning
- Harmonic-space kSZ detection (18-sigma)
- Configuration-space CAP measurements (SNR=17.5)
- GNFW profile fitting with fixed HOD
- HOD marginalization via GP emulator
- Null tests: velocity shuffling, North/South isotropy
- Y1 vs Y3 consistency checks

### In Progress / Outstanding Items (from paper TODOs)
- Figure 4: ~~Check units for axes/colorbar, extend galaxy dot size, add legend~~ **DONE** - Changed to scatter plot of individual galaxy positions (plot_no_vel.py); updated caption
- Figure 5: Consider only plotting rho, x, beta given degeneracies; add HOD variation plot
- Figure 7: Quote PTE for best-fit and Battaglia curves
- Figure 9: Add null test for combined sample; null test from 90-150 or y-map
- Figure 10: Clarify color bar description as "noise" not residual bias
- Figure 12: Reduce horizontal offset between data points
- Figure 13: Specify what the fiducial is; add caveat about r(z) degeneracy
- Table 1: Specify which HOD was used
- Table 2: Specify HOD source for marginalization
- Section 4.2: Describe the HOD used and cite
- Section 5.0.2: Clarify "wrt" in caption

- Section 5.1.2: Make clear about imperfect consistency with photo-z
- Conclusion: Check SNR scaling (sqrt(N) not N); rephrase future work positively

## Key Systematics
1. **Velocity correlation coefficient r:** Degeneracy with gas amplitude; may vary with z and mass
2. **HOD uncertainty:** ~50% degradation in rho0 constraints after marginalization
3. **Profile truncation:** Choice of rcut affects predictions at large radii
4. **Satellite fraction:** Affects velocity reconstruction via FoG effects

## External Dependencies
- `healpy`, `pixell` for map handling
- `class-sz` for SZ theory calculations (Bolliet et al. 2024)
- `NaMaster` for pseudo-Cl estimation
- `scikit-learn` for GP emulator
- `PocoMC` for MCMC/HOD posteriors

## Decisions Log
- Use r=0.65 from Hadzhiyska et al. 2024 light cone analysis
- Fix GNFW gamma=-0.2, alpha=1 following Amodeo et al. 2021 / Battaglia 2016
- Profile truncation: enforce enclosed gas mass = fb * m200c
- Block bootstrap with 2x2 deg patches for spatial correlations

---

## Template Normalization Methods (Robustness Test)

### Background
Two methods exist for normalizing the kSZ velocity template. Both aim to construct:
$$T_p = \frac{\sum_{i \in p} v_i / c}{\bar{n}_p}$$
where $\bar{n}_p$ is the expected galaxy count in pixel $p$.

### Method 1: Global nbar (Original — `prepare_templates.py`)

**Implementation:**
```python
nbar_2D = N_gal_in_footprint / Omega_survey  # gal/sr, single global value
T[p] = vsum[p] / (nbar_2D * Omega[p])
```

**Characteristics:**
- Uses uniform density assumption across survey
- Galaxy mask is binary (Heaviside): 1 if randoms exist in pixel, 0 otherwise
- `nbar_2D × Omega[p]` = expected count per pixel (same everywhere except for pixel area variation)

**Scripts:** `survey_mask.py` → `survey_mask_car.py` → `prepare_templates.py` → `namaste_car.py`

### Method 2: Local nbar (Equation 8 — `prepare_templates_local_nbar.py`)

**Implementation (based on pseudo-Cl literature, e.g., NaMaster):**
$$\bar{n}_p = \alpha_r \times N_p^{\text{randoms}}, \quad \alpha_r = \frac{N_{\text{data}}}{N_{\text{randoms}}}$$

```python
# In survey_mask_local_nbar.py:
alpha_r = N_data / N_randoms_total
mask[p] = alpha_r * random_counts[p]

# In prepare_templates_local_nbar.py:
T[p] = vsum[p] / nbar_local[p]
```

**Characteristics:**
- Uses random catalog to estimate local selection function
- Accounts for survey depth/completeness variations
- Pixels with fewer randoms get higher weight (1/nbar)

**Scripts:** `survey_mask_local_nbar.py` → `prepare_templates_local_nbar.py` → `namaste_car_local_nbar.py`

### Key Findings (2026-01-09)

| Quantity | Method 1 (Global) | Method 2 (Local) |
|----------|-------------------|------------------|
| nbar mean | 2.87×10⁻² | 2.45×10⁻² |
| 1/nbar mean | 34.86 | **55.23** |
| Template std | 4.37×10⁻³ | 7.82×10⁻³ |
| Cl amplitude | baseline | **~3× higher** |
| fsky | 0.1203 | 0.1203 (identical) |

**Critical Issue: Jensen's Inequality**
$$\text{mean}(1/\bar{n}_p) \neq 1/\text{mean}(\bar{n}_p)$$

- Local method: mean(1/nbar) = 55.23
- Global method: 1/(nbar_2D×Ω) = 34.86
- Ratio: 1.58 → Cl ratio ≈ 1.58² ≈ 2.5–3×

The local method effectively **upweights sparse regions**, creating larger template variance and higher Cl amplitude, even though both methods use the same footprint (fsky identical).

### Unresolved Questions

1. **Normalization consistency:** Should local nbar be renormalized so mean(1/nbar_local) = 1/(nbar_2D×Ω)?
2. **Theory interpretation:** Does the theory prediction need adjustment for local weighting?
3. **Which is correct?** Literature (e.g., equation 8 from pseudo-Cl papers) suggests local, but amplitude difference needs explanation.

### Catalog Requirements

For Method 2, `survey_mask_local_nbar.py` requires:
- **Catalog:** Full DESI footprint (no CMB mask), same z-cuts as analysis
- **Current:** `/scratch/jiaqu/desi/catalogue/desi_lrg_full_footprint_z0.4_1.1.csv` (4,468,483 galaxies)
- **Randoms:** `/project/rrg-rbond-ac/jiaqu/DESI/catalogs/DA2/LSS/loa-v1/LSScats/v1.1/nonKP/` (451M randoms)
