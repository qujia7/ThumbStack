# Thumbstack_DESI

A pipeline for performing stacked analyses with CMB maps around galaxy positions to extract tSZ and kSZ signals.

## Overview

This pipeline implements aperture photometry (AP) filtering of CMB maps around galaxy positions from the DESI catalog. It calculates various estimators for the thermal Sunyaev-Zeldovich (tSZ) and kinematic Sunyaev-Zeldovich (kSZ) effects.

## Pipeline Components

### Core Classes

- `ThumbStack`: Main class for performing stacked analyses
  - Handles loading galaxy catalogs and CMB maps
  - Applies various aperture photometry filters
  - Performs stacking with different estimators
  - Computes covariance matrices through bootstrap resampling and velocity shuffling
  - Produces signal-to-noise estimates and plots

### Aperture Photometry Filters

Various filter types are supported:

```python
valid_filters = [
   'diskring',      # Disk minus ring filter (default)
   'disk',          # Disk-only filter
   'ring',          # Ring-only filter 
   'meanring',      # Normalized ring filter
   'cosdisk',       # Cosine-weighted disk filter
   'taudisk',       # Disk filter for tau estimator
   'tauring',       # Ring filter for tau estimator
   'taudiskring'    # Disk-ring filter for tau estimator
]



Workflow

The pipeline operates in two main stages:


Profile Computation (test.py): Computes stacked profiles

Extracts postage stamps around galaxy positions
Applies aperture photometry filters
Computes average profiles
Bootstrap Analysis (test_cov.py): Estimates covariance matrices

Performs bootstrap resampling
Supports block bootstrap for spatially correlated data
Computes velocity shuffling as a null test

Usage
Configuration
The core settings are configured through mbatch.yaml:

stages:
  stage_profile:
    # Compute stacked profiles
    script: test.py
    exec: python
    parallel:
      nproc: 75       # 15 nodes * 5 tasks per node
      threads: 16     # 16 CPUs per task
      walltime: 00:30:00

  stage_bootstrap:
    # Compute bootstrap covariance
    script: test_cov.py
    exec: python
    depends:
      - stage_profile
    parallel:
      nproc: 50       # 10 nodes * 5 tasks per node
      threads: 16     # 16 CPUs per task
      walltime: 00:20:00

Running the Pipeline
To run the complete pipeline:
   mbatch output_fname mbatch.yaml