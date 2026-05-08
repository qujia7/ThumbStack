#!/bin/bash
#SBATCH --account=rrg-rbond-ac
#SBATCH --nodes=15
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=16
#SBATCH --job-name=stage_profile_ISO
#SBATCH --output=/gpfs/fs1/home/r/rbond/jiaqu/Thumbstack_DESI/output/ISO/stage_profile/slurm_out_stage_profile_ISO_niagara_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jq247@cam.ac.uk

export DISABLE_MPI=false

module load NiaEnv/2022a                                                                                        
module load autotools                                                                                           
module load gcc/11.3.0                                                                                          
module load openblas                                                                                            
module load gsl                                                                                                 
module load openmpi                                                                                             
module load fftw                                                                                                
module load python
source /home/r/rbond/jiaqu/.bashrc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run from the repo root so test.py resolves; this file lives in ISO/.
cd "$SLURM_SUBMIT_DIR/.."

srun python test.py --savename=ISO --catalogue=full_catalog_Y1_renorm_ISO.txt --cmb=/home/r/rbond/jiaqu/projects/SO/ISO/bossn_f90_f150_i1_nilc_coadd_blackbody_temperature_ellmin_100_ellmax_4000_beam_2arcmin_mask_i1_20250704.fits --mask=/home/r/rbond/jiaqu/projects/SO/ISO/mask_i1_20250704.fits --output-dir /gpfs/fs1/home/r/rbond/jiaqu/Thumbstack_DESI/output/ISO/stage_profile
