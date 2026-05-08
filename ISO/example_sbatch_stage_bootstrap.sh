#!/bin/bash
#SBATCH --account=rrg-rbond-ac
#SBATCH --nodes=10
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=16
#SBATCH --job-name=stage_bootstrap_ISO
#SBATCH --output=/gpfs/fs1/home/r/rbond/jiaqu/Thumbstack_DESI/output/ISO/stage_bootstrap/slurm_out_stage_bootstrap_ISO_niagara_%j.txt
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

srun python test_cov.py --savename=ISO --catalogue=full_catalog_Y1_ISO.txt --output-dir /gpfs/fs1/home/r/rbond/jiaqu/Thumbstack_DESI/output/ISO/stage_bootstrap
