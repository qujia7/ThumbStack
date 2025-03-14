#!/bin/bash
#SBATCH --nodes=15
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=5
#SBATCH --output=/scratch/r/rbond/jiaqu/mpi_output_%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=jq247@cam.ac.uk
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

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


#srun --cpu-bind=cores python test.py --random --savename full_catalog_Y3_random

srun --cpu-bind=cores python test_tsz.py --savename full_tsz_y3