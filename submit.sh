#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/r/rbond/jiaqu/mpi_output_%j.txt
#SBATCH --cpus-per-task=80
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
module load cfitsio/4.4.0 
source /home/r/rbond/jiaqu/.bashrc




export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


srun --cpu-bind=cores python /home/r/rbond/jiaqu/Thumbstack_DESI/get_spectra_master.py