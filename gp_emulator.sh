#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=05:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/jiaqu/mpi_output_%j.txt
#SBATCH --cpus-per-task=192
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


# Pass --bin via: sbatch gp_emulator.sh 2   (default: bin 2)
BIN=${1:-2}
srun --cpu-bind=cores python gp_emulator.py --bin $BIN