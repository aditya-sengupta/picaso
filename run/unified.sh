#!/bin/bash
#SBATCH --job-name=unified      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=320                  # Number of MPI ranks
#SBATCH --nodes=8                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=unified_%j.log     # Standard output and error log

pwd; hostname; date

export HDF5_USE_FILE_LOCKING=FALSE
export H5_NANOSLEEP=1000000

module load python
module load openmpi

mpiexec --bind-to core --map-by core -n 320 /home/adityars/anaconda3/envs/mpitest/bin/python -m mpi4py scripts/unified.py fine cloudless irradiated full outlier # Mukherjee+26 and Bobcat match

date
