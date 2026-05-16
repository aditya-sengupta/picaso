#!/bin/bash
#SBATCH --job-name=unified      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=640                  # Number of MPI ranks
#SBATCH --nodes=16                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=unified_%j.log     # Standard output and error log

pwd; hostname; date

export NUMBA_THREADING_LAYER=omp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLAS_NUM_THREADS=1
export LAPACK_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export H5_NANOSLEEP=1000000

module load python
module load openmpi
mpiexec --bind-to core --map-by core -n 640 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py coarse cloudless irradiated full no_rerun # Mukherjee+26 and Bobcat match
# mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py coarse cloudy full rerun # Diamondback match+irradiated
# mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py fine cloudless partial rerun # full cloudless grid, finer but with less information
# mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py fine cloudy partial rerun # full cloudy grid, finer but with less information

date
