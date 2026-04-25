#!/bin/bash
#SBATCH --job-name=cldy_parallel      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=400                  # Number of MPI ranks
#SBATCH --nodes=10                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=cldy_parallel_%j.log     # Standard output and error log

pwd; hostname; date

module load python
module load openmpi

mpiexec -n 40 /home/adityars/anaconda3/envs/picaso4/bin/python scripts/cloudless_irradiated_withinversion_refined.py

date
