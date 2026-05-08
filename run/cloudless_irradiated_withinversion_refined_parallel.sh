#!/bin/bash
#SBATCH --job-name=cldf_parallel      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=300                  # Number of MPI ranks
#SBATCH --nodes=10                    # Number of nodes
#SBATCH --ntasks-per-node=30         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=cldf_parallel_%j.log     # Standard output and error log

pwd; hostname; date

module load python
module load openmpi
mpiexec -n 300 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/cloudless_irradiated_withinversion_refined_parallel.py

date
