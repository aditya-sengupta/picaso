#!/bin/bash
#SBATCH --job-name=unified      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=50                  # Number of MPI ranks
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=25         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=unified_%j.log     # Standard output and error log

pwd; hostname; date

module load python
module load openmpi
mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py coarse cloudless full rerun # Bobcat+Mukherjee+26 match
mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py coarse cloudy full rerun # Diamondback match+irradiated
mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py fine cloudless partial rerun # full cloudless grid, finer but with less information
mpiexec -n 50 /home/adityars/anaconda3/envs/mpitest/bin/python scripts/unified.py fine cloudy partial rerun # full cloudy grid, finer but with less information

date
