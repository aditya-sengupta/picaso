#!/bin/bash
#SBATCH --job-name=hj_fixed_evolution      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=80                  # Number of MPI ranks
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=hj_fixed_evolution_%j.log     # Standard output and error log

pwd; hostname; date

/home/adityars/anaconda3/envs/picaso/bin/python scripts/hj_fixed_evolution.py

date