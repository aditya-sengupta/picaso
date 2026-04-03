#!/bin/bash
#SBATCH --job-name=cloudy_irradiated      # Job name
#SBATCH --partition=fortney-nimmo    # queue for job submission
#SBATCH --account=fortney-nimmo       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=80                  # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=48:00:00              # Time limit hrs:min:sec
#SBATCH --output=cloudy_irradiated_%j.log     # Standard output and error log

pwd; hostname; date

echo "starting"
module load python
/home/adityars/anaconda3/envs/picaso4/bin/python scripts/cloudy_irradiated.py

date
