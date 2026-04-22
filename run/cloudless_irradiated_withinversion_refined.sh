#!/bin/bash
#SBATCH --job-name=cldfr_refined      # Job name
#SBATCH --partition=cpuq    # queue for job submission
#SBATCH --account=cpuq       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=40                  # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=cldfr_refined_%j.log     # Standard output and error log

pwd; hostname; date

echo $1
module load python

teffs=(1000 1200 1400 1600 1800 2000 2200 2400)
# gravs=(17 31 100 316 1000 3160)
semimajors=(0.02 0.04 0.13 0.50)

for teff in $(seq 800 10 2400); do
    for semimajor in "${semimajors[@]}"; do
        /home/adityars/anaconda3/envs/picaso4/bin/python scripts/cloudless_irradiated_withinversion_refined.py $1 "$teff" "$semimajor"
    done
done

date
