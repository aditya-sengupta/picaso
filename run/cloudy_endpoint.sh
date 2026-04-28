#!/bin/bash
#SBATCH --job-name=cldy_end      # Job name
#SBATCH --partition=fortney-nimmo    # queue for job submission
#SBATCH --account=fortney-nimmo       # queue for job submission
#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adityars@ucsc.edu   # Where to send mail
#SBATCH --ntasks=40                  # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=40         # How many tasks on each node
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --output=cldy_end_%j.log     # Standard output and error log

pwd; hostname; date

echo $1
module load python

gravs=(10 56 177 562 1778)
semimajors=(0.02 0.04 0.13 0.50)
fseds=(1 2 3 4 8)

for grav in "${gravs[@]}"; do
    for semimajor in "${semimajors[@]}"; do
    	for fsed in "${fseds[@]}"; do
        	/home/adityars/anaconda3/envs/picaso4/bin/python scripts/cloudy_irradiated_withinversion_refined.py "$grav" 2400 "$semimajor" "$fsed"
    	done
    done
done

date
