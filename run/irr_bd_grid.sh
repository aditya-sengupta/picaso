#!/bin/bash

# Run match_sagnick_fig2.py over parameter grid

grav=(10 31 100 316 1000 3160) # 3 3.5 4 4.5 5 5.5
teffs=(200 400 600 800 1000 1200 1400 1600 1800 2000 2200 2400)
semimajors=(0.02 0.04 0.13 0.5)

for teff in "${teffs[@]}"; do
    for grav in "${gravs[@]}"; do
        for semimajor in "${semimajors[@]}"; do
            python scripts/match_sagnick_fig2.py "$grav" "$teff" "$semimajor"
        done
    done
done