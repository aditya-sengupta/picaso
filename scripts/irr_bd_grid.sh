#!/bin/bash

# Run match_sagnick_fig2.py over parameter grid

teffs=(300 500 800 1100 1400 1700 2000 2300)
semimajors=(0.02 0.025 0.04 0.06 0.13 0.5)

for teff in "${teffs[@]}"; do
    for grav in "${gravs[@]}"; do
        for semimajor in "${semimajors[@]}"; do
            python scripts/match_sagnick_fig2.py "$teff" "$semimajor"
        done
    done
done