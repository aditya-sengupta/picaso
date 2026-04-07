#!/bin/bash

# Run run_cloudy_bd.py over parameter grid

teffs=(200 400 600 800 1000)
gravs=(316 1000 3160)
semimajors=(0.02 0.04 0.06 0.08 0.10)

for teff in "${teffs[@]}"; do
    for grav in "${gravs[@]}"; do
        for semimajor in "${semimajors[@]}"; do
            python scripts/run_cloudy_bd.py fixed "$grav" "$teff" "$semimajor"
        done
    done
done