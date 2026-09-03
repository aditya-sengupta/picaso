#!/usr/bin/env bash

while true; do
    mpiexec -n 64 python -u -m mpi4py scripts/restart_cloudy_unirradiated.py
    exit_code=$?

    if [[ $exit_code -ne 9 && $exit_code -ne 137 ]]; then
        exit "$exit_code"
    fi
done
