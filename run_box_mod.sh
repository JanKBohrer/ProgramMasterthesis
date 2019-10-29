#!/bin/bash
export OMP_NUM_THREADS=8
export NUMBA_NUM_THREADS=16
python3 run_box_model.py &
