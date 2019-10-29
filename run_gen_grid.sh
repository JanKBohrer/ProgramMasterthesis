#!/bin/bash
# execute generate_grid_and_particles.py Nsim times with different seeds

storage_path="/Users/bohrer/sim_data_cloudMP/" # in here, the generated grids/SIPs will be stored as a defined system state, which can then be loaded and simulated by cloud_MP.py
sseed=2101 # SIP generation seed for the FIRST grid, seeds of other grids will be sseed+2, sseed+4, sseed+6, ...
Nsim=2 # number of generated grids
solute_type="AS"
no_spcm0=2 # number of super particles first mode
no_spcm1=3 # number of super particles second mode
dx=20 # grid step size in m
dz=20 # grid step size in m

#for x in {1..50}
for ((n=0; n < $Nsim; n++))
do
    # restrict number of threads per job
    export OMP_NUM_THREADS=8
    export NUMBA_NUM_THREADS=16
    echo $((sseed + 2*n))
    python3 generate_grid_and_particles.py $storage_path $((sseed + 2*n)) $solute_type $no_spcm0 $no_spcm1 $dx $dz &
done
