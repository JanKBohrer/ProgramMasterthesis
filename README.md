# ProgramMasterthesis
Simulation program for warm cloud microphysics using Lagrangian (discrete) particle methods.

The program was presented in the Masterthesis of Jan Kai Bohrer, submitted to the Faculty of Physics, University Freiburg, October 2019. This repository represents the program's state at date of submission and will not be changed.

The program enables the simulation of a drizzling stratocumulus cloud in a two-dimensional kinematic framework, built for the 8th International Cloud Modeling Workshop 2012 (Test case 1 in Muhlbauer et al., Bulletin of the American Meteorological Society 94, 45 (2013))

To perform and evaluate a simulation:

1. Generate grid and particles, using the shell script "run_gen_grid.sh". Set required parameters in here.
2. Simulate the spin-up period, using "run_cloudMP.sh". Set sim_type="spin_up" and other parameters in here.
3. Start the main simulation, using "run_cloudMP.sh". Set sim_type="with_collision" for collisions or "wo_collision" and other parameters in here.
4. If required, evaluate data, using "run_gen_plot_data.sh".
5. Data can be plotted with the python script "plot_results_MA.py". See comments in there.

In case of any questions, please contact the repository admin (Jan Bohrer).
