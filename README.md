# Reproducing Results 

## System Prerequisites 

* This has been tested on Ubuntu 22.04 with Python 3.10.12. 
* It is assumed that Python 3.10.12 can be invoked with `python3`.

## System Setup

* Run `bash setup.sh` and follow all prompts. 

## Running Simulation 

* Run `python3 reproduce.py -h` to see which traces and workloads are acceptable for correct usage. 
* It will also show you additional simulation configuration parameters.

# Navigating the Interface

* Upon running `python3 reproduce.py <workload>`, you will be shown an interactive figure showing the pareto front. 
* Click a point on the pareto front and click `Create Config`. This will generate a configuration file for a simulation which will then run. The simulation results are displayed at the end.
