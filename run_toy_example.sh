#!/bin/bash

# FIFO
python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy FIFO -output_file FIFO_toy_result.csv

# SRSF
python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy SRSF -output_file SRSF_toy_result.csv

# PCS-pred
python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_themis1_avg_jct_avg_pred_error_pred.pkl -output_file PCS_pred_toy_result.csv

# PCS-bal
python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_themis1_avg_jct_avg_pred_error_bal.pkl -output_file PCS_bal_toy_result.csv

# PCS-jct
python3 simulation/sim.py -workload themis1 -num_gpus 64 -num_apps 500 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_themis1_avg_jct_avg_pred_error_jct.pkl -output_file PCS_jct_toy_result.csv

#Compare results
python3 simulation/utils/result_summary.py -fnames FIFO_toy_result.csv SRSF_toy_result.csv PCS_pred_toy_result.csv PCS_bal_toy_result.csv PCS_jct_toy_result.csv