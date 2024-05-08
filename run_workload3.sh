#!/bin/bash


output_files=()
# THEMIS
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy THEMIS -output_file new_data/THEMIS_gavel_result.csv
output_files+=(new_data/THEMIS_gavel_result.csv)

# FIFO
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy FIFO -output_file new_data/FIFO_gavel_result.csv
output_files+=(new_data/FIFO_gavel_result.csv)

# SRSF
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy SRSF -output_file new_data/SRSF_gavel_result.csv
output_files+=(new_data/SRSF_gavel_result.csv)

# AFS
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy AFS -output_file new_data/AFS_gavel_result.csv
output_files+=(new_data/AFS_gavel_result.csv)

# PCS-pred
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_gavel_avg_jct_p99_pred_error_pred.pkl -output_file new_data/PCS_pred_gavel_result.csv
output_files+=(new_data/PCS_pred_gavel_result.csv)

# PCS-bal
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_gavel_avg_jct_p99_pred_error_bal.pkl -output_file new_data/PCS_bal_gavel_result.csv
output_files+=(new_data/PCS_bal_gavel_result.csv)

# PCS-jct
python3 simulation/sim.py -workload gavel -load 1.2 -num_gpus 64 -num_apps 3000 -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_gavel_avg_jct_p99_pred_error_jct.pkl -output_file new_data/PCS_jct_gavel_result.csv
output_files+=(new_data/PCS_jct_gavel_result.csv)

#Compare results
python3 new_data/parser.py -traces gavel