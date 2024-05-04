#!/bin/bash

# PCS configs
python3 run_experiment.py -scheduling_policy MCS -MCS_config_file PCS_config_toy_pred.pkl -trace toy_workload.csv -output_file PCS_pred_toy_result.csv
python3 run_experiment.py -scheduling_policy MCS -MCS_config_file PCS_config_toy_bal.pkl -trace toy_workload.csv -output_file PCS_bal_toy_result.csv
python3 run_experiment.py -scheduling_policy MCS -MCS_config_file PCS_config_toy_jct.pkl -trace toy_workload.csv -output_file PCS_jct_toy_result.csv

# FIFO
python3 run_experiment.py -scheduling_policy FIFO -trace toy_workload.csv -output_file FIFO_toy_result.csv

# SRSF
python3 run_experiment.py -scheduling_policy SRSF -trace toy_workload.csv -output_file SRSF_toy_result.csv

# FS
python3 run_experiment.py -scheduling_policy FS -trace toy_workload.csv -output_file FS_toy_result.csv