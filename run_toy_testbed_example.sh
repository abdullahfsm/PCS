#!/bin/bash

output_files=()

# PCS configs
for var in {jct,bal,pred,}
do
	python3 testbed/run_experiment.py -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_testbed_toy_"$var".pkl -trace toy_trace -output_file PCS_"$var"_toy.csv
	output_files+=(PCS_"$var"_toy.csv)
	echo "$var" done
done


# Other policies
for policy in {FIFO,SRSF,FS,}
do
	python3 testbed/run_experiment.py -scheduling_policy "$policy" -trace toy_trace -output_file "$policy"_toy.csv
	output_files+=("$policy"_toy.csv)
	echo "$policy" done
done

python3 testbed/utils/result_summary.py -fnames "${output_files[@]}" -normalize_jct