#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 toy/themis1_scaled"
    exit 1
fi

# num_nodes
trace=$1


echo "Running trace $trace"

output_files=()

# PCS configs
for var in {jct,bal,pred,}
do
	output_file=PCS_"$trace"_"$var".csv
	python3 testbed/run_experiment.py -scheduling_policy MCS -MCS_config_file data/PCS_configs/PCS_config_testbed_"$trace"_"$var".pkl -trace "$trace" -output_file "$output_file"
	output_files+=("$output_file")
	echo "$var" done
done


# Other policies
for policy in {FIFO,SRSF,AFS,THEMIS,}
do
	output_file="$policy"_"$trace".csv
	python3 testbed/run_experiment.py -scheduling_policy "$policy" -trace trace_"$trace" -output_file "$policy"_"$trace".csv
	output_files+=("$output_file")
	echo "$policy" done
done

python3 testbed/utils/result_summary.py -fnames "${output_files[@]}" -normalize_jct