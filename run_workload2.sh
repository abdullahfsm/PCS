#!/bin/bash

output_files=()

for trace in {ee9e8c,}
# for trace in {0e4a51,6214e9,6c71a0,b436b2,ee9e8c,}
do
	for metric in {avg_jct_avg_pred_error,}
	do
		for var in {jct,bal,pred,}
		do
			output_file=new_data/PCS_"$var"_"$trace"_result.csv
			python3 simulation/sim.py -scheduling_policy MCS -num_gpus 64 -output_file $output_file -trace trace_"$trace" -MCS_config_file data/PCS_configs/PCS_config_"$trace"_avg_jct_avg_pred_error_"$var".pkl -num_apps -1
			output_files+=("$output_file")
			echo "$var" done
		done
		echo "$metric" done	
	done
	echo PCS done


	for policy in {FIFO,SRSF,AFS,THEMIS,}
	do
		output_file=new_data/"$policy"_"$trace"_result.csv
		python3 simulation/sim.py -scheduling_policy "$policy" -num_gpus 64 -output_file $output_file -trace trace_"$trace" -num_apps -1
		output_files+=("$output_file")
		echo "$policy" done
	done
	
done

python3 utils/parser.py -trac "${output_files[@]}" -normalize_jct