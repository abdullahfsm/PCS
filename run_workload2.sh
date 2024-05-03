#!/bin/bash

output_files=()

# for trace in {0e4a51,}
for trace in {0e4a51,6214e9,6c71a0,b436b2}
do
	for metric in {avg_jct_avg_pred_error,}
	do
		for var in {jct,bal,pred,}
		do
			python3 simulation/sim.py -scheduling_policy MCS -num_gpus 64 -output_file PCS_"$trace"_"$metric"_"$var".csv -trace trace_"$trace" -estimate 1 -MCS_config_file data/PCS_configs/PCS_config_"$trace"_"$metric"_"$var".pkl -num_apps 5000
			output_files+=(PCS_"$trace"_"$metric"_"$var".csv)
			echo "$var" done
		done
		echo "$metric" done	
	done
	echo PCS done


	for policy in {FIFO,SRSF,AFS,}
	do
		python3 simulation/sim.py -scheduling_policy "$policy" -num_gpus 64 -output_file "$policy"_"$trace".csv -trace trace_"$trace" -estimate 1 -num_apps 5000
		output_files+=("$policy"_"$trace".csv)
		echo "$policy" done
	done
	
done

python3 simulation/utils/result_summary.py -fnames "${output_files[@]}" -normalize_jct