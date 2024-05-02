for metric in {avg_jct_avg_pred_error,}
do
	# for trace in {0e4a51,6214e9,6c71a0,b436b2}
	for trace in {6214e9,}
	do
		# for var in {jct,bal,pred}
		for var in {jct,}
		do
			python3 simulation/sim.py -scheduling_policy MCS -num_gpus 64 -output_file new_results/PCS_"$trace"_"$metric"_"$var".csv -from_trace workloads/workload_"$trace".csv -estimate 1 -MCS_config_file PCS_configs/PCS_config_"$trace"_"$metric"_"$var".pkl
		done
		echo "$var" done
	done
	echo "$metric" done
done
