import os
import sys
import pandas as pd
import numpy as np







def main():
	trace = "ee9e8c"
	fname = "{}_{}_result.csv"
	policy = "PCS_pred"

	df = pd.read_csv(fname.format(policy, trace))

	df['jct'] =  df['end_time'] - df['submit_time']
	valid_prediction = (df['estimated_end_time'] == -1) & (df['estimated_start_time'] == -1)
	# valid_prediction = df['estimated_end_time'] != -1

	filtered_df = df.drop(df[valid_prediction].index)

	filtered_df['pred_jct'] =  filtered_df['estimated_end_time'] - filtered_df['submit_time']
	filtered_df['jct'] =  filtered_df['end_time'] - filtered_df['submit_time']

	filtered_df['error'] = 100.0 * (filtered_df['pred_jct'] - filtered_df['jct']).abs() / filtered_df['pred_jct']

	print(df['jct'].mean())
	print(filtered_df['error'].mean())
	print(filtered_df['error'].quantile(0.99))







def main2():
	fdir = "new/"

	# seeds=[1954,3266,5897,6359,9005]

	# seed = 9005

	fname = "{}_gavel_result.csv"
	PCS_fname = "{}_gavel_result.csv"


	file_path = os.path.join(fdir, PCS_fname.format('THEMIS'))

	# Load the CSV file into a DataFrame
	df = pd.read_csv(file_path)
	df['jct'] =  df['end_time'] - df['submit_time']


	valid_prediction = df['estimated_end_time'] != -1 and df['start_time'] != -1
	filtered_df = df[valid_prediction]

	filtered_df['pred_jct'] =  filtered_df['estimated_end_time'] - filtered_df['submit_time']
	filtered_df['jct'] =  filtered_df['end_time'] - filtered_df['submit_time']

	filtered_df['error'] = 100.0 * (filtered_df['pred_jct'] - filtered_df['jct']).abs() / filtered_df['pred_jct']


	print(df)
	print(filtered_df)

	# print(df['jct'].mean())
	# print(filtered_df['error'].quantile(0.99))






	

def orig_main():
	

	fdir = "orig/gavel-contsingle/1.2"

	seeds=[1954,3266,5897,6359,9005]

	seed = 9005

	fname = "{}_64_1.2_gavel-contsingle_{}.csv"
	PCS_fname = "{}_64_1.2_gavel-contsingle_{}_{}.csv"


	file_path = os.path.join(fdir, fname.format('THEMIS', seed))
	# file_path = os.path.join(fdir, PCS_fname.format('MCS', seed, 2))

	# Load the CSV file into a DataFrame
	df = pd.read_csv(file_path)
	df['jct'] =  df['end_time'] - df['submit_time']
	df['pred_jct'] =  df['estimated_end_time'] - df['submit_time']
	df['error'] = 100.0 * (df['pred_jct'] - df['jct']).abs() / df['pred_jct']

	print(df['jct'].mean())
	print(df['error'].quantile(0.99))

	return

	print(df['service'].quantile(1.0))


	no_wait_condition = df['submit_time'] == df['start_time']

	filtered_df = df[no_wait_condition]

	slowdown = (filtered_df['jct'])/filtered_df['service']

	slowdown_neq_1 = filtered_df[slowdown != 1]

	gpus = slowdown_neq_1['service']/slowdown_neq_1['jct']

	# print(slowdown)

	# print(slowdown_neq_1)

	print(gpus.max())

	return

	jct = df['end_time'] - df['submit_time']
	predicted_jct = df['estimated_end_time'] - df['submit_time']

	service = df['service']

	# error = 100.0 * (predicted_jct - jct).abs() / predicted_jct

	slowdown = jct/service

	print(error.quantile(0.99))


	return


	
	fnames = [fname.format('SRSF', seed),
			fname.format('AFS', seed),
			fname.format('THEMIS', seed),
			PCS_fname.format('MCS', seed, 0),
			PCS_fname.format('MCS', seed, 1),
			PCS_fname.format('MCS', seed, 2),]

	for fname in fnames:
		file_path = os.path.join(fdir, fname)

		# Load the CSV file into a DataFrame
		df = pd.read_csv(file_path)

		# print(df.columns)

		jct = df['end_time'] - df['submit_time']
		predicted_jct = df['estimated_end_time'] - df['submit_time']

		error = 100.0 * (predicted_jct - jct).abs() / predicted_jct

		print(error.quantile(0.99))







if __name__ == '__main__':
	main()
	# orig_main()