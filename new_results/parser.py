import os
import sys
import pandas as pd
import numpy as np

def main():
	
	fdir = "orig/gavel-contsingle/1.2"

	seeds=[1954,3266,5897,6359,9005]

	seed = 9005

	policy = "SRSF"

	fname = "{}_64_1.2_gavel-contsingle_{}.csv"
	PCS_fname = "{}_64_1.2_gavel-contsingle_{}_{}.csv"





	file_path = os.path.join(fdir, fname.format('FIFO', seed))

	# Load the CSV file into a DataFrame
	df = pd.read_csv(file_path)
	df['jct'] =  df['end_time'] - df['submit_time']
	df['pred_jct'] =  df['estimated_end_time'] - df['submit_time']

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