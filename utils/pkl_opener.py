import pickle
import os
import sys


def main():
	with open(sys.argv[1], 'rb') as fp:
		obj = pickle.load(fp)


	print(obj)



if __name__ == '__main__':
	
	files = os.listdir()

	traces = ['b436b2', '6214e9', '0e4a51', 'ee9e8c', '6c71a0']



	# PCS_config_0e4a51_avg_jct_avg_pred_error_bal.pkl

	for file in files:
		new_file = file.replace('MCS', 'PCS')
		os.system(f'cp {file} {new_file}')

		traces.append(file.split('config_')[-1].split('_avg_jct')[0])

	print(list(set(traces)))

