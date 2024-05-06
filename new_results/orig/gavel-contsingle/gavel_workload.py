import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	service = list()

	for _ in range(5000):
		flip1 = np.random.uniform()
		num_jobs = 0

		if flip1 < 0.7:
			num_jobs = 1
		elif flip1 < 0.95:
			num_jobs = np.random.choice([3,4])
		else:
			num_jobs = 8


		flip2 = np.random.uniform()
		time_per_job = 0

		if flip2 < 0.8:
			time_per_job = np.random.uniform(low=1.5,high=3)
		else:
			time_per_job = np.random.uniform(low=3,high=4)

		service.append(np.power(10,time_per_job) * num_jobs)


	service.sort()
	cdf = np.linspace(0,1,len(service))

	plt.xscale('log')
	plt.plot(service, cdf)
	plt.show()



