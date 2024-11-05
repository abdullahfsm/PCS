import numpy as np



def main():
	# Generate 10 random numbers from an exponential distribution with scale parameter 2
	samples = np.random.exponential(scale=2, size=10000)

	print(np.mean(samples))


if __name__ == '__main__':
	main()