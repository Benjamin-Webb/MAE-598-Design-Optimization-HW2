# Plot for problem 4

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	# Let's plot

	# Make some vectors for I and It
	I = np.linspace(0.0, 10.0, 101)
	I[0] = 0.001
	It = np.ones((101, 1))*5.0

	# Evaluate h(I, It)
	h = np.zeros((101, 1))
	for j in range(0, 101, 1):
		if I[j] <= It[j]:
			h[j] = It[j] / I[j]
		else:
			h[j] = I[j] / It[j]

	fig = plt.figure(num=1)
	plt.plot(I[0:50], h[0:50], '-b', I[50:100], h[50:100], '-r')
	plt.grid(b=True, which='both', axis='both')
	plt.xlabel('I')
	plt.ylabel('h(I,It)')
	plt.xlim((0.0, 10.0))
	plt.ylim((0.0, 2.5))
	plt.legend(['I <= It', 'I >= It'])
	plt.show()

	# Make some vectors for I and It
	I = np.linspace(0.0, 20.0, 201)
	I[0] = 0.001
	It = np.ones((201, 1)) * 5.0

	# Evaluate h(I, It)
	h = np.zeros((201, 1))
	for j in range(0, 201, 1):
		if I[j] <= It[j]:
			h[j] = It[j] / I[j]
			# restrict h
			if h[j] <= 2.0:
				h[j] = 2.0
		else:
			h[j] = I[j] / It[j]
			# restrict h
			if h[j] <= 2.0:
				h[j] = 2.0

	fig = plt.figure(num=2)
	plt.plot(I[0:25], h[0:25], '-b', I[25:100], h[25:100], '-g', I[100:200], h[100:200], '-r')
	plt.grid(b=True, which='both', axis='both')
	plt.xlabel('I')
	plt.ylabel('h(I,It)')
	plt.xlim((0.0, 20.0))
	plt.ylim((0.0, 5.0))
	plt.show()
