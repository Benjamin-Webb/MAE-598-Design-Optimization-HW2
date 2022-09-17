# MAE 598 - Design Optimization HW # 2
# Benjamin Webb
# 9/16/2022

import numpy as np
import matplotlib.pyplot as plt


def gradx(x):
	# Calculates gradient for problem 2
	# given input vector x and x0
	# x = current point

	g = np.zeros((2, 1))
	g[0] = 10*x[0] + 12*x[1] - 8.0
	g[1] = 20*x[1] + 12*x[0] - 14.0

	return g


def hessian():
	# Generates Hessian for problem 2
	# Hessian is constant for this problem

	H = np.array([[10.0, 12.0], [12.0, 20.0]])

	return H


def GD_inexact(t, alpha, x0, epsilon0):
	# Performs gradient desecent using inexact line search for problem 2
	# t: tuning parameter between (0, 1)
	# alpha: initial step-size
	# x0: initial starting point
	# epsilon0: stopping criteria

	# Initialize stopping critera
	epsilon = np.zeros((1000, 1))
	epsilon[0] = 1.0
	n = np.uint16(0)

	# Begin gradient descent loop
	while epsilon[n] > epsilon0:
		# Problem 2 minimize function ||x - x0||^2
		f0 = 5.0*x0[0]**2 + 10.0*x0[1]**2 + 12.0*x0[0]*x0[1] - 8.0*x0[0] - 14.0*x0[1] + 5.0

		# Calculate gradient at current point
		g0 = gradx(x0)

		# Calculate f(x - alpha*g0)
		x = x0 - alpha*g0
		f = 5.0*x[0]**2 + 10.0*x[1]**2 + 12.0*x[0]*x[1] - 8.0*x[0] - 14.0*x[1] + 5.0

		# Calculate Phi(alpha) = f0 - t*g0^T*g0*alpha
		phi = f0 - t * g0.T @ g0 * alpha

		# Determine how to proceed to next step
		if np.abs(f) > np.abs(phi):
			# Update alpha
			alpha = 0.5 * alpha
		elif n >= 1000:
			break
		else:
			n = n + np.uint16(1)
			epsilon[n] = np.linalg.norm(f - f0)
			x0 = x
			alpha = 1.0

	# Print complete statement with # of iterations
	print('Gradient Descent Optimization Complete')
	print('# of Iterations: ' + str(n))
	return x, n, epsilon[1:n+1]

def newton(x0, epsilon0):
	# Function performs Newton's method for optimization of problem 2
	# x0: initial guess
	# epsilon0: stopping criteria

	# Initialize stopping criteria
	epsilon = np.zeros((1000, 1))
	epsilon[0] = 1.0
	n = np.uint16(0)

	# Calculate Hessian once, since it is constant over domain of this function
	H0 = hessian()
	H0_1 = np.linalg.inv(H0)

	# Begin optimization loop
	while epsilon[n] > epsilon0:
		# Problem 2 optimzation function
		f0 = 5.0*x0[0]**2 + 10.0*x0[1]**2 + 12.0*x0[0]*x0[1] - 8.0*x0[0] - 14.0*x0[1] + 5.0

		# Calculate gradient at current point
		g0 = gradx(x0)

		# Calculate next x values
		x = x0 - H0_1 @ g0

		# Update optimization function value
		f = f0 - 0.5*g0.T @ H0_1 @ g0

		# Update stoping criteria and determine next step
		n = n + np.uint16(1)
		epsilon[n] = np.linalg.norm(f - f0)
		x0 = x
		if n >= 1000:
			break

	# Print complete statement with # of iterations
	print('Newton Optimization Complete')
	print('# of Iterations: ' + str(n))
	return x, n, epsilon[1:n+1]


# Main program for problem 2
if __name__ == '__main__':
	# HW2 Problem 2

	# Gradient descent w/ inexact line search with t = 1.0
	x_GD, n_GD, eps_GD = GD_inexact(1.0, 1.0, np.array([[0.0], [0.0]]), 0.000001)
	x1_GD = -2.0*x_GD[0] - 3.0*x_GD[1] + 1

	print('t = ' + str(1.0))
	print('x1 = ' + str(x1_GD))
	print('x2 = ' + str(x_GD[0]))
	print('x3 = ' + str(x_GD[1]))

	# Newton's method
	x_N, n_N, eps_N = newton(np.array([[0.0], [0.0]]), 0.000001)
	x1_N = -2.0*x_N[0] - 3.0*x_N[1] + 1

	print('x1 = ' + str(x1_N))
	print('x2 = ' + str(x_N[0]))
	print('x3 = ' + str(x_N[1]))

	# Compare solutions of each method
	# Determine distance of each solution to desired point (-1, 0, 1)^T
	d_GD = np.sqrt((x1_GD + 1.0)**2 + x_GD[0]**2 + (x_GD[1] - 1.0)**2)
	d_N = np.sqrt((x1_N + 1.0)**2 + x_N[0]**2 + (x_N[1] - 1.0)**2)
	print('Gradient Descent Distance = ' + str(d_GD))
	print('Newtons Method Distance = ' + str(d_N))

	# Plot method convergence
	fig = plt.figure(num=1)
	plt.semilogy(np.linspace(1, n_GD, n_GD), eps_GD, 'bD--', np.linspace(1, n_N, n_N), eps_N, 'rD--')
	plt.grid(b=True, which='both', axis='y')
	plt.title('t = 1.0')
	plt.ylabel('error')
	plt.xlabel('# of Iterations')
	fig.legend(['Gradient Descent', 'Newtons Method'])

	# Gradient descent w/ inexact line search with t = 0.5
	x_GD, n_GD, eps_GD = GD_inexact(0.5, 1.0, np.array([[0.0], [0.0]]), 0.000001)
	x1_GD = -2.0 * x_GD[0] - 3.0 * x_GD[1] + 1

	print('t = ' + str(0.5))
	print('x1 = ' + str(x1_GD))
	print('x2 = ' + str(x_GD[0]))
	print('x3 = ' + str(x_GD[1]))

	# Determine distance of each solution to desired point (-1, 0, 1)^T
	d_GD = np.sqrt((x1_GD + 1.0) ** 2 + x_GD[0] ** 2 + (x_GD[1] - 1.0) ** 2)
	print('Gradient Descent Distance = ' + str(d_GD))

	# Plot method convergence
	fig = plt.figure(num=2)
	plt.semilogy(np.linspace(1, n_GD, n_GD), eps_GD, 'bD--', np.linspace(1, n_N, n_N), eps_N, 'rD--')
	plt.grid(b=True, which='both', axis='y')
	plt.title('t = 0.5')
	plt.ylabel('error')
	plt.xlabel('# of Iterations')
	fig.legend(['Gradient Descent', 'Newtons Method'])

	# Gradient descent w/ inexact line search with t = 0.3
	x_GD, n_GD, eps_GD = GD_inexact(0.3, 1.0, np.array([[0.0], [0.0]]), 0.000001)
	x1_GD = -2.0 * x_GD[0] - 3.0 * x_GD[1] + 1

	print('t = ' + str(0.3))
	print('x1 = ' + str(x1_GD))
	print('x2 = ' + str(x_GD[0]))
	print('x3 = ' + str(x_GD[1]))

	# Determine distance of each solution to desired point (-1, 0, 1)^T
	d_GD = np.sqrt((x1_GD + 1.0) ** 2 + x_GD[0] ** 2 + (x_GD[1] - 1.0) ** 2)
	print('Gradient Descent Distance = ' + str(d_GD))

	# Plot method convergence
	fig = plt.figure(num=3)
	plt.semilogy(np.linspace(1, n_GD, n_GD), eps_GD, 'bD--', np.linspace(1, n_N, n_N), eps_N, 'rD--')
	plt.grid(b=True, which='both', axis='y')
	plt.title('t = 0.3')
	plt.ylabel('error')
	plt.xlabel('# of Iterations')
	fig.legend(['Gradient Descent', 'Newtons Method'])
