# MAE 598 - Design Optimization HW # 2
# Benjamin Webb
# 9/16/2022

import numpy as np


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
	epsilon = 1.0
	n = np.uint16(1)

	# Begin gradient descent loop
	while epsilon > epsilon0:
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
			n = n + 1
		elif n >= 10000:
			break
		else:
			epsilon = np.linalg.norm(f - f0)
			x0 = x
			alpha = 1.0
			n = n + np.uint16(1)

	# Print complete statement with # of iterations
	print('Gradient Descent Optimization Complete')
	print('# of Iterations: ' + str(n-1))
	return x

def newton(x0, epsilon0):
	# Function performs Newton's method for optimization of problem 2
	# x0: initial guess
	# epsilon0: stopping criteria

	# Initialize stopping criteria
	epsilon = 1.0
	n = np.uint16(1)

	# Calculate Hessian once, since it is constant over domain of this function
	H0 = hessian()
	H0_1 = np.linalg.inv(H0)

	# Begin optimization loop
	while epsilon > epsilon0:
		# Problem 2 optimzation function
		f0 = 5.0*x0[0]**2 + 10.0*x0[1]**2 + 12.0*x0[0]*x0[1] - 8.0*x0[0] - 14.0*x0[1] + 5.0

		# Calculate gradient at current point
		g0 = gradx(x0)

		# Calculate next x values
		x = x0 - H0_1 @ g0

		# Update optimization function value
		f2 = 5.0*x[0]**2 + 10.0*x[1]**2 + 12.0*x[0]*x[1] - 8.0*x[0] - 14.0*x[1] + 5.0
		f = f0 - 0.5*g0.T @ H0_1 @ g0

		# Update stoping criteria and determine next step
		epsilon = np.linalg.norm(f - f0)
		x0 = x
		n = n + np.uint16(1)
		if n >= 10000:
			break

	# Print complete statement with # of iterations
	print('Newton Optimization Complete')
	print('# of Iterations: ' + str(n-1))
	return x


# Main program for problem 2
if __name__ == '__main__':
	# HW2 Problem 2

	# Gradient descent w/ inexact line search
	x_GD = GD_inexact(1.0, 1.0, np.array([[0.0], [0.0]]), 0.000001)
	x1_GD = -2.0*x_GD[0] - 3.0*x_GD[1] + 1

	print('x1 = ' + str(x1_GD))
	print('x2 = ' + str(x_GD[0]))
	print('x3 = ' + str(x_GD[1]))

	# Newton's method
	x_N = newton(np.array([[0.0], [0.0]]), 0.000001)
	x1_N = -2.0*x_N[0] - 3.0*x_N[1] + 1

	print('x1 = ' + str(x1_N))
	print('x2 = ' + str(x_N[0]))
	print('x3 = ' + str(x_N[1]))
