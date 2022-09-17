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

	f0 = np.zeros((1, 1))
	f = np.zeros((1, 1))
	phi = np.zeros((1, 1))
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
			f0 = f
			x0 = x
			alpha = 1.0
			n = n + 1

	# Print complete statement with # of iterations
	print('Gradient Descent Optimization Complete')
	print('# of Iterations: ' + str(n))
	return x

def newton(x0, epsilon0):
	# Function performs Newton's method for optimization of problem 2
	# x0: initial guess
	# epsilon0: stopping criteria

	# Initialize stopping criteria
	epsilon = 1.0
	n = np.uint16(1)

	f0 = np.zeros((1, 1))
	f = np.zeros((1, 1))
	# Begin optimization loop
	while epsilon > epsilon0:
		# Problem 2 optimzation function
		f0 = 5.0*x0[0]**2 + 10.0*x0[1]**2 + 12.0*x0[0]*x0[1] - 8.0*x0[0] - 14.0*x0[1] + 5.0

		

# Main program for problem 2
if __name__ == '__main__':
	# HW2 Problem 2

	x = GD_inexact(1.0, 1.0, np.array([[0.0], [0.0]]), 0.000001)
	x1 = -2.0*x[0] - 3.0*x[1] + 1

	print('x1 = ' + str(x1))
	print('x2 = ' + str(x[0]))
	print('x3 = ' + str(x[1]))
