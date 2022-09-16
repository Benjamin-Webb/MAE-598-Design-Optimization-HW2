# MAE 598 - Design Optimization HW # 2
# Benjamin Webb
# 9/16/2022

import numpy as np

def gradx(x, x0):
	# Calculates gradient for problem 2
	# given input vector x and x0
	# x = current point
	# x0 = target point

	g = np.zeros((2, 1))
	g[0] = 8*x[0] + 4*x0[0]
	g[1] = 18*x[1] + 6*x0[1]

	return g

def hessian():
	# Generates Hessian for problem 2
	# Hessian is constant for this problem

	H = np.array([[8.0, 0.0], [0.0, 18.0]])

	return H

def GD_inexact(t, alpha, x0, xt, epsilon):
	# Performs gradient desecent using inexact line search for problem 2
	# t: tuning parameter between (0, 1)
	# alpha: initial step-size
	# x0: initial starting point
	# xt: target point
	# epsilon: stopping criteria

	# Problem 2 minimize function ||x1 - x0||^2
	f0 = np.zeros((2, 1))
	f0[0] = 4.0*x0[0]**2 + 4.0*x0[0]*xt[0] + xt[0]**2
	f0[1] = 9.0*x0[1]**2 + 6.0*x0[1]*xt[1] + xt[1]**2

	# Calculate gradient at current point
	g0 = gradx(x0, xt)

	# Calculate f(x - alpha*g0, xt), where delta_x = -g0
	x = x0 - alpha*g0

	# For Testing
	print(x)


# Main program for problem 2
if __name__ == '__main__':
	# HW2 Problem 2

	x = np.array([[0.0], [0.0]])
	x0 = np.array([[0.0], [1.0]])

	GD_inexact(1.0, 1.0, x, x0, 0.000001)
