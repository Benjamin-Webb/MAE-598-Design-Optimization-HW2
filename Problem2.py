# MAE 598 - Design Optimization HW # 2
# Benjamin Webb
# 9/16/2022

import numpy as np

def gradx(x, x0):
	# Calculates gradient for problem 2
	# given input vector x and x0
	# x = current point
	# x0 = target point

	g = np.array([[8*x[0] + 4*x0[0]], [18*x[1] + 6*x0[1]]])

	return g

def hessian():
	# Generates Hessian for problem 2
	# Hessian is constant for this problem

	H = np.array([[8.0, 0.0], [0.0, 18.0]])

	return H

# Main program for problem 2
if __name__ == '__main__':
	# HW2 Problem 2

	x = np.array([[0.0], [0.0]])
	x0 = np.array([[0.0], [1.0]])
	g0 = gradx(x, x0)
	H0 = hessian()
	print(H0)
