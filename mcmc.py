import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt

class MCMC():

	def __init__(self, iterations, joint, burn, sigma=1, plot=False):
		self.x_initial = 1
		self.iterations = iterations
		self.joint = joint
		self.burn = burn
		self.sigma=sigma
		self.plot=plot

	def MH(self):		

		x_list = [self.x_initial]
		alpha_list = []

		for i in range(self.iterations):

			x_new = self.x_initial + np.random.normal(0, self.sigma)

			alpha = min(1, self.joint.pdf(x_new)/self.joint.pdf(self.x_initial))

			alpha_list.append(alpha)

			if np.random.uniform(0, 1) < alpha:

				self.x_initial = x_new

			if i > 0:

				x_list.append(self.x_initial)

		if self.plot == True:
			plt.figure()
			x_plot = np.linspace(0, 30, 100)
			plt.plot(x_plot, self.joint.pdf(x_plot), label = 'Gamma Density')
			plt.hist(x_list, bins = 'auto', density=True, label='Sampled Distribution')
			# plt.plot(x_list)
			# plt.xlabel("Iteration")
			# plt.ylabel("Sample Value")
			# print(np.mean(alpha_list[:self.burn]), np.mean(alpha_list[self.burn:]))
			# plt.legend()
			plt.show()
		return(x_list)
		
if __name__ == '__main__':

	# plt.rcParams.update({'font.size': 15})
	# plt.figure(figsize=(40, 10))

	# for i,sigma in enumerate([0.5, 1, 2, 5]):
	# 	M = MCMC(1750, gamma(a=1, scale=5.5), burn=750, sigma=sigma)

	# 	samples = M.MH()

	# 	plt.subplot(2, 2, i+1)
	# 	plt.plot(samples)
	# 	plt.xlabel("Iteration")
	# 	plt.ylabel("Sample Value")
	# 	plt.title("Variance = {}".format(sigma))

	# plt.show()
	# plt.savefig("/home/parth/Projects/PGM/MCMC/Burn_slide.png")
		# M = MCMC(1, gamma(1, 5.5))
	# M.gibbs(5)

	M = MCMC(10000, gamma(a=3, scale=2), burn=750, sigma=1, plot=True)
	_ = M.MH()

		





		
