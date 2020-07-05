import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
from tqdm import tqdm

class Ising():

	def __init__(self, size, iterations, J=1, ext_field=0):
		self.size = size
		self.iterations = iterations
		self.grid = np.random.choice([-1,1], size=(size, size))
		self.J=J
		self.mask = np.ones((self.size, self.size))
		self.ext_field = ext_field

	def ising_step(self, temperature = 1):
		self.beta = 1/temperature
		for i in range(self.iterations):
			for j in range(self.size**2):
				self.ising_update(j)
		self.E = 0
		for m in range(self.size**2):
			neighbours = self.neighbours(m)
			for n in neighbours:
				self.E += -0.5 * self.J * self.grid[np.unravel_index(m, (self.size, self.size))] \
				* self.grid[np.unravel_index(n, (self.size, self.size))]		
		self.E /= self.size**2
		self.m = (np.sum(self.grid)/self.size**2)**2

	def ising_update(self, j):

		local_field = 0
		idx = np.unravel_index(j, (self.size, self.size))
		for neighbour in self.neighbours(j):
			local_field += self.J*self.grid[np.unravel_index(neighbour, (self.size, self.size))]
		local_field += self.ext_field
		dE = 2 * local_field * self.grid[idx]
		if dE <= 0:
			self.grid[idx] *=-1
		elif np.random.rand() < np.exp(-dE * self.beta):
			self.grid[idx] *=-1

	def neighbours(self, idx):

		neighbours = []
		if idx+self.size < self.size**2:
			neighbours.append(idx+self.size)
		if idx-self.size > 0:
			neighbours.append(idx-self.size)
		try:
			if np.unravel_index(idx-1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx-1)
		except:
			pass
		try:
			if np.unravel_index(idx+1, (self.size,self.size))[0] ==  np.unravel_index(idx, (self.size,self.size))[0]:
				neighbours.append(idx+1)
		except:
			pass
		return(neighbours)

if __name__ == '__main__':

	T = np.full(len(np.linspace(0.1, 15, 150))*2, 0.1)

	for i,val in enumerate(np.linspace(0.1, 15, 150)):
		T[i*2] = val

	I = Ising(16, 1, J=1)

	E = []
	m = []

	I.ising_step()

	plt.figure(figsize=(10,20))
	plt.subplot(1, 2, 1)
	plt.title("Original State")
	plt.imshow(I.grid, cmap='Greys', vmin=-1., vmax=1.)
	I.ising_step()
	plt.subplot(1, 2, 2)
	plt.title("Equilibrium State at T = {}".format(1))
	plt.imshow(I.grid, cmap='Greys', vmin=-1., vmax=1.)
	plt.show()
	print(I.E, I.m)

	# iterations = 10
	# E = np.zeros((iterations, 150))
	# m = np.zeros((iterations, 150))

	# for i in tqdm(range(iterations)):
	# 	for j,t in tqdm(enumerate(T)):
	# 		I.ising_step(t)
	# 		if i%2==0:
	# 			E[i, j//2] = I.E
	# 			m[i, j//2] = I.m

	# print(E, m)

	# np.save('Energy.npy', E)
	# np.save('Magnetization.npy', m)

