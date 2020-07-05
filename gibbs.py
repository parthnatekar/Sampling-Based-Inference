import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt

class Gibbs():

	def __init__(self, size, iterations):
		self.size = size
		self.iterations = iterations
		self.grid = np.random.randint(2, size=(size, size))
		print(self.grid)

	def gibbs(self): 

		history = np.array(self.grid)
		count = [np.count_nonzero(self.grid)]

		for i in range(self.iterations):
			for j in range(self.size**2):
				idx = np.unravel_index(j, (self.size, self.size))
				self.grid[idx] = np.random.choice([0,1], p=self.conditional(j)[0])	
				# print(self.grid)
				history = np.dstack((history,self.grid))
				count.append(np.count_nonzero(self.grid))
		print(self.grid)
		return(history, count)

	def conditional(self, idx):

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
		# print(neighbours)
		return(self.kernel(idx, neighbours), neighbours)

	def kernel(self, idx, neighbours):

		for idx in neighbours:
			if self.grid[np.unravel_index(idx, (self.size, self.size))] == 1:
				return([1, 0])
		return([0.5,0.5])

if __name__ == "__main__":
	size = 10
	plt.figure()
	
	# for i in range(4):
	# 	G = Gibbs(size, 10)
	# 	history, count = G.gibbs()
	# 	plt.subplot(2,2,i+1)
	# 	plt.xticks([])
	# 	plt.yticks([])
	# 	plt.imshow(history[:,:,-1], cmap=plt.cm.RdBu_r)
		
	# plt.show()
	# print(history.shape)
	# print(count)

	G = Gibbs(size, 10)
	history, count = G.gibbs()
	# count_adj = []
	# for i in range(len(history[0,0,:])):
	# 	count_adj_t = 0
	# 	for j in range(size**2):
	# 		idx = np.unravel_index(j, (size, size))
	# 		_, neighbours = G.conditional(j)
	# 		for nbr in neighbours:
	# 			nbr = np.unravel_index(nbr, (size, size))
	# 			if history[:, :, i][idx] == 1 and history[:, :, i][nbr] == 1:
	# 				count_adj_t += 0.5
	# 	count_adj.append(count_adj_t)

	# print(count_adj)
	# # print(np.diff(history[:,:,0]), np.diff(history[:,:,0], axis=0))
	# plt.plot(np.linspace(0, len(count_adj), len(count_adj)), count_adj)
	# plt.xlabel("Iteration")
	# plt.ylabel("Number of adjacent zeros")
	# plt.show()
	# count_random = []
	# for i in range(1000):
	# 	grid_random = np.random.randint(2, size=(size, size))
	# 	count_random.append(np.count_nonzero(grid_random))

	# for i in range(10):
	# 	print(np.mean(count[i*100:(i+1)*100]), np.mean(count_random[i*100:(i+1)*100]))
		