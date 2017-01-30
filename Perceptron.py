import numpy as np 
class Perceptron(object):

	def __init__(self, learning_rate=0.01, num_iterations=10):
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations

	def fit(self, X, y):
		"""
		X : 2-D matrix = n by m, n = number of samples, m = number of features
		y : list = n, n = number of samples
		"""
		matrix_dimensions = X.shape
		#First slot holds weight of threshold, others hold weight of input
		self.weights = np.zeros(1 + matrix_dimensions[1]) 
		self.errors = []

		for _ in range(self.num_iterations):
			for row_sample, target in zip(X,y):
				update_value = self.learning_rate * (target - self.predict(row_sample))
				#Using less than here only to avoid floating point comparison
				if update_value > 0.0:
					self.errors.append(1) 
				self.weights[0] += update_value
				#Np arrays allows for easy vectorized operation as shown here 
				self.weights[1:] += update_value * row_sample #row_sample contains the values for n features of this sample
		return self

	def net_input(self):
		return self.weights[0] + np.dot(X, self.weights[1:])

	def predict(self):
		return np.where(np.net_input(X) >= 0.0, 1, -1) 
