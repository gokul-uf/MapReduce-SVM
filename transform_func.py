import numpy as np

# CONSTANTS, can be varied. Oh, the irony!
np.random.seed(123456789)
d_input = 400
d_transform = 100
lamda = 0.0001

# RFF matrices
# W = np.random.randn(d_input, d_transform) # TODO Check uniqueness? 
W = np.random.standard_cauchy((d_input, d_transform))
b = np.random.uniform(0, 2*np.pi, (1,d_transform)) # TODO Check uniqueness? 

def transform(X):
	value_batch_size = len(X)
	# bias = np.ones((value_batch_size, 1))
	# X = np.hstack((X, bias)) # append the bias at the very end
	X = np.sqrt(2.0 / d_transform)*np.cos(np.dot(X, W) + b)
	return X

def fit(x, y, update, lamda): # TODO Maybe introduce minibatches?
	for i in range(len(x)): # Iterate through the data points
		alpha = 1.0/(lamda * (i+1)) # TODO Find another strategy to update alpha?
		if y[i] * np.dot(x[i], update.T) < 1:
			update = (1 - alpha*lamda)*update + alpha*y[i]*x[i]
		else:
			update = (1 - alpha*lamda)*update
	return update

def mapper(key, value):
	value = [[float(i) for i in v.strip().split()]for v in value]
	value =  np.asarray(value)
	x = value[:, 1:]
	y = value[:, 0] 
	x = transform(x)
	update = np.random.randn(1, d_transform) 	
	# update = np.zeros((1, d_transform))
	update = fit(x, y, update, lamda)
	yield "", update

def reducer(key, value):
	output = value[0]
	for v in value[1:]:
		output += v
	output = (1.0 / len(value))*output
	output = output.flatten()
	yield output
		
