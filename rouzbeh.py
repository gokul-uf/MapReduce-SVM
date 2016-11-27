import numpy as np

# CONSTANTS, can be varied. Oh, the irony!
np.random.seed(123456789)
d_input = 401
d_transform = 500
num_mini_batch = 1000
mini_batch_size = 500
# lamda = 0.00000000000000000000000000000000000000000000000000000000001
# lamda = 0.00000000000000000000000000000000000000000000001
lamda = 0.0000000000000000000000000000000000000001

# RFF matrices
W = np.asarray(np.random.standard_cauchy((d_input-1, d_transform)), dtype = np.float32) # TODO Check uniqueness?
b = np.asarray(np.random.uniform(0, 2*np.pi, (1,d_transform)), dtype = np.float32) # TODO Check uniqueness? 

def transform(X):
	value_batch_size = len(X)
	X =10* np.sqrt(2.0 / d_transform)*np.cos(np.dot(X, W) + b)
	bias = np.ones((value_batch_size, 1), dtype = np.float32)
	X = np.hstack((X, bias)) # append the bias at the very end
	return X

def fit(x, y, update, lamda): 
	for i in range(len(x)): 
		alpha = 1.0/(lamda * (i+1)) 
		if y[i] * np.dot(x[i], update.T) < 1:
			update = (1 - alpha*lamda)*update + alpha*y[i]*x[i]
		else:
			update = (1 - alpha*lamda)*update
	return update

def minibatch_fit(x, y, update, lamda):
	for i in range(num_mini_batch):
		# alpha = 1.0 / (lamda*(i+1))
		samples = np.random.choice(x.shape[0], mini_batch_size, replace=False)
		x_minibatch = x[samples, :]
		y_minibatch = y[samples]
		temp_update = np.zeros(update.shape)
		for j in range(mini_batch_size):
			alpha = 1.0/(lamda * (j+1))
			if y_minibatch[j] * np.dot(x_minibatch[j], update.T) < 1:
				temp_update += (1.0 / mini_batch_size)*((1 - alpha*lamda)*update + alpha*y_minibatch[j]*x_minibatch[j])
			else:
				temp_update += (1.0 / mini_batch_size)*(1 - alpha*lamda)*update
		update += (1.0 / num_mini_batch)*temp_update
	return update

def mapper(key, value):
	value = [[float(i) for i in v.strip().split()]for v in value]
	value =  np.asarray(value, dtype=np.float32)
	x = value[:, 1:]
	y = value[:, 0] 
	x = transform(x)
	# update = np.random.randn(1, d_transform + 1)
	# update = np.asarray(update, dtype = np.float32) 	
	# update = np.zeros((1, d_transform+1), dtype=np.float32)
	update = np.zeros((1, d_transform+1))
	# update = fit(x, y, update, lamda)
	update = minibatch_fit(x, y, update, lamda)
	yield "", update

def reducer(key, value):
	yield np.mean(np.array(value), axis = 0).flatten()