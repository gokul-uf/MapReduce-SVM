import numpy as np
np.random.seed(1234) # TODO Vary this, should not change the outcome by much

# CONSTANTS, can be varied. Oh, the irony!
d_input = 400
d_transform = 50
lamda = 0.01

# RFF matrices
W = np.random.randn(d_input, d_transform) # TODO Check uniqueness? 
b = np.random.uniform(0, 2*np.pi, (1,d_transform)) # TODO Check uniqueness? 

def transform(X):
	return np.cos(np.dot(X, W) + b) # np.dot handles both 1D and 2D

def fit(x, y, update, lamda): # TODO Maybe introduce minibatches?
	# print "In fit"
	for i in range(len(x)):
		alpha = 1.0/(lamda * (i+1))
		if y[i] * np.dot(x[i], update.T) < 1:
			update = (1 - alpha*lamda)*update + alpha*y[i]*x[i]
		else:
			update = (1 - alpha*lamda)*update
	return update

def mapper(key, value):
	print "In mapper"
	value_batch_size = len(value)
	value = [ [float(i) for i in v.strip().split()]for v in value]
	value =  np.asarray(value)
	# print value.shape
	x = value[:, 1:] # x.shape = (80k, 400)
	# print "after x"
	y = value[:, 0]  # y.shape = (80k, 1)
	# print "after y"
	x = transform(x) # Transform using the RFF, x.shape = (80k, 50)
	bias = 	np.ones((value_batch_size, 1))
	# Add the bias 
	# print "X shape: {}".format(x.shape)
	# print "bias shape: {}".format(bias.shape)
	x = np.hstack((x, bias)) # x.shape = (80k, 51)
	assert x.shape[1] == d_transform + 1 # Make sure the bias are appended correctly
	update = np.random.randn(1, d_transform + 1) # TODO Maybe use np.zeros?
	update = fit(x, y, update, lamda)
	# print "Update dims: {}".format(update.shape)
	yield "", update

def reducer(key, value):
	print "In reducer"
	output = value[0]
	print type(value[0])
	for v in value[1:]:
		output += v
	output = (1.0 / len(value))*output
	output = output.flatten()
	# print output.shape
	# print output
	yield "", output
		
