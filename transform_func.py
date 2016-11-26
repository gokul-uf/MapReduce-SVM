import numpy as np

d_input = 400
d_transform = 50 # TODO Vary this

W = np.random.randn(d_input, d_transform) # TODO Check uniqueness? 
b = np.random.uniform(0, 2*np.pi, (1,d_transform)) # TODO Check uniqueness? 

def transform(X):
	# np.dot handles both 1D and 2D
	return np.cos(np.dot(X, W) + b)

def train(x, y, alpha,lmbd,w):
    if y * np.dot(w.T, x) < 1: #if W'x . y < 1, then it misclassified
        for i in xrange(len(x)):
            w[i] = (1. - alpha * lmbd) * w[i] + alpha * y * x[i]
    else:
        for i in xrange(len(x)):
            w[i] = (1. - alpha * lmbd) * w[i]
    return w

def fit(X,Y,lmbd):
    X=transform(X)
    #The additional element is to add bias or intercept
    w=[0.] * 401 # weight vector initialized to 0, to train
    for i in range(len(X)): # iterate through the train samples
        t=i + 1.
        x=X[i]+[1]
        target=Y[i]
        alpha = 1. / (lmbd * t)
        w=train(x, target, alpha,lmbd,w)
    return w

def mapper(key, value):
    X= transform(value[:, 1:])
    Y= value[:, 0].ravel()
    yield "", fit(X,Y,0.1)

def fit(x, y, update, alpha, lamda): # TODO other parameters?
	for i in range(len(x)):
		update = (1 - alpha*lamda)
		if y[i] * np.dot(x[i], u.t) < 1:
			update = 
	return update

def mapper(key, value):
	x = value[:, 1:] # x.shape = (80k, 400)
	y = value[:, 0] # y.shape = (80k, 1)
	x = transform(x) # Transform using the RFF, x.shape = (80k, 50)
	bias = 	np.ones((80000,1)) 
	x = np.hstack((x, bias)) # Add the bias, # x.shape = (80k, 51)
	assert x.shape[1] == d_transform + 1 # Make sure the bias are appended correctly
	update = np.random.randn((1, d_transform + 1)) # TODO Maybe use np.zeros?
	update = fit(x, y, update, alpha, lamda) # TODO Any other parameters? lambda, alpha values?
