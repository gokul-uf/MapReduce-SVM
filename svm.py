def train(x, y, alpha,lmbd,ww):
    w=ww
    if y * WTX(x,w) < 1:
        for i in xrange(len(x)):
            w[i] = (1. - alpha * lmbd) * w[i] + alpha * y * x[i]
    else:
        for i in xrange(len(x)):
            w[i] = (1. - alpha * lmbd) * w[i]
    return w

def WTX(x,w):
    wTx = 0.
    for i in xrange(len(x)):
        wTx += w[i] * x[i]
        return wTx

def fit(X,Y,lmbd):
    X=transform(X)
    #The additional element is to add bias or intercept
    w=[0.] * 401
    for i in range(len(X)):
        t=i
        x=X[i]+[1]
        target=Y[i]
        alpha = 1. / (lmbd * (t + 1.))
        w=train(x, target, alpha,lmbd,w)
    return w


def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return (X)


def mapper(_, value):
    X=[]
    Y=[]
    for v in value:
        Y.append(float(v.split(" ")[0]))
        vals_v=[]
        for i in v.split(" ")[1:]:
            vals_v.append(float(i))
        X.append(vals_v)
    yield "", fit(X,Y,0.1)


def reducer(_, values):
    final=[]
    for i in range(400):
        sum=0
        for v in values:
            sum=sum+v[i]
        final.append(sum/400)
    yield final
