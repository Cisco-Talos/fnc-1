from theano import function
import numpy as np
# The prediction function for pylearn2.models.*
def predict(model, x, U = 1000):
    a = model.get_input_space().make_theano_batch()
    b = model.fprop(a)
    f = function([a], b)
    n = x.shape[0]
    yhat = []
    for i in range(n // U + 1):
        yhat.append(f(x[(i * U):((i + 1) * U),:]))
    return np.vstack(yhat)

