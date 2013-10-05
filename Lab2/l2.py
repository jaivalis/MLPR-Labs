import gzip, cPickle
import numpy as np
import pylab as plt
from math import exp, log1p

def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	data = cPickle.load(f)
	f.close()
	return data

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()

def plot_digit(digit) :
    plt.imshow(digit.reshape(28,28), interpolation='nearest', cmap='Greys')
    plt.show()

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()




# x is a vector: 784         (datapoint)
# t is a scalar              (corresponding class)
# W is a matrix: 784 x 10    (weights)
# b is a vector: 10          (bias)
#
# returns a matrix: 784 x 10 (partial derivative of log likelihood wrt w)
# returns a vector: 10       (partial derivative of log likelihood wrt b)
def logreg_gradient(x, t, W, b): # log_Q -> Z -> log_P -> delta
    log_Q = []
    log_P = []
    partial_derivative_logLikelihood_b = []
    partial_derivative_logLikelihood_W = []

    Z = 0.0 # Z = normalizing factor
    for j in range(0,10):
        log_q = np.dot(np.transpose(W)[j], x) + b[j]
        log_Q.append(log_q)
        Z += exp(log_Q[j])

        log_P.append(log_Q[j] - log1p(Z))
                
        if j == t:
            partial_derivative_logLikelihood_b.append(1 - exp(log_Q[j])/Z)
        else:
            partial_derivative_logLikelihood_b.append(exp(log_Q[j])/Z)
        
    
    partial_derivative_logLikelihood_W = np.outer(x, partial_derivative_logLikelihood_b)
    
    return partial_derivative_logLikelihood_W, partial_derivative_logLikelihood_b
    
# x_train is a matrix: 50000 x 784 (dataset)
# t_train is a vector: 50000       (class of x)
# W       is a matrix: 784 x 10    (weights)
# b       is a vector: 10          (bias for each class)
def sgd_iter(x_train, t_train, W, b):
    alpha = 1E-4      # learning rate
    
    x_trainIndex = np.arange(len(x_train), dtype = int)
    np.random.shuffle(x_trainIndex) # shuffle indices

    for xIndex in x_trainIndex:
        x = x_train[xIndex]
        t = t_train[xIndex]
        
        W = W + alpha * logreg_gradient(x, t, W, b)[0]
        
    return W

# x is a vector: 784         (datapoint)
# t is a scalar: 1           (class of x)
# W is a matrix: 784 x 10    (weights)
# b is a vector: 10          (bias for each class)
# returns conditional log probability(scalar)
def get_log_P(x, t, W, b):
    log_Q = []
    for i in range(0,10):
        log_q = np.dot(np.transpose(W)[i], x) + b[i]
        log_Q.append(log_q)
    Z = 0.0
    for i in range (0, 10): # Z = normalizing factor        
        Z += exp(log_Q[i])
    log_p = log_Q[t] - log1p(Z)
    return log_p

def plot_training(handful, x_train, t_train, x_valid, t_valid):
    W = np.zeros((784,10))
    b = np.zeros(10)
    
    point = 0
    for i in range(0, handful):
        W = sgd_iter(x_train, t_train, W, b)
    
        for xIndex in range(0, len(x_valid)):
            x = x_valid[xIndex]
            t = t_valid[xIndex]
            log_P = get_log_P(x, t, W, b)
            if i%1000 == 0:
                plt.plot(point, log_P, 'o')
            point += 1
    plt.show()
    return W
    
def visualize_weights(W):
    plot_digits(W, 5, shape=(28,28))

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
W = plot_training(1, x_train, t_train, x_valid, t_valid)
visualize_weights(np.transpose(W))