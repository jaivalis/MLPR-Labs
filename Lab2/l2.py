import gzip, cPickle
import numpy as np
import pylab as plt
from math import isnan
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
# W is a matrix: 10 x 784    (weights)
# b is a vector: 10          (bias)
#
# returns a matrix: 784 x 10 (partial derivative of log likelihood wrt w)
# returns a vector: 10       (partial derivative of log likelihood wrt b)
def logreg_gradient(x, t, W, b): # log_Q -> Z -> log_P -> delta
    log_Q = []
    log_P = []
    partial_derivative_logLikelihood_b = []
    partial_derivative_logLikelihood_W = []

    Z = 0 # Z = normalizing factor
    for j in range(0,10):
        log_q = np.dot(W[j], x) + b[j]
        log_Q.append(log_q)
        
        Z += np.exp(log_q)
            
        log_P.append(log_q - np.log(Z))
                
        if j == t:
            partial_derivative_logLikelihood_b.append(1 - np.exp(log_q)/Z)
        else:
            partial_derivative_logLikelihood_b.append(np.exp(log_q)/Z)

    partial_derivative_logLikelihood_W = np.outer(partial_derivative_logLikelihood_b,x)

    return partial_derivative_logLikelihood_W, np.array(partial_derivative_logLikelihood_b)
    
# x_train is a matrix: 50000 x 784 (dataset)
# t_train is a vector: 50000       (class of x)
# W       is a matrix: 10 x 784    (weights)
# b       is a vector: 10          (bias for each class)
def sgd_iter(x_train, t_train, W, b):
    alpha = 1E-4      # learning rate
    
    x_trainIndex = np.arange(len(x_train), dtype = int)
    np.random.shuffle(x_trainIndex) # shuffle indices

    for xIndex in x_trainIndex:
        x = x_train[xIndex]
        t = t_train[xIndex]
        
        tupl = logreg_gradient(x, t, W, b)
        W = W + alpha * tupl[0]
        b = b + alpha * tupl[1]
        
    return W, b

# x is a vector: 784         (datapoint)
# t is a scalar: 1           (class of x)
# W is a matrix: 784 x 10    (weights)
# b is a vector: 10          (bias for each class)
# returns conditional log probability(scalar)
def get_log_P(x, t, W, b):
    log_Q = []
    Z = 0
    for i in range(0,10):
        log_q = np.dot(W[i], x) + b[i]
        log_Q.append(log_q)
        
        Z = Z + np.exp(log_q) # Z = normalizing factor 
    log_p = log_Q[t] - np.log(Z)
#    if isnan(log_p):
#        print log_p
#        print Z
#        print log_Q
    return log_p

def plot_training(handful, x_train, t_train, x_valid, t_valid):
    W = np.zeros((10, 784))
    b = np.zeros(10)
    
    point = 0
    i = 0
    for i in range(0, handful):
        tupl = sgd_iter(x_train, t_train, W, b)
        W = tupl[0]
        b = tupl[1]
    
        for xIndex in range(0, len(x_valid)):
            x = x_valid[xIndex]
            t = t_valid[xIndex]
            log_P = get_log_P(x, t, W, b)
            if i%10 == 0:
                plt.plot(point, log_P, 'o')
            point += 1
            i += 1
        for xIndex in range(0, len(x_train)):
            if xIndex % 10:
                x = x_train[xIndex]
                t = t_train[xIndex]
                log_P = get_log_P(x, t, W, b)
                if i%10 == 0:
                    plt.plot(point, log_P, 'x')
                point += 1
                i += 1
    plt.show()

# unused 
def plot_training2(handful, x_train, t_train, x_valid, t_valid):
    W = np.zeros((10, 784))
    b = np.zeros(10)
    
    validAverages = []
    trainAverages = []
    for i in range(0, handful):
        print 'Training cycle #', i+1, ' started.'
        tupl = sgd_iter(x_train, t_train, W, b)
        W = tupl[0]
        b = tupl[1]
        print 'Training cycle #', i+1, ' complete.'
        avgProbValidation = 0
        for xIndex in range(0, len(x_valid)):
            if xIndex % 10:
                x = x_valid[xIndex]
                t = t_valid[xIndex]
                log_P = get_log_P(x, t, W, b)
                avgProbValidation += log_P / len(x_valid)
#        avgProbValidation /= len(x_valid)
        validAverages.append(avgProbValidation)
        print '~Added Validation #', i+1, ' to plot.'
        
        avgProbTraining = 0
        for xIndex in range(0, len(x_train)):
            if xIndex % 10:
                x = x_train[xIndex]
                t = t_train[xIndex]
                log_P = get_log_P(x, t, W, b)
                avgProbTraining += log_P / len(x_train)
#        avgProbTraining /= len(x_train)
        trainAverages.append(avgProbTraining)
        print '~Added Training #', i+1, ' to plot.'
        print 'traingAvg ', trainAverages
        print 'validAvg ', validAverages
        
    ind = np.arange(handful)  # the x locations for the groups
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, validAverages, width, color='r')
    rects2 = ax.bar(ind + width, trainAverages, width, color='y')
    
    # add some
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )
    ax.legend( (rects1[0], rects2[0]), ('Training', 'Validation') )

    plt.show()
    
def visualize_weights(W):
    plot_digits(W, 5, shape=(28,28))

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
plot_training(3, x_train, t_train, x_valid, t_valid)

#train_2_iters_n_viz()