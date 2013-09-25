import gzip, cPickle
import numpy as np
import pylab as plt

def load_mnist():
	f = gzip.open('mnist.pkl.gz', 'rb')
	data = cPickle.load(f)
	f.close()
	return data

(x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()

def plot_digits(data, numcols, shape=(28,28)):
	numdigits = data.shape[0]
	numrows = int(numdigits/numcols)
	for i in range(numdigits):
		plt.subplot(numrows, numcols, i)
		plt.axis('off')
		plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
	plt.show()
    
plot_digits(x_train[0:8], numcols=4)

# j : class
# x : image for some datapoint
# b : 
# W : 
# Returns conditional probability of class label t given image x 
# for some datapoint.
def get_conditional_probability_of_class(j, x, b, W):
	b_j = 0 # TODO
	w_j = W[j]
	q_j = get_unnormalized_probability_of_class(j, x, w_j, b_j)
	Z = get_normalizing_factor(q)
	return q_j / Z

# Returns Z
def get_normalizing_factor(q):
	Z = 0
	for i in q:
		Z = Z + q
	return Z

# j   : class
# x   : image for some datapoint
# w_j : j-th column of matrix W
# b_j : j-th element of vector w
# Returns [q_j] the unnormalized probability of the class j (q_j)
def get_unnormalized_probability_of_class(j, x, w_j, b_j):
	q_j = exp( np.transpose(w_j).dot(x) + b_j )
	return q_j

# Task 1.1.2
# x : datapoint
# t : class
# Returns [q_j] the gradient wrt w & b of the log-likelihood for x,t
def logreg_gradient(x, t, w, b):
	return 0 # TODO

# Task 1.1.3
def sgt_iter(x_train, t_train, w, b):
	learning_rate = 1E-4
	return None # TODO