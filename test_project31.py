import numpy as np
import matplotlib.pyplot as plt
from time import time
import types
import data_util
import im_util

"""Load CIFAR10 data"""

num_classes=10
num_dims=32*32*3

cifar10_names=['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']

num_train=49000
num_valid=1000
num_test=10000

im_train,Y_train,im_valid,Y_valid,im_test,Y_test = data_util.get_CIFAR10_data(num_train,num_valid,num_test)

X_train=np.reshape(im_train,(num_train,num_dims))
X_valid=np.reshape(im_valid,(num_valid,num_dims))
X_test=np.reshape(im_test,(num_test,num_dims))

# edit this line to change the figure size
plt.rcParams['figure.figsize'] = (16.0, 10.0)
plt.rcParams['font.size'] = 16
# force auto-reload of import modules before running code

def linear_classify(X,W,Y):
  T_hat = np.dot(X,W)
  Y_hat = np.argmax(T_hat,1)
  accuracy = np.sum(Y_hat==Y)/np.size(Y)
  return Y_hat, accuracy

def one_hot(Y, num_classes):
    """convert class labels to one-hot vector"""
    num_train=Y.size
    T = np.zeros((num_train, num_classes))
    T[np.arange(num_train), Y]=1
    return T

"""Linear Classifier by Stochastic Gradient Descent"""

batch_size = 32
weight_decay = 0.01  # same as lambda
learning_rate = 0.01

num_epochs = 10
num_iterations = num_epochs * (int)(num_train / batch_size)

np.random.seed(42)
W = np.random.randn(num_dims, num_classes)

valid_acc_seq = []
iteration_seq = []
W_seq = []
W_sq_seq = []

summary_interval = 1000

for i in range(num_iterations):

    # FORNOW: random gradient
    grd = np.random.randn(num_dims, num_classes)
    dW = -grd
    """
    *************************************************************************************
    *** TODO: implement stochastic gradient descent for the regularized linear classifier
    *************************************************************************************

    Select a random batch of data and take a step in the direction of the gradient
    """

    batch_idx = np.random.choice(num_train, batch_size)
    X_batch = X_train[batch_idx]
    Y_batch = Y_train[batch_idx]
    Y_batch_hot = one_hot(Y_batch, num_classes)
    grd = np.dot(X_batch.T, (np.dot(X_batch, W) - Y_batch_hot))
    dW = -learning_rate * grd

    """
    *************************************************************************************
    """
    W = W + dW

    if (i % summary_interval == 0):
        _, valid_acc = linear_classify(X_valid, W, Y_valid)
        valid_acc_seq.append(valid_acc)
        iteration_seq.append(i)
        print(' valid acc =% .2f%%' % (100.0 * valid_acc))
        W_seq.append(W)
        W_sq_seq.append(np.sum(W ** 2))

# plot validation accuracy and weight trends
plt.rcParams['figure.figsize'] = (16.0, 6.0)

fig = plt.figure()
plt.grid(True)
plt.plot(iteration_seq, valid_acc_seq, 'r')
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.ylim(0, 0.5)
plt.legend(['valid'])

fig = plt.figure()
plt.grid(True)
plt.plot(iteration_seq, np.log(W_sq_seq))
plt.xlabel('iteration')
plt.ylabel('log |W|^2')

# compute test accuracy
Y_hat, test_acc = linear_classify(X_test, W, Y_test)
print('\ntest accuracy = %.2f%%' % (100.0 * test_acc))
im_util.plot_classification_examples(Y_hat, Y_test, im_test, cifar10_names)
im_util.plot_weights(W, cifar10_names)