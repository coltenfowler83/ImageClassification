import numpy as np
import matplotlib.pyplot as plt
from time import time
import types
import data_util
import im_util

# edit this line to change the figure size
plt.rcParams['figure.figsize'] = (16.0, 10.0)
plt.rcParams['font.size'] = 16
# force auto-reload of import modules before running code

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

"""Visualise average images"""

avg_im=[]

# FORNOW: use first image of each class
for i in range(10):
    j = next(k for k in range(num_train) if Y_train[k]==i)
    avg_im.append(im_train[j])

"""
************************************************************
*** TODO: write code to compute average image for each class
************************************************************

Compute the average image for each class and store in avg_im
"""
avg_im = []

for i in range(num_classes):
    avg_list = []
    for j in range(num_train):
        if Y_train[j] == i:
            avg_list.append(im_train[j])
    avg_arr = np.array(avg_list)
    mean_im = np.mean(avg_arr, axis=0)
    avg_im.append(mean_im)

"""
************************************************************
"""

for i in range(10):
    ax=plt.subplot(1,10,i+1)
    im_util.remove_ticks(ax)
    plt.xlabel(cifar10_names[i])
    plt.imshow(avg_im[i])

"""Nearest Mean Classifier"""

#FORNOW: random labels
Y_hat=np.random.randint(0,10,num_test)

"""
**********************************************
*** TODO: classify test data using mean images
**********************************************

Set the predictions Y_hat for the test set by finding the nearest mean image 
"""

for i in range(num_test):
    pred_dist, pred_label = float('inf'), -1
    for j in range(num_classes):
        dist = np.linalg.norm(im_test[i] - avg_im[j])
        if dist < pred_dist:
            pred_dist = dist
            pred_label = j
    Y_hat[i] = pred_label

"""
**********************************************
"""

nm_accuracy=np.sum(Y_hat==Y_test)/num_test
im_util.plot_classification_examples(Y_hat,Y_test,im_test,cifar10_names)

print('Nearest mean classifier accuracy = %.2f%%' % (100.0*nm_accuracy))

"""Nearest Neighbour Classifier"""

num_test_small=1000
X_test_small=X_test[0:num_test_small]
Y_test_small=Y_test[0:num_test_small]

#FORNOW: random labels
Y_hat=np.random.randint(0,10,num_test_small)


"""
*****************************************************
*** TODO: classify test data using nearest neighbours
*****************************************************

Set the predictions Y_hat for the test set using nearest neighbours from the training set
"""

def compute_distances(M1, M2):
    N1, num_dims = M1.shape
    N2, num_dims = M2.shape
    ATB = np.dot(M1, M2.T)
    AA = np.sum(M1 * M1, 1)
    BB = np.sum(M2 * M2, 1)
    return -2*ATB + np.expand_dims(AA, 1) + BB

dists = compute_distances(X_test_small, X_train)

for i in range(num_test_small):
    argmin = np.argmin(dists[i])
    Y_hat[i] = Y_train[argmin]


"""
*****************************************************
"""

nn_accuracy=np.sum(Y_hat==Y_test_small)/num_test_small
print('Nearest neighbour classifier accuracy =% .2f%%' % (100.0*nn_accuracy))