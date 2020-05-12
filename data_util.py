# Copyright 2020 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# data_util.py

import numpy as np
import pickle
import os

def load_CIFAR_batch(filename):
  """
  Load a batch of images from the CIFAR10 python dataset
  """
  fh = open(filename, 'rb')
  data_dict = pickle.load(fh, encoding='latin1')
  X = data_dict['data'].reshape(10000,3,32,32).transpose(0,2,3,1)
  X = X.astype("float")/255.0
  Y = np.array(data_dict['labels'])
  fh.close()
  return X, Y

def load_CIFAR10(data_dir):
  """
  Load entire CIFAR10 python dataset
  """
  X_list = []
  Y_list = []
  for b in range(1,6):
    filename = os.path.join(data_dir, 'data_batch_%d' % (b, ))
    X_b, Y_b = load_CIFAR_batch(filename)
    X_list.append(X_b)
    Y_list.append(Y_b)
  X_train = np.concatenate(X_list)
  Y_train = np.concatenate(Y_list)
  X_test, Y_test = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
  return X_train, Y_train, X_test, Y_test

def get_CIFAR10_data(num_train=49000, num_valid=1000, num_test=1000):
  """
  Load CIFAR10 dataset and assign train, test and val splits
  (total training data = 50k, test = 10k)
  """
  data_dir = 'data/cifar-10-batches-py'
  X_train, Y_train, X_test, Y_test = load_CIFAR10(data_dir)

  X_val = X_train[num_train:(num_train+num_valid)]
  Y_val = Y_train[num_train:(num_train+num_valid)]
  X_train = X_train[0:num_train]
  Y_train = Y_train[0:num_train]
  X_test = X_test[0:num_test]
  Y_test = Y_test[0:num_test]

  return X_train, Y_train, X_val, Y_val, X_test, Y_test
