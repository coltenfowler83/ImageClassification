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

# im_util.py

import matplotlib.pyplot as plt
import numpy as np

def remove_ticks(ax):
  """
  Remove axes tick labels
  """
  ax.set_xticklabels([])
  ax.set_yticklabels([])
  ax.set_xticks([])
  ax.set_yticks([])

def plot_classification_examples(Y_hat,Y_test,im,names):
  """
  Plot sample images with predictions Y_hat and true labels Y_test
  """
  fh = plt.figure()
  num_test=Y_test.size
  for i in range(10):
    r = np.random.randint(num_test)
    ax=plt.subplot(1,10,i+1)
    remove_ticks(ax)
    lh=plt.xlabel(names[Y_hat[r]])
    if (Y_hat[r]==Y_test[r]):
      lh.set_color('green')
    else:
      lh.set_color('red')
    plt.imshow(im[r])

def plot_weights(W, names):
  """
  Plot images for each weight vector in W
  """
  fh = plt.figure()
  for i in range(10):
    W_im = np.reshape(W[:,i],(32,32,3))
    W_im = normalise_01(W_im)
    ax=plt.subplot(1,10,i+1)
    remove_ticks(ax)
    plt.xlabel(names[i])
    plt.imshow(W_im)

def normalise_01(im):
  """
  Normalise image to the range (0,1)
  """
  mx = im.max()
  mn = im.min()
  den = mx-mn
  small_val = 1e-9
  if (den < small_val):
    print('image normalise_01 -- divisor is very small')
    den = small_val
  return (im-mn)/den
