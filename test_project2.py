import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# edit this line to change the figure size
plt.rcParams['figure.figsize'] = (16.0, 10.0)
plt.rcParams['font.size'] = 16
# may be needed to avoid mulitply defined openmp libs

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Load CIFAR10 dataset
(train_images0,train_labels0),(test_images,test_labels) = keras.datasets.cifar10.load_data()

# Normalise images
train_images0=train_images0.astype('float')/255.0
test_images=test_images.astype('float')/255.0

# Create a validation set
num_valid=1000
valid_images=train_images0[0:num_valid]
valid_labels=train_labels0[0:num_valid]
train_images=train_images0[num_valid:]
train_labels=train_labels0[num_valid:]

cifar10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
num_classes=10
num_train=train_labels.size
num_valid=valid_labels.size
num_test=test_labels.size

# Make one-hot targets
train_one_hot=tf.one_hot(train_labels[:,0],num_classes)
valid_one_hot=tf.one_hot(valid_labels[:,0],num_classes)
test_one_hot=tf.one_hot(test_labels[:,0],num_classes)

# Show a random image and label
rnd=np.random.randint(num_train)
plt.rcParams['figure.figsize'] = (4.0, 4.0)
plt.imshow(train_images[rnd])
print(cifar10_names[train_labels[rnd][0]])

# Initialize a Keras sequential model
model=keras.models.Sequential()

#FORNOW: placeholder model, replace this with your own model
#model.add(layers.Conv2D(filters=10,kernel_size=1,input_shape=(32,32,3)))
#model.add(layers.GlobalAveragePooling2D())

"""
*************************************************************
*** TODO: implement a linear model using Keras Sequential API
*************************************************************

The model should compute a single linear function of the input pixels
"""

model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images0, train_labels0, epochs=10)

"""
*************************************************************
"""

# output a summary of the model
model.summary()