
# create a cnn with KERAS 

# ---- for installation

"""
!pip install numpy==2.0.2
!pip install pandas==2.2.2
!pip install tensorflow_cpu==2.18.0
!pip install matplotlib==3.9.2

"""

# ---- imports 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical
from keras.layers import Conv2D # to add convolutional layers
from keras.layers import MaxPooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers

# ---- import data, load data and reshape (normalize)

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

# ---- convert target variable to binary categories 

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1] # number of categories

# ---- define our convolutional model 

# one set of convolutional and pooling layers 

def convolutional_model_1():
    
    # create model
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

# two sets of convolutional and pooling layers 

def convolutional_model_2():
    
    # create model
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model


# ---- build model, fit, and evaluate 

# build the model
model = convolutional_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

'''
Model 1
Accuracy: 0.9884999990463257 
 Error: 1.1500000953674316

Model 2 
Accuracy: 0.9866999983787537 
 Error: 1.3300001621246338

'''





# ---- parameters to change that influence accuracy 

# epochs, batch_size 

# ---- batch_size example - change batch size 

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=1024, verbose=2)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

'''
Accuracy: 0.9887999892234802 , Error: 1.1200010776519775
'''

# ---- epochs example - change epochs count 

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=1024, verbose=2)
# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1], 100-scores[1]*100))

"""
Accuracy: 0.9890000224113464 , Error: 1.0999977588653564
"""

# ---- Results 

"""
One Set vs. Two sets - in theory adding more layers would mean improved accuracy, 
however, you must account for overfitting, changing/tuning hyperparameters, and limited optimization results

epochs - lowest error, best accuracy - more epochs better accuracy 

batch_size - lower error compared to the previous chosen batch size - greater the better for this model 

"""