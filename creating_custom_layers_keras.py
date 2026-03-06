
# create custom layers and models 

# ---- Installations
'''
!pip install tensorflow==2.16.2
!pip install pydot graphviz
'''

# imports 
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax
import numpy as np 
from tensorflow.keras.layers import Dropout

# ---- Step 1: Define Custom Layer Class

# Define custom layer 
class CustomDenseLayer(Layer):
    def __init__(self, units=32):
        super(CustomDenseLayer, self).__init__()
        self.units = units
        # where units = neurons 

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)


# ---- Step 2: Define Model and Compile 

# Define the model with Softmax in the output layer
model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10),  # Hidden layer with ReLU activation
    Softmax()              # Output layer with Softmax activation for multi-class classification
])

model2 = Sequential([
    # with a dropout 
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10),
    Softmax()
])


model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building:")
model.summary()

# Build the model to show parameters
model.build((1000, 20))
print("\nModel summary after building:")
model.summary()

# Generate random data 
x_train = np.random.random((1000, 20)) 
y_train = np.random.randint(10, size=(1000, 1)) 

# Convert labels to categorical one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) 
model.fit(x_train, y_train, epochs=10, batch_size=32) 

# ---- Results
'''
By changing neuron count and adding a dropout layer, you 
can see the total loss being affected in either direction
'''

'''
Result from running first time: loss - 2.2869 

Result from adding dropout: loss - 2.3031 

Result from adding more neurons (128) loss - 3.3301

'''