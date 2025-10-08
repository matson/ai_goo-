
# lab for learning basic tensor operations 

import torch 
import numpy as np 
import pandas as pd

# converting integer lists (multiple lengths) to a tensor 
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())


# Convert a float list with length 5 to a tensor
floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
floats_to_tensor.type()

# Convert a integer list with length 5 to float tensor
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()

# Introduce the tensor_obj.size() & tensor_ndimension.size() methods
print("The size of the new_float_tensor: ", new_float_tensor.size())
print("The dimension of the new_float_tensor: ",new_float_tensor.ndimension())

# Introduce the tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(5, 1)

# Introduce the use of -1 in tensor_obj.view(row, column) method
twoD_float_tensor = new_float_tensor.view(-1, 1)

# Convert a numpy array to a tensor
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

# Convert a tensor to a numpy array
back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)

# Set all elements in numpy array to zero 
numpy_array[:] = 0

# Convert a panda series to a tensor
pandas_series=pd.Series([0.1, 2, 0.3, 10.1])
new_tensor=torch.from_numpy(pandas_series.values)

# item() function 
this_tensor=torch.tensor([0,1, 2,3]) 
print("the first item is given by",this_tensor[0].item(),"the first tensor value is given by ",this_tensor[0])

# indexing
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
print("Inital value on index 0:", tensor_sample[0])
tensor_sample[0] = 100
print("Modified tensor:", tensor_sample)

# Slice tensor_sample
subset_tensor_sample = tensor_sample[1:4]

# Change the values on index 3 and index 4
print("Inital value on index 3 and index 4:", tensor_sample[3:5])
tensor_sample[3:5] = torch.tensor([300.0, 400.0])

# Using variable to contain the selected index, and pass it to slice operation
selected_indexes = [3, 4]
subset_tensor_sample = tensor_sample[selected_indexes]

#Using variable to assign the value to the selected indexes
print("The inital tensor_sample", tensor_sample)
selected_indexes = [1, 3]
tensor_sample[selected_indexes] = 100000

#Calculate the mean for math_tensor
mean = math_tensor.mean()

#Calculate the standard deviation for math_tensor
standard_deviation = math_tensor.std()

# Sample for introducing max and min methods
max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
max_val = max_min_tensor.max()
min_val = max_min_tensor.min()

# Method for calculating the sin result of each element in the tensor
pi_tensor = torch.tensor([0, np.pi/2, np.pi])
sin = torch.sin(pi_tensor)

# First try on using linspace to create tensor
len_5_tensor = torch.linspace(-2, 2, steps = 5)

# Second try on using linspace to create tensor
len_9_tensor = torch.linspace(-2, 2, steps = 9)

# Construct the tensor within 0 to 360 degree
pi_tensor = torch.linspace(0, 2*np.pi, 100)
sin_result = torch.sin(pi_tensor)

# tensor + scalar
u = torch.tensor([1, 2, 3, -1])
v = u + 1

# tensor * scalar
u = torch.tensor([1, 2])
v = 2 * u

# tensor * tensor
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v

# dot product 
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
print("Dot Product of u, v:", torch.dot(u,v))

# plotting
# Plot u, v, w
plotVec([
    {"vector": u.numpy(), "name": 'u', "color": 'r'},
    {"vector": v.numpy(), "name": 'v', "color": 'b'},
    {"vector": w.numpy(), "name": 'w', "color": 'g'}
])
