

# how common tensor operations work in pytorch
# define simple dataset 

# ---- Installations 
'''
%%time
%pip install pandas numpy matplotlib
%pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
'''

# imports
import numpy as np 
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
torch.manual_seed(1)
# %matplotlib inline  

# ---- Tensor Operations 

# to tensor 
ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype) # torch.int64
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type()) # torch.LongTensor

floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]) # torch.float32, torch.FloatTensor

# int list -> float tensor 
new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])

# size of tensor_obj 
new_float_tensor.size()
new_float_tensor.ndimension()

# returns value of tensor 
this_tensor=torch.tensor([0,1, 2,3]) 
print(this_tensor[0].item()) # 0 

# to list 
torch_to_list=this_tensor.tolist()

# to reshape 
tensor_1 = torch.tensor([1, 2, 3, 4, 5])
my_new_tensor = tensor_1.view(1,5)
print(my_new_tensor)

# slicing / indexing is the same 
tensor_sample = torch.tensor([20, 1, 2, 3, 4])
tensor_sample[4]
subset_tensor_sample = tensor_sample[1:4]

# mean and standard deviation 
math_tensor = torch.tensor([1.0, -1.0, 1, -1])
mean = math_tensor.mean()
standard_deviation = math_tensor.std()

# min and max
max_min_tensor = torch.tensor([1, 1, 3, 5, 5])
max_val = max_min_tensor.max()
min_val = max_min_tensor.min()

# tensor add 
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
w = u + v

# tensor sub 
u = torch.tensor([1, 0])
v = torch.tensor([0, 1])
y = u - v 

# scalar 
u = torch.tensor([1, 2, 3, -1])
v = u + 1
print ("Addition Result: ", v)

# multiplication - scalar + tensor 
u = torch.tensor([1, 2])
v = 2 * u
print("The result of 2 * u: ", v)

u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
w = u * v

# dot product 
u = torch.tensor([1, 2])
v = torch.tensor([3, 2])
torch.dot(u,v)


# ---- Dataset : an example 

# define a class - 
# these are arbitrary values 

class toy_set(Dataset):
    
    # Constructor with defult values 
    def __init__(self, length = 100, transform = None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform
     
    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)     
        return sample
    
    # Get Length
    def __len__(self):
        return self.len
    
our_dataset = toy_set()
length_of_dataset = len(our_dataset)

# datasets are iterable 
for x,y in our_dataset:
    print(' x:', x, 'y:', y)

