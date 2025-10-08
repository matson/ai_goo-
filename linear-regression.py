
# a simple linear regression model 

# imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# create a url with the dataset (this one from IBM)
# use pandas to read it 
# sample 5 rows 
# describe the data 
url = "ibm dataset_csv"
df = pd.read_csv(url)
df.sample(5) 
df.describe() 

# select a few features 
cdf = df[['Enginesize', 'Cylinders', 'Co2emissions' ]]
cdf.sample(9)
cdf.hist() # makes a histogram 
plt.show() # shows plots

# plotting
plt.scatter(cdf.Enginesize, cdf.Co2emissions, color='blue')
plt.xlabel("Enginesize") # x-axis
plt.ylabel("Emission") # y-axis
plt.show()

# train and test datasets
# use 80 % of dataset for training, 20 % for testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create a model object 
regressor = linear_model.LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)

# select a feature from the dataframe and split the data 
X = cdf.FuelConsumption.to_numpy()

# calculate mean squared error
mean_squared_error(y_test, y_test)