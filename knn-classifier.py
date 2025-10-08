

# Problem: categorize customers into four buying groups 
# Example of classification 

# imports 
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# reading data
data = pd.read_cvs('IBM_customer_data_file_link_here.cvs')
data.head()

# drop any features that you would not need to prepare the data

# normalize data for k-nn models
X_norm = StandardScaler().fit_transform(X)

# train / test split 
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# train model
# pick a k value 
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# predict 
yhat = knn_model.predict(X_test)

# calculate accuracy score
print("Test set Accuracy: ", accuracy_score(y_test, yhat))