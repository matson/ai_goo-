
'''
Goal: Build a logistic regression model to predict
 the outcomes of League of Legends matches 

Data is from lab's provided csv file 

'''
# ---- Installation 

'''
%%time
%pip install pandas scikit-learn matplotlib
%pip install torch==2.8.0+cpu torchvision==0.23.0+cpu torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

'''

# imports 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# ---- STEP 1: Data Loading and Preprocessing 

# Load the dataset: Use pd.read_csv() to load the dataset into a pandas DataFrame.
data = pd.read_csv('league_of_legends_data_large.csv')
# make sure it is correct - show the data 
data.head()

# Split data into features and target: Separate win (target) and the remaining columns (features).
# Features: all columns except 'win'
X = data.drop('win', axis=1)

# Target: 'win' column
y = data['win']

# Split the Data into Training and Testing Sets: Use train_test_split() from sklearn.model_selection 
# to divide the data. Set test_size=0.2 to allocate 20% for testing 
# and 80% for training, and use random_state=42 to ensure reproducibility of the split.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features: Use StandardScaler() from sklearn.preprocessing to scale the features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors: Use torch.tensor() to convert the data to PyTorch tensors.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# ---- STEP 2: Implement Logistic Regression Model using PyTorch 

class LogisticRegressionModel(nn.Module):
    
    # Constructor
    def __init__(self, n_inputs):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
    
    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat
    
# Assume X_train is your training feature tensor
input_dim = X_train.shape[1]  # Number of features

# Initialize the logistic regression model
model = LogisticRegressionModel(input_dim)

# Define the loss function (Binary Cross-Entropy Loss)
criterion = nn.BCELoss()

# Initialize the optimizer with SGD and learning rate 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ---- STEP 3: Model Training 

# Number of epochs for training
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    
    optimizer.zero_grad()  # Zero the gradients
    
    # Forward pass
    outputs = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    predicted_train = (train_outputs >= 0.5).float()
    train_accuracy = (predicted_train == y_train_tensor).float().mean()
    
    test_outputs = model(X_test_tensor)
    predicted_test = (test_outputs >= 0.5).float()
    test_accuracy = (predicted_test == y_test_tensor).float().mean()

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# ---- STEP 4: Model Optimization and Evaluation 

# Assuming model, criterion, and tensors are already defined:
# model = LogisticRegressionModel(input_dim)
# criterion = nn.BCELoss()
# X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# Initialize optimizer with weight decay (L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()  # Training mode
    
    optimizer.zero_grad()  # Clear gradients
    
    outputs = model(X_train_tensor)  # Forward pass
    
    loss = criterion(outputs, y_train_tensor)  # Compute loss
    
    loss.backward()  # Backpropagation
    
    optimizer.step()  # Update weights
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate model performance
model.eval()
with torch.no_grad():
    train_outputs = model(X_train_tensor)
    predicted_train = (train_outputs >= 0.5).float()
    train_accuracy = (predicted_train == y_train_tensor).float().mean()
    
    test_outputs = model(X_test_tensor)
    predicted_test = (test_outputs >= 0.5).float()
    test_accuracy = (predicted_test == y_test_tensor).float().mean()

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# ---- STEP 5: Visualization and Interpretation 

# Assuming your tensors and LogisticRegressionModel are already defined:
# X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
# LogisticRegressionModel class and input_dim defined

# Initialize model, criterion, optimizer with L2 regularization
model = LogisticRegressionModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    # Predictions (probabilities)
    y_train_pred_prob = model(X_train_tensor)
    y_test_pred_prob = model(X_test_tensor)

    # Convert probabilities to binary predictions
    y_train_pred = (y_train_pred_prob >= 0.5).float()
    y_test_pred = (y_test_pred_prob >= 0.5).float()

# Convert tensors to numpy for sklearn functions
y_train_true = y_train_tensor.numpy()
y_train_pred_np = y_train_pred.numpy()
y_test_true = y_test_tensor.numpy()
y_test_pred_np = y_test_pred.numpy()
y_test_pred_prob_np = y_test_pred_prob.numpy()

# Classification report
print("Classification Report (Test Data):")
print(classification_report(y_test_true, y_test_pred_np, target_names=['Class 0', 'Class 1']))

# Confusion matrix plot
cm = confusion_matrix(y_test_true, y_test_pred_np)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title('Confusion Matrix - Test Data')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pred_prob_np)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print final accuracy
train_accuracy = (y_train_pred == y_train_tensor).float().mean()
test_accuracy = (y_test_pred == y_test_tensor).float().mean()
print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# ---- STEP 6: Model Saving and Loading 

# Save the model
torch.save(model.state_dict(), 'logistic-model.pth')
print("Model saved successfully.")

# Load the model
model = LogisticRegressionModel(input_dim)
model.load_state_dict(torch.load('logistic-model.pth'))


# Ensure the loaded model is in evaluation mode
model.eval()  # Set to evaluation mode

with torch.no_grad():
    # Get predictions (probabilities) on test data
    y_pred_probs = model(X_test_tensor)
    
    # Convert probabilities to binary predictions using 0.5 threshold
    y_pred_labels = (y_pred_probs >= 0.5).float()
    
    # Calculate accuracy
    accuracy = (y_pred_labels == y_test_tensor).float().mean()
    
print(f'Test Accuracy after loading the model: {accuracy:.4f}')

# ---- STEP 7: Hyperparameter Tuning 

learning_rates = [0.01, 0.05, 0.1]
num_epochs = 50
input_dim = X_train_tensor.shape[1]

best_lr = None
best_accuracy = 0.0
results = {}

# loop through learning rates 
for lr in learning_rates:
    print(f"\nTraining with learning rate: {lr}")
    
    # Reinitialize model and optimizer for each learning rate
    model = LogisticRegressionModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_test_pred_prob = model(X_test_tensor)
        y_test_pred = (y_test_pred_prob >= 0.5).float()
        accuracy = (y_test_pred == y_test_tensor).float().mean().item()
    
    print(f"Test Accuracy at lr={lr}: {accuracy:.4f}")
    results[lr] = accuracy
    
    # Track best learning rate
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_lr = lr

print(f"\nBest learning rate: {best_lr} with Test Accuracy: {best_accuracy:.4f}")


# ---- STEP 8: Feature Importance 

# Extract the weights of the linear layer
## Write your code here
weights = model.linear.weight.data.numpy().flatten()

# Create a DataFrame for feature importance
## Write your code here
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': weights
})

# Sort by absolute importance
feature_importance_df['Abs_Importance'] = feature_importance_df['Importance'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Abs_Importance', ascending=True)

# Plot horizontal bar chart
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance (Weight)')
plt.title('Feature Importance from Logistic Regression Model')
plt.show()