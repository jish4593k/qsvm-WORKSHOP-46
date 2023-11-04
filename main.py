import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier
svm = SVC()

# Define hyperparameters for tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4]
}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the SVM classifier with the best hyperparameters
best_svm = SVC(C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'])
best_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display a classification report
class_names = iris.target_names
report = classification_report(y_test, y_pred, target_names=class_names)
print("Classification Report:\n", report)

# Visualize the results using Seaborn
df = pd.DataFrame(data=np.c_[X_test, y_pred], columns=iris.feature_names + ['predicted'])
g = sns.PairGrid(df, hue="predicted")
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot)
g.add_legend()

# Show the Seaborn plot
plt.show()
