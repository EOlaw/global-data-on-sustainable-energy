# Load the dataset data = pd.read_csv('global-data-on-sustainable-energy.csv')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('global-data-on-sustainable-energy.csv')

# Drop non-numeric columns or encode them, e.g., using one-hot encoding
data_numeric = pd.get_dummies(data.drop(['Entity', 'Year'], axis=1))

# Check for missing values and handle them if needed
data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
data_numeric.dropna(inplace=True)

# Select features and target variable
features = data_numeric[['Access to electricity (% of population)', 'Renewable energy share in the total final energy consumption (%)']]
target = data_numeric['Renewable-electricity-generating-capacity-per-capita']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


