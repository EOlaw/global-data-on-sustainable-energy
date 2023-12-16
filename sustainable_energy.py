# Load the dataset data = pd.read_csv('global-data-on-sustainable-energy.csv')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Calculate slope and y-intercept
slope = model.coef_[0]
intercept = model.intercept_

# Print the slope and y-intercept
print(f'Slope (m): {slope}')
print(f'Y-Intercept (b): {intercept}')

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
#print(f'Mean Squared Error: {mse}')

# Scatter plot of actual vs. predicted values
plt.scatter(y_test, predictions, label='Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values for Renewable-electricity-generating-capacity-per-capita')

# Plot the regression line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Regression Line')

plt.legend()
plt.show()


