
# Renewable Energy Analysis Using Linear Regression

## Overview
This Python script analyzes a dataset on sustainable energy and uses linear regression to model the relationship between key energy-related features and the renewable electricity-generating capacity per capita. The code involves preprocessing the data, training a machine learning model, and evaluating its performance using visualization.

---

## Steps in the Code

### 1. Import Libraries
The necessary libraries are imported at the beginning:
- **`numpy`**: For numerical computations.
- **`pandas`**: To handle and preprocess the dataset.
- **`matplotlib.pyplot`**: To create visualizations.
- **`sklearn` modules**: For splitting data, training a linear regression model, and evaluating performance.

### 2. Load the Dataset
```python
data = pd.read_csv('global-data-on-sustainable-energy.csv')
```
The dataset is loaded using `pandas`. The CSV file contains global data related to sustainable energy metrics.

### 3. Data Preprocessing
- **Drop non-numeric columns**:
  ```python
  data_numeric = pd.get_dummies(data.drop(['Entity', 'Year'], axis=1))
  ```
  Columns like `Entity` and `Year` are excluded since they are not useful for regression modeling. If there are categorical columns, they are encoded using one-hot encoding.

- **Handle missing or infinite values**:
  ```python
  data_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
  data_numeric.dropna(inplace=True)
  ```
  Missing and infinite values are replaced or removed to ensure the dataset is clean.

### 4. Feature Selection and Target Variable
```python
features = data_numeric[['Access to electricity (% of population)', 'Renewable energy share in the total final energy consumption (%)']]
target = data_numeric['Renewable-electricity-generating-capacity-per-capita']
```
The script selects:
- **Features**: Factors that influence renewable electricity-generating capacity, such as:
  - Access to electricity (% of population).
  - Renewable energy share in total final energy consumption (%).
- **Target variable**: Renewable electricity-generating capacity per capita, the variable we aim to predict.

### 5. Split the Dataset
```python
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```
The dataset is split into:
- **Training set**: 80% of the data for training the model.
- **Testing set**: 20% of the data for evaluating the model.

### 6. Train a Linear Regression Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
A linear regression model is created and trained using the training data.

### 7. Model Parameters: Slope and Intercept
```python
slope = model.coef_[0]
intercept = model.intercept_
```
The slope (`m`) and intercept (`b`) of the regression line are calculated and printed, representing the model's equation:

\[
y = mx + b
\]

### 8. Evaluate the Model
```python
mse = mean_squared_error(y_test, predictions)
```
The **mean squared error (MSE)** is calculated to assess the model's prediction accuracy.

### 9. Visualize the Results
- **Scatter Plot**:
  ```python
  plt.scatter(y_test, predictions, label='Actual vs. Predicted')
  ```
  A scatter plot of actual vs. predicted values shows the model's performance.
- **Regression Line**:
  ```python
  plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Regression Line')
  ```
  The regression line (ideal predictions) is plotted for comparison.

---

## Purpose of the Code
The script aims to:
1. **Understand the relationship** between renewable energy indicators and electricity-generating capacity per capita.
2. **Predict renewable electricity-generating capacity** based on accessible data.
3. **Evaluate model performance** using metrics and visualizations to ensure reliability.
4. **Support sustainable energy decision-making** by identifying key factors influencing renewable energy generation.

This analysis is essential for stakeholders in sustainable energy to make informed decisions and enhance global renewable energy capacity.
