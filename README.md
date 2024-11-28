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


 
