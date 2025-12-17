from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # shape (5,1)
y = np.array([2, 5, 4, 8, 7]) # shape (5,)



# Create and train the model
model = LinearRegression()
model.fit(X, y)


# Check coefficients
print("Slope (coefficient):", model.coef_)
print("Intercept:", model.intercept_)

# Scatter plot of original points
plt.scatter(X, y, color='blue', label='Original Data')

# Line of best fit using predicted y values
plt.plot(X, model.predict(X), color='red', label='Line of Best Fit')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Make predictions
predictions = model.predict(np.array([[6], [7]]))
print("Predictions:", predictions)

