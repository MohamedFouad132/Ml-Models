import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load in the CSV

data = pd.read_csv("data/Student_Performance.csv")

# Convert feature from categorical to numerical

data["Extracurricular Activities"] = data["Extracurricular Activities"].map({"Yes": 1, "No": 0})

# Claculate Z score for each data value to identify outliers

z_scores = np.abs(stats.zscore(data))
print(np.where(z_scores > 3))

"""

No outliers were found so further handling is not required.

"""

# Seperate features and target

X = data.drop("Performance Index", axis= 1)
y = data["Performance Index"]

# Split data into training and testing (80/20) split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 7)

# train regression model using training data

model = LinearRegression()
model.fit(X_train, y_train)

# test the model using the testing data 

y_pred = model.predict(X_test)

# Check model performance using metrics (mean squared error and R^2)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

"""

MSE = 4.2
RSME would roughly be 2 which is great for target value from 0-100
R^2 of 99 percent so most variance is explained and data is fit well
Overall model generalises well

"""








