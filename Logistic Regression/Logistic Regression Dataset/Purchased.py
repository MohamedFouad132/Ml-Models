import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
data = pd.read_csv("Social_Network_Ads.csv")

print("First 20 rows:")
print(data.head(20))

# Convert categorical feature to numerical

data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

# Drop ID column as it is not useful for prediction

data = data.drop("User ID", axis= 1)

# Check for outliers using Z-score method

z_scores = np.abs(stats.zscore(data[['Age','EstimatedSalary']]))
print(np.where(z_scores > 3))  # rows with extreme outliers



# Standardize numerical features

scaler = StandardScaler()
data[['Age', 'EstimatedSalary']] = scaler.fit_transform(data[['Age', 'EstimatedSalary']])

# Separate features and target variable

X = data[['Gender','Age','EstimatedSalary']]
y = data['Purchased']

# Check class distribution in target variable
print(y.value_counts(normalize=True))

# Give more weight to minority class to handle imbalance

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42)

model = LogisticRegression(class_weight='balanced', random_state=42)
# Fit the logistic regression model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance using confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

TN, FP, FN, TP = cm.ravel()  # unpack the matrix

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
