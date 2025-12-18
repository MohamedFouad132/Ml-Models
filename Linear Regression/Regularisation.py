import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
true_weights = np.array([5, 0, -3, 0, 2])
y = X @ true_weights + np.random.randn(100) * 0.5  # add some noise


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
weights_before = lr.coef_

print("Linear Regression MSE - Train:", mse_train)
print("Linear Regression MSE - Test :", mse_test)

train_test_ratio = mse_test / mse_train
overfitting = (train_test_ratio > 1.1)  # 10% worse on test set
if overfitting:
    print("\nModel is overfitting. Regularization required.")

    alphas = [0.01, 0.1, 1, 10, 100]

    # Ridge
    ridge_cv = GridSearchCV(Ridge(), {'alpha': alphas}, scoring='neg_mean_squared_error', cv=5)
    ridge_cv.fit(X_train, y_train)
    best_ridge = ridge_cv.best_estimator_
    ridge_test_mse = mean_squared_error(y_test, best_ridge.predict(X_test))

    # Lasso
    lasso_cv = GridSearchCV(Lasso(max_iter=10000), {'alpha': alphas}, scoring='neg_mean_squared_error', cv=5)
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_
    lasso_test_mse = mean_squared_error(y_test, best_lasso.predict(X_test))

    # Choose model
    if ridge_test_mse < lasso_test_mse:
        model = best_ridge
        model_name = "Ridge"
        mse_after = ridge_test_mse
    else:
        model = best_lasso
        model_name = "Lasso"
        mse_after = lasso_test_mse

    print(f"\nSelected model: {model_name}")
    print(f"Test MSE before regularization: {mse_test:.4f}")
    print(f"Test MSE after regularization:  {mse_after:.4f}")
    print(f"Weights before regularization: {weights_before}")
    print(f"Weights after regularization:  {model.coef_}")

else:
    print("\nNo overfitting detected. Keeping plain Linear Regression.")
    print(f"Test MSE: {mse_test:.4f}")
    print(f"Weights: {weights_before}")
