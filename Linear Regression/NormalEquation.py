import numpy as np
def normal_equation(X, Y):
    # Solves for theta in one step using the normal equation:
    # theta = (X^T X)^(-1) X^T Y
    theta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return theta

