import numpy as np
def cost_function(X,Y, theta):
    m = len(Y) # or m = X.shape[0] i.e the number of rows.
    predictions = np.dot(X,theta)
    errors = Y - predictions
    cost = 1 / (2 * m) * np.sum(errors** 2)
    return cost


def gradient_descent(X, Y, theta, alpha, num_iters):
    m = X.shape[0]
    cost_history = []
    for i in range(num_iters):
        predictions = np.dot(X, theta)
        errors = predictions - Y
        gradient = (1 / m) * np.dot(X.T, errors)
        theta = theta - alpha * gradient
        cost_history.append(cost_function(X, Y, theta))
        
    return theta, cost_history




    


