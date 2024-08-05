import numpy as np
import pandas as pd


def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def RMSE(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))





class LinearRegressor():

    def __init__(self, max_iter=1000, lr=0.05, threshold=10e-6):
        self.max_iter = max_iter
        self.lr = lr
        self.threshold = threshold
        self.w = None
        self.bias = None

    def fit(self, X=None, y=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        dw = 0
        current_iter = 1
        while current_iter <= max_iter and dw >= self.threshold:
            y_predicted = X.T @ self.weights + self.bias
            dw = (1 / n_samples) * X.T @ (y_predicted - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.w -= lr * dw
            self.b -= lr * db
            current_iter += 1
        pass
    
    def predict(self, X=none):
        return X.T @ self.weights + self.bias
    
