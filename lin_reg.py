import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets


def RMSE(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2))


class LinearRegressor():

    def __init__(self, max_iter=1000, lr=0.05):
        self.max_iter = max_iter
        self.lr = lr
        self.w = None
        self.b = None

    def fit(self, X=None, y=None):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        current_iter = 1
        while current_iter <= self.max_iter:
            y_predicted = np.dot(X, self.w) + self.b
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            current_iter += 1
        pass
    
    def predict(self, X=None):
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
    
    def r2_score(self, X=None, y=None):
        y_pred = np.dot(X, self.w) + self.b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

