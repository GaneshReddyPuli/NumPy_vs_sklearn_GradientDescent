#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:16:10 2024

@author: ganeshreddypuli
"""

# Sales based on TV marketing expenses

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

adv = pd.read_csv("tvmarketing.csv")
adv.head()
adv.plot(x='TV', y='Sales', kind='scatter', c='black')

X = adv['TV']
Y = adv['Sales']

# 1) Linear Regression using NumPy

m_numpy, b_numpy = np.polyfit(X, Y, 1)
print(f"Linear regression with NumPy. Slope: {m_numpy}. Intercept: {b_numpy}")

def plot_linear_regression(X, Y, x_label, y_label, m, b):
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.plot(X, Y, 'o', color='black')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.plot(X, m*X + b, color='red')
    
plot_linear_regression(X, Y, 'TV', 'Sales', m_numpy, b_numpy)

def pred_numpy(X, m, b):
    
    X = X.to_numpy()
    return X*m + b

Y_pred_numpy = pred_numpy(X, m_numpy, b_numpy)

#print(f"TV marketing expenses:\n{X}")
#print(f"Predictions of sales using NumPy linear regression:\n{Y_pred_numpy.T}")

##############################################

# 2) Linear Regression using sklearn
lr_sklearn = LinearRegression()
    
X_sklearn = X.to_numpy().reshape(-1, 1)
Y_sklearn = Y.to_numpy().reshape(-1, 1)

lr_sklearn.fit(X_sklearn, Y_sklearn)
m_sklearn = lr_sklearn.coef_
b_sklearn = lr_sklearn.intercept_

print(f"\nLinear regression using Scikit-Learn. Slope: {m_sklearn[0][0]}. Intercept: {b_sklearn[0]}")

def pred_sklearn(X, lr_sklearn):
 
    Y = lr_sklearn.predict(X)
 
    return Y

Y_pred_sklearn = pred_sklearn(X_sklearn, lr_sklearn)

#print(f"TV marketing expenses:\n{X_sklearn}")
#print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")

##############################################

# 3) Linear Regression using Gradient Descent

print("\nLinear Regression using Gradient Descent")
X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

def E(m, b, X, Y):
    return 1/(2*len(Y))*np.sum((m*X + b - Y)**2)

def dEdm(m, b, X, Y):
    res = 1/len(Y)*np.dot(m*X + b - Y, X)
    
    return res

def dEdb(m, b, X, Y):
    res = 1/len(Y)*np.sum(m*X + b - Y)
    
    return res

def gradient_descent(dEdm, dEdb, m, b, X, Y, learning_rate = 0.001, num_iterations = 1000, print_cost=False):
    for iteration in range(num_iterations):
        m_new = m - learning_rate*dEdm(m, b, X, Y)
        b_new = b - learning_rate*dEdb(m, b, X, Y)
        m = m_new
        b = b_new
        if print_cost:
            print (f"Cost after iteration {iteration}: {E(m, b, X, Y)}")
        
    return m, b

m_initial = 0; b_initial = 0; num_iterations = 1000; learning_rate = 0.01
m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial, 
                              X_norm, Y_norm, learning_rate, num_iterations, print_cost=False)  # Make print_cost=True to print the cost in each iteration

print(f"Gradient descent result: m_min, b_min = {m_gd}, {b_gd}") 

X_pred_norm = X_norm.to_numpy().reshape(-1, 1)

Y_pred_gd_norm = m_gd * X_pred_norm + b_gd

# Remember, that the initial datasets were normalized. 
# To make the predictions, you need to normalize `X_pred` array, 
# calculate `Y_pred` with the linear regression coefficients `m_gd`, `b_gd` 
# and then **denormalize** the result (perform the reverse process of normalization):
    
Y_pred_gd = Y_pred_gd_norm * np.std(Y) + np.mean(Y)

print(f"\nTV marketing expenses:\n{X}")
print(f"\nPredictions of sales using NumPy:\n{Y_pred_numpy.T}")
print(f"\nPredictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")
print(f"\nPredictions of sales using Gradient Descent:\n{Y_pred_gd.T}")




    