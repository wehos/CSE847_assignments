#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Calculate prediction based on data matrix X and weight vector w
def predict(X, w):
    return sigmoid(X@w)

# Calculate sigmoid function
def sigmoid(X):
    return 1/(1+np.exp(-X))

def logistic_train(data, labels, epsilon=1e-5, maxiter=1000):
    # Zero initialization
    weights = np.zeros((data.shape[1],1))
        
    n_iter = 0
    tol = epsilon+1
    n = data.shape[0];
    last_prediction = predict(data, weights)
    
    while tol > epsilon and n_iter < maxiter:
        # Calculate gradient based on prediction
        g = np.mean(data.T @ (labels - last_prediction), 1).reshape(-1, 1)
        # Update weights vector
        weights += ETA * g;
        
        n_iter += 1
        if n_iter>1:
            # Calculate absolute difference in prediction
            tol = np.mean(np.abs(last_prediction - predict(data, weights)))
        last_prediction = predict(data, weights)
        
    return weights

# Train model with first m rows of training data, evaluate it with testing data
def evaluate(n):
    w = logistic_train(X_train[:n], y_train[:n])
    return (((X_test@w)>0) == y_test).mean()


# In[2]:


# Load data from text file
X = [[float(i) for i in line.strip().split('  ')] for line in open('data.txt', 'r').readlines()]
y = [float(line.strip()) for line in open('labels.txt', 'r').readlines()]

# Add ones column to X
X = np.array(X)
X = np.concatenate([X, np.ones([X.shape[0], 1])], 1)
y = np.array(y).reshape(-1, 1)

X_test = X[2000:]
y_test = y[2000:]
X_train = X[:2000]
y_train = y[:2000]

ETA = 1e-3


# In[3]:


for n in [200, 500, 800, 1000, 1500, 2000]:
    print(n, evaluate(n))


# In[ ]:




