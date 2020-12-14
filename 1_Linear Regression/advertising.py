# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:22:44 2020.
"""

# =============================================================================
# A simple Linear Regression (LR) ML algorithm on the Advertising dataset from
# Kaggle. Made with numpy and pandas library from scratch with LR algorithms
# =============================================================================


# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd

# read and understand the data
PATH = r'C:\Users\zeeali\Desktop\Aliasgar\ML DL AI\datasets'
DATASET_PATH = PATH + str(r'\advertising.csv')
dataset = pd.read_csv(DATASET_PATH)
print('\nData(First 5 rows):')
print(dataset.head())
print('\n Summary of Data:')
print(dataset.describe())

# plot the data
fig, axs = plt.subplots(1, 3)
axs[0].plot(dataset['TV'], dataset['Sales'], 'rx')
axs[1].plot(dataset['Radio'], dataset['Sales'], 'rx')
axs[2].plot(dataset['Newspaper'], dataset['Sales'], 'rx')

# plot for better understanding
# Sales and TV vary linearly
plt.figure()
plt.plot(dataset['TV'], dataset['Sales'], 'rx')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()


# turn dataframe to numpy array for segregation of data
data = dataset.values
data = np.array([(data[:, 0]), (data[:, 3])])
data = np.transpose(data)
train_x = []
train_y = []
cv_x = []
cv_y = []
test_x = []
test_y = []
fig = plt.figure()

# parameters
cv_ratio = 0.2
test_ratio = 0.2
alpha = 0.000007
iters = 500
# weight = 0.054
# bias = 6.948
weight = 0.1
bias = 7
lambda_r = 1000


# split the data into train, cv, test sets
def split(data, cv_ratio, test_ratio):
    # print(np.shape(data))
    index_list = []
    for i in range(int(cv_ratio * 100)):
        index = rnd.randrange(0, len(data))
        index_list.append(index)
        cv_x.append(data[index, 0])
        cv_y.append(data[index, 1])
    data_n = np.delete(data, index_list, 0)
    # print(np.shape(data_n))
    index_list = []
    for i in range(int(test_ratio * 100)):
        index = rnd.randrange(0, len(data_n))
        index_list.append(index)
        test_x.append(data_n[index, 0])
        test_y.append(data_n[index, 1])
    data_nw = np.delete(data_n, index_list, 0)
    # print(np.shape(data_nw))
    for i in range(len(data_nw)):
        train_x.append(data_nw[i, 0])
        train_y.append(data_nw[i, 1])


# call split functions and convert the arrays from lists to numpy arrays
split(data, cv_ratio, test_ratio)
train_x = np.array(train_x)
train_y = np.array(train_y)
cv_x = np.array(cv_x)
cv_y = np.array(cv_y)
test_x = np.array(test_x)
test_y = np.array(test_y)


# creating model
def model(x, y, flag):
    cost_iter = []
    m = len(x)
    global weight
    global bias
    for i in range(iters):
        y_pred = (x * weight) + bias
        loss = y_pred - y
        if flag == 1:
            cost = ((sum(pow(loss, 2)) / (2 * m)) +
                    ((lambda_r * pow(weight, 2)) / (2 * m)))
        else:
            cost = (sum(pow(loss, 2)) / (2 * m))
        cost_iter.append(cost)
        w_grad = (sum(np.multiply(loss, x))) / m
        b_grad = (sum(loss)) / m
        weight -= (alpha * w_grad)
        bias = bias - (alpha * b_grad)

    for i in range(iters):
        plt.plot(i, cost_iter[i], 'rx')
    plt.xlabel("No of iterations")
    plt.ylabel("Cost")
    plt.show()
    # if m == 20:
    #     print(sum(cost_iter)/iters)


def model_test(x, y):
    m = len(x)
    global weight
    global bias
    y_pred = (x * weight) + bias
    loss = y_pred - y
    cost = (sum(pow(loss, 2)) / (2 * m))
    print('\n Cost of test Set:', cost)
    plt.plot(x, y, 'rx')
    plt.plot(x, y_pred)
    plt.show()
    # print('Accuracy of model:', accuracy)

# traning model
model(train_x, train_y, 0)
model(cv_x, cv_y, 1)

# test model
model_test(test_x, test_y)
