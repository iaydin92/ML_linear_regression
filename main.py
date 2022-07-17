from statistics import LinearRegression, linear_regression
import numpy as np
from sklearn.linear_model import LogisticRegression


# Linear regression model

class linear_regression:
    # initiating the parameters (learning rate, number of iterations)
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        # number of training examples & number of features
        # m is the number of data points that we are using and n is the number of features in our dataset which is 1 in our case (years of experience, bear in mind that the salary is not a feature it is a target that we are searching for)

        self.m, self.n = X.shape  # number of rows and colums m and n

        # initiating the weight (slope) and bias (intercept)

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # implementing Gradient decent

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self, ):
        Y_prediction = self.predict(self.X)

        # calculate gradients

        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction) / self.m

        # updating the weights

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b


# Importing dependicies

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data pre-processing

# loading the data from csv file to a pandas dataframe

salary_data = pd.read_csv(
    'C:/Users/ibrah/Desktop/MACHINE LEARNING/Linear Regression/salary_data.csv')

# printing the first 5 columns of the dataframe

# print(salary_data.head())

# printing the last 5 columns of the dataframe

# print(salary_data.tail())

# print number of rows and columns in the data frame

# print(salary_data.shape)


# checking for missing values

# print(salary_data.isnull().sum())


# Splitting the feature and target

X = salary_data.iloc[:, :-1].values
Y = salary_data.iloc[:, 1].values

# Splitting the data set into training and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=2)

# Training the Linear Regression Model

model = linear_regression(learning_rate=0.02, no_of_iterations=1000)
model.fit(X_train, Y_train)

# printing the parameter values ( weights & bias)
print('weight = ', model.w[0])
print('bias = ', model.b)

# predicting the salary value for test data

test_data_prediction = model.predict(X_test)
print(test_data_prediction)

# Visualizing the predicted values & actual values

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, test_data_prediction, color='blue')
plt.xlabel('Work experience')
plt.ylabel('Salary')
plt.title('Work experience vs Salary')
plt.show()
