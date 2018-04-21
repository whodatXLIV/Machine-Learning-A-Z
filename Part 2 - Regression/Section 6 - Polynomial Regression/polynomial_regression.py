# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Importing the data set
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Linear regression model
regressor = LinearRegression()
regressor.fit(X, y)
linear_predict = regressor.predict(6.5)

# Poly regression model
X_poly = PolynomialFeatures(degree=4).fit_transform(X)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
poly_predict = poly_regressor.predict(
    PolynomialFeatures(degree=4).fit_transform(6.5))
print(poly_predict)
# Visualize the linear regresson
plt.figure(0)
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')

# Visualize the poly Regression
plt.plot(X, poly_regressor.predict(X_poly), color='green')
plt.title('Position Level vs Salary')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
