# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Regression
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
ypred = regressor.predict(6.5)
print(ypred)

# Plotting
plt.scatter(X, y, c='blue')
X_plot = np.linspace(X[0, 0], X[-1, 0], 100).reshape((-1, 1))
plt.plot(X_plot, regressor.predict(X_plot), c='red')
plt.plot()
plt.show()
