# Week 1 Udacity Deep Learning Course
# Imports
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data (fixed width formatted)
dataframe = pd.read_fwf('data/brain_data.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

# Train the linear regression model
clf = linear_model.LinearRegression()
clf.fit(x_values, y_values)

# Visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, clf.predict(x_values))
plt.show()
