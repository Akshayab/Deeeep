# Week 1 Udacity challenge
# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Ignore known warning
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# read data (copy separated)
dataframe = pd.read_csv('data/challenge_data.txt', header=None)
x_values = dataframe[0].values.reshape(-1, 1)
y_values = dataframe[1].values.ravel()

# fit the data
lrr = LinearRegression()
svm = SVR() # This shouldn't work for this data lol, but just seeing how the graph looks

lrr.fit(x_values, y_values)
svm.fit(x_values, y_values)

# Print accuracies
print lrr.score(x_values, y_values)*100
print svm.score(x_values, y_values)*100

# Plot
plt.scatter(x_values, y_values)
plt.plot(x_values, lrr.predict(x_values))
plt.plot(x_values, svm.predict(x_values))
plt.show()
