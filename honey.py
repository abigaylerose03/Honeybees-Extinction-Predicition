import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year["year"]

X = X.values.reshape(-1, 1) # reshaping the column into a 1x1 matrix
# print(X)

y = prod_per_year["totalprod"]
# print(y)

plt.scatter(X, y)
plt.xlabel("Year")
plt.ylabel("Honey Production")

regre = linear_model.LinearRegression()
regre.fit(X, y)
print(regre.coef_[0]) # coefficient is slope
print(regre.intercept_)
y_predict = regre.predict(X) # mx + b
plt.plot(X, y_predict)
# plt.show()

X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)
print(X_future)

future_predict = regre.predict(X_future)
plt.plot(X_future, future_predict)
plt.show()