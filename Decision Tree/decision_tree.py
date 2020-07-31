import pandas as pd 
import numpy as np

# Step 1. Load dataset
df = pd.read_csv('salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values


# Step 2. Fit Decision tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse")
regressor.fit(X,y)


# Step 3. Visualize
import matplotlib.pyplot as plt
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X,y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Decision Tree Regressor")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()

# Step 4. Predict
y_pred = regressor.predict([[6.75]])
print("The predicted salary of a person  at 6.5 level is ",y_pred)