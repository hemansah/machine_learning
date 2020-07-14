import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('Height_Weight_Dataset.csv')
data.head()

X = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)


from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

plt.scatter(X_train, y_train, color='green')

plt.plot(X_train, LinReg.predict(X_train), color='blue')
plt.title('Linear Regression')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
polynom = PolynomialFeatures(degree=2)
X_polynom = polynom.fit_transform(X_train)
X_polynom

PolyReg = LinearRegression()
PolyReg.fit(X_polynom, y_train)

plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, PolyReg.predict(polynom.fit_transform(X_train)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Age')
plt.ylabel('Height')
plt.show()

