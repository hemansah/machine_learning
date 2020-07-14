import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('home_price_Monroe.csv', sep=',')
df

# Filling NaN values

# df['bedrooms'] = df['bedrooms'].fillna(math.floor(df['bedrooms'].median()))

# making model

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age ']],df['price'])


reg.coef_
reg.intercept_

reg.predict([[3000, 4, 15]])
reg.predict([[3000, 3, 30]])
reg.predict([[3000, 3, 18]])
reg.predict([[2500, 3, 5]])


