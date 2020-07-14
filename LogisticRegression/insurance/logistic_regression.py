import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('insurance_data.csv')
df

plt.scatter(df.Age, df.bought_insurance, marker='+', color='red')

df.shape

from sklearn.model_selection import  train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Age']], df.bought_insurance, test_size=0.1)
X_test

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)

model.score(X_test,y_test)

model.predict_proba(X_test)

