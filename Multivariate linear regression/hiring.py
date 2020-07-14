import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')
df

df['experience'] = df['experience'].fillna('zero') 
df


df['experience'] = df['experience'].apply(w2n.word_to_num)
df

df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(math.floor(df['test_score(out of 10)'].mean()))
df


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']], df['salary($)'])

reg.predict([[2,9,6]])
reg.predict([[12,10,10]])
reg.predict([[0,8.0,9]])
reg.predict([[0,8.0,6]])
reg.predict([[0,8.0,9]])
reg.predict([[20,10,10]])




