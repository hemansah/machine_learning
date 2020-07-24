import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Classified Data.csv", index_col=0)
df.head()

#Standardize the variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(df.drop("TARGET CLASS", axis=1))
scaled_features = scaler.transform(df.drop("TARGET CLASS", axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()

# Pair plot
sns.pairplot(df, hue='TARGET CLASS')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)

# Using KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

pred = knn.predict(X_test)


# Predictions and evaluations

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
print(confusion_matrix(y_test, pred))

