from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
from scipy.sparse.construct import random
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.distributions import kdeplot
from seaborn.palettes import color_palette 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


titanic_data = pd.read_csv('titanic_train.csv')
titanic_data.head()

print('Number of passengers: {}'.format(len(titanic_data)))


# Passengers who survived and who didn't survived

sns.countplot(x='Survived', data=titanic_data)

# Passengers male survived and female survived
sns.countplot(x='Survived', hue='Sex', data=titanic_data)

# Survivors based on passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic_data)

titanic_data['Age'].plot.hist()

titanic_data['Fare'].plot.hist(bins=40)

titanic_data.info()

titanic_data['SibSp'].value_counts()
sns.countplot(x='SibSp', data=titanic_data)

## Data Wrangling
titanic_data.isnull()
titanic_data.isnull().sum()

flatui = ["#34495e","#95a5a6"]
sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap=sns.color_palette(flatui))

# Studying Age column

sns.boxplot(x='Pclass', y='Age', data=titanic_data)


titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.head()

# Filling age columns
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

sns.heatmap(titanic_data.isnull(),yticklabels=False, cmap=sns.color_palette(flatui))

titanic_data.isnull().sum()

#
sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
sex.head()

embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
embark.head()


pclass = pd.get_dummies(titanic_data['Pclass'], drop_first=True)
pclass.head()


titanic_data = pd.concat([titanic_data, sex, embark, pclass], axis=1)
titanic_data.head()

titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1, inplace=True)
titanic_data.head()


X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
type(y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logistic = LogisticRegression()

logistic.fit(X_train, y_train)


prediction = logistic.predict(X_test)
classification_report(y_test, prediction)

confusion_matrix(y_test, prediction)

accuracy_score(y_test, prediction)

