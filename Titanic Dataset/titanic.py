import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('titanic_train.csv')
train.head()

"""Exploratory data analysis"""

#Missing values
train.isnull() # Not a good way...

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# Plotting who has survived or not
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train) # 0 = not survived, 1 = survived

# Plotting who has survived or not based on the gender
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=train, palette='RdBu_r')

# Plotting who has survived or not based on the class
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train, palette='rainbow')

"""Age Distribution"""
sns.distplot(train['Age'].dropna(), kde=True, color='darkred', bins=40)
 #OR

train['Age'].hist(bins=30, color='darkred', alpha=0.3)

# people travelling together i.e. spouse, children etc
sns.countplot(x='SibSp', data=train)

# Fare of the travel
train['Fare'].hist(color='green', bins=30, figsize=(8,4))

#Cufflinks for plots

plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train, palette='winter')

# Removing the null values in Age column

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

# Now apply this function on Age column

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)             
train['Age'].isnull().value_counts()

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# Feature Engineering 
# - dropping Cabin column
train.drop('Cabin', inplace=True,axis=1)
train.head()

train['Embarked'].isnull().value_counts()


"""Converting Categorical Features i.e. sex and embarked columns"""

train.info()
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True).head()

# Dropping not required columns
train.drop(['Name','Sex','Embarked','Ticket'], axis=1, inplace=True)
train.head()

train = pd.concat([train, sex, embark], axis=1)
train.head()


"""Building a logistic regression model"""

# Train Test Split
train.drop('Survived', axis=1).head()
train.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,  y_train)
