# Importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib

# Importing dataset
from sklearn.datasets import  load_boston

# Importng Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Loading boston dataset in variable
boston = load_boston()

type(boston)
#sklearn.utils.Bunch

boston.keys()
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

boston.DESCR

boston.feature_names
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

boston.target

data = boston.data
type(data)
data.shape

# Converting into pandas dataframe

data = pd.DataFrame(data=data, columns=boston.feature_names)
data.head()

data['Price'] = boston.target
data.head()

""" Understand your data"""
data.describe()
data.info()
data.isnull().sum()
# data.isnull().count()


""" Data Visualization """
""" Shift + Alt + A """

sns.pairplot(data)

""" Distribution plots """

rows = 2
cols = 7
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,4))
col = data.columns 
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(data[col[index]] ,ax=ax[i][j], k)
        index  = index+1
plt.tight_layout()


# Corelation matrix
corrmat = data.corr()

# Plotting correlation data

fig, ax = plt.subplots(figsize=(18,10))
sns.heatmap(corrmat, annot=True, annot_kws={'size':'12'})

# method to get correlated data

def getCorrelatedFeature(corrdata, threshold):
    feature=[]
    value=[]

    for i, index in enumerate(corrmat.index):
        if abs(corrdata[index]) > threshold:
            feature.append(index)
            value.append(corrdata[index])
    df = pd.DataFrame(data=value, index=feature, columns=['Corr Values'])        
    return df

threshold = 0.50
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value

corr_value.index.values
correlated_data = data[corr_value.index]
correlated_data.head()

""" Pairplot and Corrmat of correlated data """
sns.pairplot(correlated_data)
plt.tight_layout()

sns.heatmap(correlated_data.corr(), annot=True, annot_kws={"size":"12"})


""" Shuffle and Split the data """
X = correlated_data.drop(labels=['Price'], axis=1)
y = correlated_data['Price']
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape


""" Let's train the model """
model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

df = pd.DataFrame(data=[y_predict, y_test])
df.T


from sklearn.metrics import r2_score

score = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("R2 score: ", score)
print("MAE : ",mae)
print("MSE",mse)

""" Store Feature Performance """

total_features = []
total_feature_name = []
selected_correlation_value = []
r2_scores = []
mae_value = []
mse_value = []

def performance_metrics(features, th, y_true, y_pred):
    score = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    total_features.append(len(features)-1)
    total_feature_name.append(str(features))
    selected_correlation_value.append(th)
    r2_scores.append(score)
    mae_value.append(mae)
    mse_value.append(mse)

    metrics_dataframe = pd.DataFrame(data=[total_feature_name, total_features, selected_correlation_value, r2_scores, mae_value, mse_value],
                        index=['feature names','total_feature','corr_value','r2_score','MAE', 'MSE'])
    return metrics_dataframe.T

performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)    



""" Regression plot of the features correlated with the house price """
rows = 2
cols = 2
fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(16,4))

col = correlated_data.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.regplot(x=correlated_data[col[index]], y=correlated_data["Price"], ax=ax[i][j])
        index = index + 1
fig.tight_layout()            




""" Let's find out other combination of columns to get better accuracy with >60% """
corrmat['Price']

threshold = 0.60
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value

correlated_data = data[corr_value.index]
correlated_data.head()


def get_y_predict(corrdata):
    X = corrdata.drop(labels=['Price'], axis=1)
    y = corrdata['Price']

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict


y_predict = get_y_predict(correlated_data)

performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


""" Let's find out other combination of columns to get better accuracy with >60% """
corrmat['Price']
threshold = 0.70
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value

correlated_data = data[corr_value.index]
correlated_data.head()

y_predict = get_y_predict(correlated_data)
performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict)


""" Selecting only RM feature """
correlated_data  = data[['RM','Price']]
correlated_data.head()

y_predict = get_y_predict(correlated_data)
performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict) 

""" Let's find out other combination of columns to get better accuracy > 40% """

threshold = 0.40
corr_value = getCorrelatedFeature(corrmat['Price'], threshold)
corr_value

correlated_data = data[corr_value.index]
correlated_data.head()


y_predict = get_y_predict(correlated_data)
performance_metrics(correlated_data.columns.values, threshold, y_test, y_predict) 



""" Defining Performance Metrics """

# plotting learning curve

from sklearn.model_selection import learning_curve, ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross Validation Score")

    plt.legend(loc='best')
    return plt

X = correlated_data.drop(labels=['Price'], axis=1)
y = correlated_data['Price']

title = "Learning Curves(Linear Regression) " + str(X.columns.values)

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = LinearRegression()
plot_learning_curve(estimator, title, X, y, ylim=(0.7,1.01), cv=cv , n_jobs=4)

plt.show()

