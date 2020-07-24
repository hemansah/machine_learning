import pandas as pd 
import numpy as np

df = pd.read_csv('iphone_purchase.csv')
df.head()

# X contains all 3 independent variables
# y contains dependent variable "Purchased iPhone"
X = df.iloc[:,:-1].values
y = df.iloc[:, 3].values


# Convert gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender = LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])
X = np.vstack(X[:, :]).astype(np.float)


#Split data into training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fit KNN classifier
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
clf.fit(X_train, y_train)

# Make predictions
pred = clf.predict(X_test)


# Check accuracy of predictions
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, pred)
print("Accuracy score: ",accuracy)
precision = metrics.precision_score(y_test, pred)
print("Precision score: ", precision)
recall = metrics.recall_score(y_test, pred)
print("Recall score: ",recall)