import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def load_housing_data():
    return pd.read_csv('housing.csv')

housing = load_housing_data()
housing.head()
print(housing.head())

housing.info()

housing["ocean_proximity"].value_counts()

housing.describe()

housing.hist(bins=50,figsize=(20,15))
plt.show()

# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)    
# len(train_set)
# len(test_set)


train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


len(train_set)
len(test_set)


housing["income_cat"] = pd.cut(housing["median_income"], 
                                bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels= [1,2,3,4,5])

housing["income_cat"].hist()     

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.15)


housing.plot(kind="scatter", x="longitude", y="latitude")


