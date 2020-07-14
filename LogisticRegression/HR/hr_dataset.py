import numpy as np
from numpy.core.fromnumeric import _transpose_dispatcher
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('hr_dataset.csv')

# Based on satisfaction level, we will see whether employee will leave or not

sns.boxplot(x="left", y="satisfaction_level", data=train_df)

sns.boxplot(x="left", y="last_evaluation", data=train_df, notch=True)
