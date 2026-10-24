import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Iris.csv")
print(dataset.head())
print(dataset.info())
print(dataset.isnull().sum())
print("Duplicated rows:",dataset.duplicated().sum())

dataset.fillna(dataset.mean(numeric_only=True),inplace=True)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

Y = le.fit_transform(Y)
print(Y)
print(le.classes_)

X_df = pd.DataFrame(
    X,
    columns=dataset.columns[:-1]
)
print(X_df.describe())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
X_original = X.copy()

X_train_df = pd.DataFrame(
    X_train,
    columns=dataset.columns[:-1]
)

print("BEFORE SCALING")
print(pd.DataFrame(
    X_original,
    columns=dataset.columns[:-1]
).describe().round(2))

print("\nAFTER SCALING (TRAINING DATA)")
print(X_train_df.describe().round(2))
"""