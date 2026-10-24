import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier #Provides the machine learning algorythme
from sklearn import metrics #evaluating the model

dataset = pd.read_csv("Iris.csv")
print(dataset.head())
print(dataset.info())

#Encoding the labels to numeric labels
dataset["Species"] = dataset["Species"].replace({
    "setosa":0,
    "versicolor":1,
    "virginica":2
})

#Separating the features from the target
Y = dataset["Species"]
X = dataset.drop("Species",axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
print(X_train.shape)
print(X_test.shape)

model = DecisionTreeClassifier(max_depth=3,random_state=1)

#Training the model

#The fit method will train the model, in this training, we give the correct flower names with the training measurements to the model. It looks for the pattern that helps to distinguish the 3 species
model.fit(X_train,Y_train)
#Ask the model to make the predictions, we give the model the measurements from our testing set but we do not give the answers, the model will predict the names of the flowers
predictions = model.predict(X_test)

#Checking the accuracy
print("accuracy: ",
      metrics.accuracy_score(Y_test,predictions))
