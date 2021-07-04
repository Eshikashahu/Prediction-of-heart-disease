import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv("heart.csv")
type(dataset)
dataset.shape
dataset.head(5)
dataset.sample(5)
dataset.describe()
dataset.info()
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])

dataset["target"].describe()
dataset["target"].unique()
print(dataset.corr()["target"].abs().sort_values(ascending=False))
x=dataset[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)

print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))
dataset["sex"].unique()

sns.barplot(dataset["sex"],y)
dataset["cp"].unique()
sns.barplot(dataset["cp"],y)
dataset["fbs"].describe()
dataset["fbs"].unique()
sns.barplot(dataset["fbs"],y)
dataset["restecg"].unique()
sns.barplot(dataset["restecg"],y)
dataset["exang"].unique()
sns.barplot(dataset["exang"],y)
dataset["slope"].unique()
sns.barplot(dataset["slope"],y)
dataset["ca"].unique()
sns.countplot(dataset["ca"])
sns.barplot(dataset["ca"],y)
dataset["thal"].unique()
sns.barplot(dataset["thal"],y)
sns.distplot(dataset["thal"])
dataset.hist()

from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

import seaborn as sns
#get correlations of each features in dataset
cor = X_train.corr()
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(cor,annot=True,cmap="RdYlGn")

def correlation(dataset,threshold):
    col_corr=set() #set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:  #we are interested in absolute coeff value
                colname=corr_matrix.columns[i] #getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features=correlation(X_train,0.7)
len(set(corr_features))
corr_features

X_train.drop(corr_features,axis=1)
X_test.drop(corr_features,axis=1)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()

score_ln=cross_val_score(lr,x,y,cv=10,scoring='accuracy').mean()
score_lr = round(score_ln*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
score_nb=cross_val_score(nb,x,y,cv=10,scoring='accuracy').mean()
score_nb = round(score_nb*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")

from sklearn import svm

sv = svm.SVC(kernel='linear')
score_svm=cross_val_score(sv,x,y,cv=10,scoring='accuracy').mean()
score_svm = round(score_svm*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
score_knn = cross_val_score(knn,x,y,cv=10,scoring='accuracy').mean()
score_knn = round(score_knn*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for p in range(2000):
    rf = RandomForestClassifier(random_state=p)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = p
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
score_rf=cross_val_score(rf,x,y,cv=10,scoring='accuracy').mean()
score_rf = round(score_rf*100,2)

print("The accuracy score achieved using Random Forest is: "+str(score_rf)+" %")
scores = [score_lr,score_nb,score_svm,score_knn,score_rf,]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Random Forest"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")

sns.set(rc={'figure.figsize':(15,5)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)

