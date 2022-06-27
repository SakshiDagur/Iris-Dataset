import pandas as pd
df=pd.read_csv("C:/Users/CC-080/Downloads/iris.csv")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

rf=RandomForestClassifier(random_state=1)
lr=LogisticRegression(random_state=0)
mnb=MultinomialNB()
gbm=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
sv=svm.SVC()
nm=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=0)

logr=LogisticRegression()
X=df.drop('Species',axis=1)
Y=df['Species']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)
#print(X_train)
#print(X_test)
print(Y_train)
#print(Y_test)

train=logr.fit(X_train,Y_train)
Y_pred1=logr.predict(X_test)
print("logistic Regression")
print(accuracy_score(Y_test,Y_pred1))

train=mnb.fit(X_train,Y_train)
Y_pred2=mnb.predict(X_test)
print("Multinomial Naive Bayes")
print(accuracy_score(Y_test,Y_pred2))

train=rf.fit(X_train,Y_train)
Y_pred3=rf.predict(X_test)
print("logistic Regression")
print(accuracy_score(Y_test,Y_pred3))

train=lr.fit(X_train,Y_train)
Y_pred4=lr.predict(X_test)
print("linear Regression")
print(accuracy_score(Y_test,Y_pred4))

train=mnb.fit(X_train,Y_train)
Y_pred5=mnb.predict(X_test)
print("multinomial Regression")
print(accuracy_score(Y_test,Y_pred5))

train=dt.fit(X_train,Y_train)
Y_pred6=dt.predict(X_test)
print("Decision Tree")
print(accuracy_score(Y_test,Y_pred6))

train=gbm.fit(X_train,Y_train)
Y_pred7=gbm.predict(X_test)
print("gradient boosting method")
print(accuracy_score(Y_test,Y_pred7))