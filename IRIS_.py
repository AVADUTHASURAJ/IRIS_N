
from sklearn.metrics import roc_curve, auc
import json
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = open('algoparams_from_ui.json','r')
parsed_json = data.read()
data.close()
parsedjson = json.loads(parsed_json)
iris = pd.read_json(parsed_json)

df = pd.read_csv('iris.csv')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df.iloc[:,0:4]
Y = df['species']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.50)

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(max_depth=7,random_state=1)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('metrics')
plt.legend(loc="lower right")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.linear_model import Ridge

model = Ridge(max_iter=50)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.linear_model import Lasso

model = Lasso(max_iter=50)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.linear_model import ElasticNet

model = ElasticNet(max_iter=50)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=50)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=25,min_samples_leaf=5)
model.fit(X_train,Y_train)

Y_prediction = (model.predict(X_test))
print('prediction: ',Y_prediction)
print('Accuracy: ',model.score(X_test,Y_test)*100)

fpr, tpr, threshold = roc_curve(Y_test,Y_prediction,pos_label=2)
roc_auc= auc(fpr,tpr)
print(roc_auc)

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()