import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
print(cancer.keys())
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(df_feat.head())

X = df_feat
y = cancer['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=101)


model = SVC()
model.fit(X_train,y_train)
predict = model.predict(X_test)
print(confusion_matrix(y_test,predict))
print('\n')
print(classification_report(y_test,predict))

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

sns.heatmap(confusion_matrix(y_test,grid_predictions),annot=True,
            cmap='viridis',fmt='d')