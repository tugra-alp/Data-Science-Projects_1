#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
from sklearn.datasets import load_breast_cancer
# %%
cancer = load_breast_cancer()
# %%
cancer.keys()
#print(cancer['DESCR']) #information of data
# %%
cancer['data'].shape
# %%
df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df.head()
# 0 : melignant(kötü huylu) , 1 : benign (iyi huylu)
# %%
sns.pairplot(df, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
# %%
sns.countplot(df['target'], label = "Count") 
#%%
sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df)

# %%
plt.figure(figsize=(20,10)) 
sns.heatmap(df.corr(), annot=True) 
# %%
X = df.drop(['target'],axis=1)
y = df['target']
# %%
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5)
# %%
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix
# %%
scv_model = SVC()
scv_model.fit(X_train, y_train)
# %%
y_predict = scv_model.predict(X_test)
# %%
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
# %%
print(classification_report(y_test, y_predict))
# %% İmproving model
#1-) Normalization data 2 -) C and gamma parameter are important for SCV
min_train = X_train.min()

# %%
range_train = (X_train - min_train).max()

# %%
X_train_scaled = (X_train - min_train)/range_train
#X_train_scaled #normalleştirilmiş data
# %% normalization for test data
min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test
# %%
scv_model.fit(X_train_scaled, y_train)
# %%
y_norm_predict = scv_model.predict(X_test_scaled)
# %%
cm1 = confusion_matrix(y_test, y_norm_predict)
sns.heatmap(cm1, annot= True)
# %%
print(classification_report(y_test,y_predict))
print(classification_report(y_test,y_norm_predict ))
# %%
# GRid Search method
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 
# %%
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose = 4)
grid.fit(X_train_scaled,y_train)
# %%
grid.best_params_
# %%
grid_predictions = grid.predict(X_test_scaled)
cm2 = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm2, annot=True)
# %%
print(classification_report(y_test,grid_predictions))
# %%
