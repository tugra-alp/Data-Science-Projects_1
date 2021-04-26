#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#%%
data = pd.read_csv("preProcessedData.csv")
data = data.drop('Unnamed: 0',axis='columns')
#%% Train-Test Split
X = data.drop(['price'],axis='columns')
y = data.price

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
#%% 
from lazypredict.Supervised import LazyRegressor #to run almost all regression algorithm in a few rows
# we will use regression algorithms(42) and choose the best ones.
# fit all models with LazyRegressor module to get insight which algorithms can fit to our data
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
models 
# 1-) MLPregressor in Lazypredict =  Adj. R2: 0.87 RMSE: 23.38 
# 2-) BayesianRidge in Lazypredict = Adj. R2: 0.84 RMSE: 26.65
# 3-) Linear Regression in Lazypredict = Adj. R2: 0.84 RMSE: 26.67
#%% --- MLP Regression Model Building ----
from sklearn.neural_network import MLPRegressor #Multi-Layer Perceptron
from sklearn.preprocessing import StandardScaler
# Firstly, we need to standardize our data for MLP 
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
# MLP algorithm with default parameters
mlp_model = MLPRegressor(random_state=1).fit(X_train_scaled, y_train)

before_tuned_y_pred = mlp_model.predict(X_test_scaled)
before_tuned_RMSE = np.sqrt(mean_squared_error(y_test, before_tuned_y_pred)) #RMSE: 22.69
before_tuned_R2 = r2_score(y_test,before_tuned_y_pred) # R2: 0.90
#%% --- Model Tuning With GridSearcCV ----
mlp_params = {
                'alpha': [0.0001, 0.1, 0.001, 0.02, 0.005],
                'solver': ['adam', 'lbfgs'],
                'learning_rate': ['constant','adaptive'],
                'activation': ['relu'],
                'max_iter': [300,500]
            }

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 5)

mlp_cv_model.fit(X_train_scaled, y_train)
print('Best Paramter Found:\n' , mlp_cv_model.best_params_)

#%% -- recreate the model with best fit parameters -- 
mlp_tuned = MLPRegressor(alpha = 0.02, 
                         learning_rate='constant',
                         activation = 'relu',
                         solver='adam',
                         max_iter = 300,
                         random_state = 1).fit(X_train_scaled, y_train)

after_tuned_y_pred = mlp_tuned.predict(X_test_scaled)
after_tuned_RMSE = np.sqrt(mean_squared_error(y_test, after_tuned_y_pred)) # RMSE :22.45
after_tuned_R2 = r2_score(y_test,after_tuned_y_pred) # R^2 of Tuned Model: 0.902
# %%
