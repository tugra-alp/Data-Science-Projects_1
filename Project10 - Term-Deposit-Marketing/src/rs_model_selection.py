# rs_model_selection.py
# Random Search 
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("input/pre-processed.csv")
    
    
    X = df.drop("y_new", axis=1).values
    # and the targets
    y = df['y_new'].values
    
    # train test split
    X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3)
    
    # SMOTE oversampling method on imbalanced data
    X_train_resample , y_train_resample = SMOTE().fit_sample(X_train, y_train)
    
    # algorithms and parameters
    algos = {
        'RandomForestClassifier' : {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': np.arange(100, 1500, 100),
                'max_depth': np.arange(1, 31),
                'criterion': ["gini", "entropy"]
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(objective='binary:logistic', verbosity = 0),
            'params': {
                'n_estimators': np.arange(100, 1500, 100),
                'max_depth':  [3, 5, 6, 7, 10],
                'learning_rate': [0.3,0.1, 0.01, 0.05, 0.001],
            }
        },
        'LogisticRegression' : {
            'model' : LogisticRegression(),
            'params': {
                'C' : [0.1, 0.01, 0.001, 10, 100]
            }
        }
    }
    scores = []
    for algo_name, config in algos.items():
        model_selection =  RandomizedSearchCV(config['model'], config['params'], n_iter=20, 
                                              n_jobs = -1, scoring="accuracy" ,
                                              cv=5)
                                              
                                              
        model_selection.fit(X_train_resample,y_train_resample)
        scores.append({
            'model': algo_name,
            'best_accuracy': model_selection.best_score_,
            'best_params': model_selection.best_params_
        })
        # save the train accuracies as csv file (path: models/)
        results = pd.DataFrame(scores,columns=['model','best_accuracy','best_params'])
        results.to_csv('models/train_accuracy.csv')
        
        print(tabulate(results,  headers = 'keys', tablefmt = 'psql'))