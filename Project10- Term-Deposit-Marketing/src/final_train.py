#final_train_model.py
#RandomForest with tuned hyperparamters
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import joblib 
import pandas as pd

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import config

def run(fold):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    X_train = df_train.drop("y_new", axis=1).values
    y_train = df_train['y_new'].values

    # Applying SMOTE method on training data
    X_train_resample, y_train_resample = SMOTE().fit_sample(X_train,y_train)

    # similarly, for validation, we have
    X_valid = df_valid.drop("y_new", axis=1).values
    y_valid = df_valid['y_new'].values

    # XGBoost Algoritm will applied.
    # initialize simple classifier
    clf = RandomForestClassifier(n_jobs=-1, 
                                 n_estimators = 100, 
                                 max_depth = 10 ,
                                 criterion = 'gini')
    # fit the model on training data
    clf.fit(X_train_resample, y_train_resample)

    # create predictions for validation samples
    preds = clf.predict(X_valid)

    # calculate & print accuracy , f1 and roc-auc score
    accuracy = metrics.accuracy_score(y_valid, preds)
    f1_score = metrics.f1_score(y_valid, preds)
    roc_auc = metrics.roc_auc_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    # save the model as pkl file
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f"clf_{fold}.pkl")
    )

if __name__ == "__main__":
        run(fold=0)
        run(fold=1)
        run(fold=2)
        run(fold=3)
        run(fold=4)

#run :  python src/final_train_model.py 