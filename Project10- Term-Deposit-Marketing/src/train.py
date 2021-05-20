#train.py

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import joblib 
import pandas as pd
from sklearn import metrics
from imblearn.over_sampling import SMOTE

import config
import model_dispatcher


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # training data is where kfold is not equal to provided fold
    
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from dataframe and convert it to
    # target is y_new column in the dataframe
    X_train = df_train.drop("y_new", axis=1).values
    y_train = df_train['y_new'].values

    # Applying SMOTE method on training data 
    X_train_resample , y_train_resample = SMOTE().fit_sample(X_train, y_train)

    # for validation set
    X_valid = df_valid.drop("y_new", axis=1).values
    y_valid = df_valid['y_new'].values

    # initialize simple classifier from model_dispatcher
    clf = model_dispatcher.models[model]
    # fit the model on training data
    clf.fit(X_train_resample, y_train_resample)

    # create predictions for validation samples
    preds = clf.predict(X_valid)

    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    
    print(f"Fold={fold}, Accuracy={accuracy}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument(
	"--fold",
	type=int
	)
	parser.add_argument(
	"--model",
	type=str
	)

	args = parser.parse_args()
	
	run(
	fold=args.fold,
	model=args.model
	)
# python src/train.py --fold 0 --model XGB_Classifier
# python src/train.py --fold 1 --model Logistic_Regression
# python src/train.py --fold 2 --model RandomForest_Classifier
