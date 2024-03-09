# import pandas and model_selection module of scikit-learn
# Creating stratified k-fold dataset for imbalanced data
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Training data path
    df = pd.read_csv("input/pre-processed.csv")
    
    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # fetch targets 
    y = df['y_new'].values
    
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
        
    # save the new csv with kfold column
    df.to_csv("input/termDeposit_stratified_folds.csv", index=False)
