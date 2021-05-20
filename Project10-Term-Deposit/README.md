## Term Deposit Marketing Project : Predicting Customers Subscription with Using Machine Learning 

### Data Description 

The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.

### Folder Structure & Description

1. input/:  This folder consists of all the input files and data for this project.

- term-deposit-marketing-2020 : original dataset
- pre-processed: Saved as csv file after performed exploraty data analysis and feature engineering.
- termDeposit_stratified_folds : Saved as csv file after split the pre-processed.csv with using stratified k-fold cv. This will be used while model training.

2. src/: This file include all the python scripts associated with the project here.

 **create_stratified_kfold.py :** Split the pre-proccessed.csv file into 5 folds and saved as csv file whcih name is termDeposit_stratified_folds.csv
 **rs_model_selection.py :** Model selection with using 5 fold cross validation. (Random Search CV method )
 **train.py and final_train.py :** For model training with using best hyperparamters.
 **config.py :** includes input and model output file paths
 **model_dispatcher.py :** includes selected models and their best paramters for train.py file. 

3. jupyter-lab/ : includes jupyter-lab files.

 **pre-processing&EDA :** All pre processing, EDA and feature angineering part to prepare data for training.

4. models/ : Includes model output filse as .pkl format.

5. screenshot-of-model-results/ : Includes screenshot of test accuracy of final model.

### Description of Project 

