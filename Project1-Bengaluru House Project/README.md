- Bengaluru House Project

In this project, I actually try a low-code python module which name is LazyPredict to see how it works. LazyPredict is allow us to see **which model can fit better**, with a few line codes and without any parameter tuning. Thus you get some insight which model or models can fit your data before using these models with hyperparameter tuning.
Here is the link of the [LazyPredict Documentation](https://lazypredict.readthedocs.io/en/latest/index.html)

Project steps:

- Get the result of regression algorithms in LazyPredict, here is the top 3 models fitted the data:
  1. MLPregressor in Lazypredict =  **Adj. R2:** 0.87 **RMSE:** 23.38 
  2. BayesianRidge in Lazypredict = **Adj. R2:** 0.84 **RMSE:** 26.65
  3. Linear Regression in Lazypredict = **Adj. R2:** 0.84 **RMSE:** 26.67
- Then I used **MLP(Multi-Layer Perceptron)** algorithms with Sklearn module to train data.
- GridSearchCV was used for hyperparameter tuning.
- Finally, I got the result of the test set.
   **RMSE :** 22.45
   **R2 of Tuned Model:** 0.902
