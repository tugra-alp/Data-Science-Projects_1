# Overwiev of Projects
 
## [Project 1: Bengaluru House Project : Predict house Prices](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project1-Bengaluru%20House%20Project)

In this project, I actually try a low-code python module which is LazyPredict to see how it works. LazyPredict is allow us to see **which model can fit better**, with a few lines code and without any parameter tuning. Thus you get some insight which model or models can fit your data before using these model or models with hyperparameter tuning.
Here is the link of the [LazyPredict Documentation](https://lazypredict.readthedocs.io/en/latest/index.html)

- Get the result of regression algorithms in LazyPredict, here is the top 3 models fitted the data:
  1. MLPregressor in Lazypredict =  **Adj. R2:** 0.87 **RMSE:** 23.38 
  2. BayesianRidge in Lazypredict = **Adj. R2:** 0.84 **RMSE:** 26.65
  3. Linear Regression in Lazypredict = **Adj. R2:** 0.84 **RMSE:** 26.67
- Then I used **MLP(Multi-Layer Perceptron)** algorithms with Sklearn module to train data.
- GridSearchCV was used for tuning.
- Finally, I got the result of the test set.
   **RMSE :** 22.45
   **R2 of Tuned Model:** 0.902

## [Project 2: Celebrity Face Recognition (End-to-End) : Image Classification with SVM](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project2-Celebrity%20Face%20Recognition)
![](https://github.com/tugra-alp/Data-Science-Projects/blob/main/images/Celebrity%20Person%20Classifier%20Ex1.png)
In this machine learning project, I classify celebrity personalities. I restrict classification to only 5 people. This project includes from data collection(**Image Scrapping**) to Deployment on AWS.

1. Cristiano Ronaldo
2. Cheki Chen
3. Brad Pitt
4. Johnny Depp
5. Lionel Messi

**Here is the folder structure:**
* UI : This contains ui website code 
* server: Python flask server
* model: Contains python notebook for model building
* google_image_scrapping: code to scrap google for images
* images_dataset: Dataset used for our model training

**Technologies used in this project:**
1. Python 
2. Numpy and OpenCV for data cleaning
3. Matplotlib & Seaborn for data visualization
4. Sklearn for model building
5. Jupyter notebook, visual studio code and pycharm as IDE
6. Python flask for http server
7. HTML/CSS/Javascript for UI



## [Project 3: Data Analysis Project On TABLEAU : Sales Insight](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project3-TABLEAU%20Data%20Analysis%20Project)
## [Project 4: Breast Cancer : ML Classification Project via SVC](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project4-Breast%20Cancer)
## [Project 5: Fashion MNIST Dataset : Image Classification with CNN](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project5-Fashion%20Mnist)
