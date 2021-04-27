# Overwiev of Projects

Hi, I am Tugra and I'm new here :blush: , so here is the small portfolio for you. I will share more porjects soon.
You can check description of whole my projects that i pushed.
 
## [Project 1: Bengaluru House Project : Predict house Prices](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project1-Bengaluru%20House%20Project)

In this project, I actually try a low-code python module which is LazyPredict to see how it works. LazyPredict is allow us to see **which model can fit better**, with a few line codes and without any parameter tuning. Thus you get some insight which model or models can fit your data before using these models with hyperparameter tuning.
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

## [Project 2: Celebrity Face Recognition (End-to-End) : Image Classification with SVM](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project2-Celebrity%20Face%20Recognition)
In this machine learning project, I classify celebrity personalities. I restrict classification to only 5 people. This project includes from data collection **(Image Scrapping)** to Deployment on AWS. Random Forest, Logistic Regression and Support Vector Machines alogrithm were used for this study, and GridSearchCV method was used for model selection with tuning parameters.

**Choosen People:**
1. Cristiano Ronaldo
2. Cheki Chen
3. Brad Pitt
4. Johnny Depp
5. Lionel Messi

![](https://github.com/tugra-alp/Data-Science-Projects/blob/main/Project2-Celebrity%20Face%20Recognition/Project%20Outcome%20Screenshots/Celebrity%20Person%20Classifier%20Ex1.png)

**Here is the folder structure:**
* **UI:** This contains ui website code 
* **server:** Python flask server 
* **model:** Contains python notebook for model building 
* **google_image_scrapping:** Code to scrap google for images 
* **datasets:** Dataset used for our model training which includes celebrity images 

**Technologies used in this project:**
1. Python :arrow_lower_left:
2. Numpy and OpenCV for data cleaning :arrow_lower_left:
3. Matplotlib & Seaborn for data visualization :arrow_lower_left:
4. Sklearn for model building :arrow_lower_left:
5. Jupyter notebook, visual studio code as IDE :arrow_lower_left:
6. Python flask for http server :arrow_lower_left:
7. HTML/CSS/Javascript for UI :arrow_lower_left:

**A Screenshot after model deployment**

![](https://github.com/tugra-alp/Data-Science-Projects/blob/main/Project2-Celebrity%20Face%20Recognition/Project%20Outcome%20Screenshots/Model%20Deployment%20on%20AWS.jpg)

## [Project 3: Data Analysis Project On TABLEAU : Sales Insight](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project3-TABLEAU%20Data%20Analysis%20Project)
## [Project 4: Breast Cancer : ML Classification Project via SVC](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project4-Breast%20Cancer)
## [Project 5: Fashion MNIST Dataset : Image Classification with CNN](https://github.com/tugra-alp/Data-Science-Projects/tree/main/Project5-Fashion%20Mnist)
