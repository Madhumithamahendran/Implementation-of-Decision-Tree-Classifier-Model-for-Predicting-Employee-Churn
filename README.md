# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by: MADHUMITHA M

RegisterNumber: 212222220020

import pandas as pd

df=pd.read_csv("/content/Employee.csv")

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])

df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y=df["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="entropy")

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
Initial data set:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/aa28e68f-3b3a-43e4-a6b3-76686e76f6b4)

Data info:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/9922d4a0-9867-4569-8adb-0eb7a06b0527)

Optimization of null values:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/8f3a797f-fd4b-4461-9168-7476c2a601eb)

Assignment value of x and y values:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/d656f49e-cc57-4ab0-9cbd-5fdf27181b40)

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/c937543c-a661-46fe-baa2-2ea68e2da5cf)

Converting string literals to numerical values using label recorder:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/6b73ce9a-118d-407c-b07e-cdc469ca1bc1)

Accuracy:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/cbb6a3c3-0930-4724-89f3-ab570386040c)

Prediction:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119394403/385201c7-f391-49bb-b803-829d324829c9)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
