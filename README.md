# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Naadira Sahar N
RegisterNumber: 212221220034
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
print("Placement data:")
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or col
print("Salary data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") #A Library for Large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score =(TP+TN)/
#accuracy_score(y_true,y_pred,normalize=False)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot (41)](https://user-images.githubusercontent.com/128135126/235361003-f4c121ae-ccbb-4ae8-8c8c-d2d79aaa2184.png)

![Screenshot (42)](https://user-images.githubusercontent.com/128135126/235361010-5afd2cca-6663-4cd2-a7ef-28a6edf47bc6.png)

![Screenshot (43)](https://user-images.githubusercontent.com/128135126/235361029-b32cb180-6cdb-4705-bdfe-92b78792c586.png)

![Screenshot (44)](https://user-images.githubusercontent.com/128135126/235361056-b935a3b9-3afa-4222-9ee5-9b383c3e8e8e.png)

![Screenshot (45)](https://user-images.githubusercontent.com/128135126/235361075-a20cd36b-92c8-4297-8ede-08cc69cbe801.png)

![Screenshot (46)](https://user-images.githubusercontent.com/128135126/235361099-afa5fdd0-eb32-42b4-93f9-39804846b88b.png)

![Screenshot (47)](https://user-images.githubusercontent.com/128135126/235361126-d240fb5b-d3a6-4226-bbd7-b8670fac7895.png)

![Screenshot (48)](https://user-images.githubusercontent.com/128135126/235361147-e48f02e1-93cd-4d38-aa9f-19ca7ecb28f6.png)

![Screenshot (49)](https://user-images.githubusercontent.com/128135126/235361183-fee3c44a-7b23-4c86-b2de-ada0389cd998.png)

![Screenshot (51)](https://user-images.githubusercontent.com/128135126/235361287-1c73101b-035d-40ec-b20f-4279387f4785.png)

![Screenshot (50)](https://user-images.githubusercontent.com/128135126/235361264-703784d8-19a5-4d9d-9ff2-19571a67a323.png)

![image](https://user-images.githubusercontent.com/128135126/235361325-ff2f9e56-5ece-456b-8bf7-690d65eb97db.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
