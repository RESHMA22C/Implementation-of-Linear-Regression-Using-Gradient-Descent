# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load data from CSV into a DataFrame.

2.Extract features (x) and target values (y) from the DataFrame.

3.Convert all values to float for computation compatibility.

4.Initialize two StandardScalers: one for x, one for y.

5.Standardize (normalize) both x and y to have mean 0 and std 1.

6.Define linear_regression function that:

*Adds bias term (intercept) to x

*Initializes theta (weight vector)

7.Runs num_iters iterations of gradient descent to minimize MSE loss

8.Train the model by calling linear_regression(x_scaled, y_scaled)

9.Prepare a new input sample, scale it using the same x_scaler.

10.Make prediction using theta, then inverse transform the result using y_scaler.

11.Print predicted output in original scale.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Reshma C
RegisterNumber:  212223040168
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors =(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print("Name: Reshma C")
print("Reg No: 212223040168")
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")
```

## Output:
## DATA INFORMATION:
![image](https://github.com/user-attachments/assets/c801e52d-240b-4710-b809-d1b9feaded58)

## THE VALUE OF X:
![image](https://github.com/user-attachments/assets/baac3c20-8a84-43cb-841a-790278ff584f)

## THE VALUE OF Y:
![image](https://github.com/user-attachments/assets/ccb86e34-9188-4a63-ba1e-937aec97422a)

## THE VALUE OF X_SCALED:
![image](https://github.com/user-attachments/assets/b21dcd77-93e6-41a8-ab9b-b20591accdd0)


## THE VALUE OF Y_SCALED:

![image](https://github.com/user-attachments/assets/7d607b2c-1fe8-4f85-aacf-78c13c3a2355)


## PREDICTED VALUE:

![image](https://github.com/user-attachments/assets/954f9454-d958-4319-b62a-9088e15d74f4)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
