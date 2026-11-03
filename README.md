# Implementation of Univariate Linear Regression
## AIM:
To implement univariate Linear Regression to fit a straight line using least squares.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y.
2. Calculate the mean of the X -values and the mean of the Y -values.
3. Find the slope m of the line of best fit using the formula. 
<img width="231" alt="image" src="https://user-images.githubusercontent.com/93026020/192078527-b3b5ee3e-992f-46c4-865b-3b7ce4ac54ad.png">
4. Compute the y -intercept of the line by using the formula:
<img width="148" alt="image" src="https://user-images.githubusercontent.com/93026020/192078545-79d70b90-7e9d-4b85-9f8b-9d7548a4c5a4.png">
5. Use the slope m and the y -intercept to form the equation of the line.
6. Obtain the straight line equation Y=mX+b and plot the scatterplot.

## Program:
```
/*
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by:Ranthiya S
RegisterNumber:212225230225
*/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Study hours and marks scored
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([20, 30, 40, 50, 60])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 hours of study
predicted_marks = model.predict([[6]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted marks for 6 hours:", predicted_marks[0])

# Plot the graph
plt.scatter(X, y, color="blue", label="Actual Marks")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks Scored")
plt.title("Study Hours vs Marks Scored")
plt.legend()
plt.grid(True)
plt.show()
```

## Output:
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/88e8f1d8-bd6e-4679-b8ad-c1ed2c0301ae" />



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
