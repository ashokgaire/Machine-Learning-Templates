################ Polynomial linear regression ###############


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("data/Position_Salaries.csv")
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:,2].values


#spliting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3 , random_state=0)


#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)


#Fitting polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly =  poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)



# Visualizing the Linear regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color= 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


# Visualizing the Polynomial Regression results
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color= 'green')
plt.title('Truth or Bluff (ploynomial Regression)')
plt.xlabel('Position label')
plt.ylabel('Salary')
plt.show()


# Predicting a new result with Linear Regression
b = lin_reg.predict(np.array([6.5]).reshape(-1, 1))

# Predicting a new result with Polynomial Regression

a= lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(-1, 1)))


####################### THEORY ##############################


'''
 equation  Y = b0 + b1x1 + b2x1^2 + ......

'''