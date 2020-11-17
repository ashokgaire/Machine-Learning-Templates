############## Random Forest Regression ####################################
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("data/Position_Salaries.csv")
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:,2].values


#Fitting Simple Vector Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state= 0)
regressor.fit(X,Y)


# Predicting the Test set results
Y_pred = regressor.predict(np.array([6.5]).reshape(-1, 1))


# Visuaizing the Regression results (for higher resolution and smooth curves)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff ( Random forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()




































































# Algorithm  ########################################### 

'''
step 1: Pick at random K data points from the Training set.
step 2: Buid the Decision Tree associated to these K data points.
step 3: hoose the number Ntree of tress you want to build and repeat step 1 & 2
step 4: for a new data point, make each one of your Ntree trees predict the value of Y to for the data point
in question, and assign the new data point the average across all of the predicted Y values.
'''