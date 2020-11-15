#################### Decision Tree regression ##################

 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("data/Position_Salaries.csv")
X = datasets.iloc[:, 1:2].values
Y = datasets.iloc[:,2].values

'''
#Feature Scaling  (at this time svm desn't provide auto featur scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)'''


#Fitting Decision Treer Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)


# Predicting the Test set results
Y_pred = regressor.predict(np.array([6.5]).reshape(1, -1))


# Visuaizing the Training set Results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff ( Decison Tree Regression )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


























