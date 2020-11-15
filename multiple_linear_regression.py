## Multiple Linear Regression

#import 
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("data/50_Startups.csv")
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:,4].values


#Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])


transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())


# Avoiding the Dummy variable Trap
X = X[:, 1:]


#spliting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2 , random_state=0)


# Fitting Multipe Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# Predicting the Test set results

y_pred = regressor.predict(X_test)



############## Building the optimal model using Backward Elimination ##########

import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),  values=X ,  axis = 1)

# step 1:
X_opt = X[:, [0,1,2,3,4,5]]  # all of them

#step 2:
regressor_OLS = sm.OLS(endog =Y , exog =X_opt ).fit()
regressor_OLS.summary()
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog=Y, exog = X_opt).fit()
regressor_OLS.summary()







































































############################Theories#############################################




'''

  equation : y = bo + b1x1 + b2x2 ---+ bnxn


     ######################### Backward Elimination ---###############################################
     
     step 1: select a significance level to stay in the model (eg SL = 0.05)
     setp 2: fit the full model with all possible predictors
     setp 3: consider the predictor with the highest P-value. if P > SL, go to step 4 otherwise go to fin.
     setp 4: Remove the predictor
     setp 5: fit model without this variable
     
     
     
     
     ############################## Fordward Selection ##########################
     
     step1: select a significance level to enter the model ( eg. SL = 0.05)
     step2: Fit all simple regression models y ~ Xn Select the one with the lowest P value
     setp3: keep this variable and fit all possible models with one extra predictor added to the one(s)
     setp 4:consider the predictor with the lowest P-value. if P < SL, got to STEP 3, otherwise go to FIN
     
     
     
     ######################### Biderectional Elimination ######################################
     
     setp 1 :selecta significance level to enter and to stay in the model
     e.g.: SLENTER = 0.05, SLSTAY = 0.05
     setp 2: perform the next step of Fordward Selection (new variables must have: P < SLENETR to enter)
     step 3: perform all steps of Backward Elimination ( old variables must have P < SLSTAY to stay )
     step 4: NO new variables can enter and no odd variabes can exit.
     
     '''
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
