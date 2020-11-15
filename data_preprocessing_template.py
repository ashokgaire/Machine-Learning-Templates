# Data Preprocessing Template

#import 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("Data.csv")
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:,3:4].values

"""
#Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])



transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
"""

#spliting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2 , random_state=0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""









































