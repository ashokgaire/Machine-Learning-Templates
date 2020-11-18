########################### Artificial Neural Network ###################


# Installing Theano
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git ( works with gpu)

# install tensorflow
#instal keras
#pip install --upgrade keras


# Data preprocessing

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('./data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


#Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [1]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())

X = X[:, 1:]


# Spliting the dataset into the traning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


####  ANN ##########

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialising the ann
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation='relu', input_dim = 11 ))

# Adding the second hidden layer
classifier.add(Dense(units = 6, activation='relu'))


# Addinig the output layer
classifier.add(Dense(units = 1, activation='sigmoid'))

# compling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= [ 'accuracy' ] )

# fitting the ann to the traning set
classifier.fit(X_train, y_train, batch_size= 10 , nb_epoch = 100)



### making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


























######################## Algorithm ############################################


'''
################## Training ANN with Stochastic Gradient Descent ############################

setp 1 : randomly initialise the weights to smal numbers close to 0 ( but not 0)
step 2 : Input the first observation of your dataset in the input layer, each feature in one input mode.
step 3 : Fordward-Propagation: from left to right,the neurons are activated in a way that the impact of each 
neuron's activation is limited by the weights. Propagate the actvation until getting the predicted result y.
step 4: Compare the predicted result to the actual result. Measure the generated error.
setp 5: Back-Propagation: from right to left , the error is back-propagated. Update the weights according to how much they are responsible for the error . Ther learning rate
decides by how much we update the weights.
step 6: Repeat Steps 1 t0 5 and update the weights after each observation (Reinforcement Learning). Or repeat steps 1 to % but update the weights only after a bach 
of observations ( Batch Learning)
step 7 : when the whole training set passed through the ANN, that makes an epoch . redo more epochs












'''





























   

