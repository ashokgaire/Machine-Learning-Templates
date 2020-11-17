##################### Classification Template ##########################

#import 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("./data/Social_Network_Ads.csv")
X = datasets.iloc[:, 2:4].values
Y = datasets.iloc[:,4].values


#Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#spliting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


#### fiting  the classifier to the  training set
from sklearn.naive_bayes  import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# predicting the Test set results
y_pred = classifier.predict(X_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop=X_set[:, 0].max() +1, step =0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop=X_set[:, 1].max() +1, step =0.01))


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
             .reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('NV ( Traning set)')
plt.xlabel('Agel')
plt.ylabel('Salary')
plt.show()


