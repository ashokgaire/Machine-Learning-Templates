##################### K means  ##########################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import dataset
datasets = pd.read_csv("./data/mail.csv")
X = datasets.iloc[:, [3,4]].values


#Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title(" The Elbow Method")
plt.xlabel('number of clusters')
plt.ylabel('Wcss')
plt.show()
    
    

# Applying k-means to the mai dataset
kmeans = KMeans(n_clusters =5, init = 'k-means++', max_iter = 300, n_init =10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Cluster 1')





####################### ALgorithm ###################

'''

step 1: Choose the number K of clusters
step 2:Select at random K points, the centroids (not necessarily from your dataset)
setp 3: Assign each data point to the closest centroid  => that forms K clusters
step 4: Compute and place the new centroid of each cluster
step 5:Reassign each data point to new closest centroid.

'''





































