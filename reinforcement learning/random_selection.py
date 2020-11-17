# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('./data/Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    

# Visuaising the results - Histogram
plt.hist(ads_selected)
plt.title("History of ads Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

