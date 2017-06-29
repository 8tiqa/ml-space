import pandas as pd
import numpy as np


#Load up the Dataset
path ='E:\Study\MPPData\Python\DAT210x-master\Module6\Datasets\parkinsons.data'
X = pd.read_csv(path)
X.drop('name', inplace =True, axis =1)
X.head()

#Splice out the status column into a variable y and delete it from X.
y = X.status
X.drop('status', inplace =True, axis =1)

#Perform a train/test split. 30% test group size, with a random_state equal to 7.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)


# Scaling
from sklearn import preprocessing
scaling = preprocessing.StandardScaler()
#scaling = preprocessing.Normalizer()
#scaling = preprocessing.MaxAbsScaler()
#scaling = preprocessing.MinMaxScaler()
#scaling = preprocessing.KernelCenterer()
scaling.fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

#Reducing dimensions
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 14)
#pca.fit(X_train)
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)

from sklearn.manifold import Isomap
isomap = Isomap(n_neighbors = 2, n_components = 6)
isomap.fit(X_train)
X_train = isomap.transform(X_train)
X_test = isomap.transform(X_test)

#Create a SVC classifier. Don't specify any parameters, just leave everything as default. 
#Fit it against your training data and then score your testing data.
from sklearn.svm import SVC
#svc = SVC(C = 1.65, gamma = 0.005)
#svc.fit(X_train,y_train)

## Calculate the score of your SVC against the testing data
#score = svc.score(X_test,y_test)
#print "Score:\n", score


# Program a naive, best-parameter search by creating nested for-loops.
best_score = 0
C_range = np.arange(0.05,2,0.05) 
gamma_range = np.arange(0.001, 0.1, 0.001)

for c in C_range:
    for gamma in gamma_range:
        svc = SVC(C=c, gamma = gamma)
        svc.fit(X_train,y_train)
        score = svc.score(X_test,y_test)
        if best_score<score:
            best_score = score
            print "Score: "+ str(score)+ "\t C: "+ str(c)+ "\t gamma: "+ str(gamma)
        
