import pandas as pd
# If you'd like to try PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = True


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot')

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

 
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
data = pd.read_csv('E:\Study\MPPData\Python\DAT210x-master\Module5\Datasets\\breast-cancer-wisconsin.data',header =None)
data.columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
data.head()

data.dtypes
data.nuclei = pd.to_numeric(data.nuclei, errors = 'coerce')

# Copy out the status column into a slice, then drop it from the main
# dataframe. 
labels = data['status'].copy()
data.drop('status', inplace = True, axis =1)

#
# If you goofed up on loading the dataset and notice you have a `sample` column,
# this would be a good place to drop that too if you haven't already.
data.drop('sample', inplace = True, axis =1)


# With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
data.fillna(data.mean(),inplace = True)


# Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.50, random_state=7)


# Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
from sklearn import preprocessing
#scaling = preprocessing.Normalizer()
#scaling = preprocessing.StandardScaler()
scaling = preprocessing.MinMaxScaler()
#scaling = preprocessing.MaxAbsScaler()
#scaling = preprocessing.RobustScaler()

scaling.fit(data_train)
data_train = scaling.transform(data_train)
data_test = scaling.transform(data_test)



# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print "Computing 2D Principle Components"
  from sklearn.decomposition  import PCA
  model = PCA(n_components=2)

  

else:
  print "Computing 2D Isomap Manifold"
  from sklearn import manifold
  model = manifold.Isomap(n_neighbors=5, n_components=2)    


# Train your model against data_train, then transform both
# data_train and data_test using your model. 
model.fit(data_train)
data_train = model.transform(data_train)
data_test = model.transform(data_test)


# Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.



# Implement KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
knmodel =  KNeighborsClassifier(n_neighbors = 3, weights = "distance")
knmodel.fit(data_train, label_train)


#Find the accuracy of the testing set (data_test and label_test).
print knmodel.score(data_test,label_test)


plotDecisionBoundary(knmodel, data_test, label_test)
