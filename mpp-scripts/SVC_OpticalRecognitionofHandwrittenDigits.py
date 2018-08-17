import pandas as pd

# The Dataset comes from:
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

def load(path_test, path_train):
  with open(path_test, 'r')  as f: testing  = pd.read_csv(f)
  with open(path_train, 'r') as f: training = pd.read_csv(f)

  n_features = testing.shape[1]

  X_test  = testing.ix[:,:n_features-1]
  X_train = training.ix[:,:n_features-1]
  y_test  = testing.ix[:,n_features-1:].values.ravel()
  y_train = training.ix[:,n_features-1:].values.ravel()
  
  
  return X_train, X_test, y_train, y_test


def peekData(X_train):
  print "Peeking your data..."
  fig = plt.figure()

  cnt = 0
  for col in range(5):
    for row in range(10):
      plt.subplot(5, 10, cnt + 1)
      plt.imshow(X_train.ix[cnt,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
      plt.axis('off')
      cnt += 1
  fig.set_tight_layout(True)
  plt.show()


def drawPredictions(X_train, X_test, y_train, y_test):
  fig = plt.figure()

  # Make some guesses
  y_guess = model.predict(X_test)

  num_rows = 10
  num_cols = 5

  index = 0
  for col in range(num_cols):
    for row in range(num_rows):
      plt.subplot(num_cols, num_rows, index + 1)

      # 8x8 is the size of the image, 64 pixels
      plt.imshow(X_test.ix[index,:].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')

      # Green = Guessed right
      # Red = Fail!
      fontcolor = 'g' if y_test[index] == y_guess[index] else 'r'
      plt.title('Label: %i' % y_guess[index], fontsize=6, color=fontcolor)
      plt.axis('off')
      index += 1
  fig.set_tight_layout(True)
  plt.show()



path ='E:\Study\MPPData\Python\DAT210x-master\Module6\Datasets'
path_test= path + '\optdigits.tes'
path_train= path + '\optdigits.tra'
X_train, X_test, y_train, y_test = load(path_test, path_train)

import matplotlib.pyplot as plt
from sklearn import svm

# 
# Get to know your data. It seems its already well organized in
# [n_samples, n_features] form. Our dataset looks like (4389, 784).
# Also your labels are already shaped as [n_samples].
peekData(X_train)


# SVC classifier. Leave C=1, but set gamma to 0.001
# and set the kernel to linear. Then train the model on the training
# data / labels:
print "Training SVC Classifier..."
from sklearn.svm import SVC
svc = SVC(kernel = 'rbf', C = 1, gamma = 0.001)
svc.fit(X_train,y_train)


# Calculate the score of your SVC against the testing data
print "Scoring SVC Classifier..."
score = svc.score(X_test,y_test)
print "Score:\n", score


# Visual Confirmation of accuracy
drawPredictions(X_train, X_test, y_train, y_test)


# Print out the TRUE value of the 1000th digit in the test set
# By TRUE value, we mean, the actual provided label for that sample
true_1000th_test_value = y_test[1000]
print "1000th test label: "+ str(true_1000th_test_value)


#Predict the value of the 1000th digit in the test set.
# Was your model's prediction correct?
# INFO: If you get a warning on your predict line, look at the
# notes from the previous module's labs.
#
guess_1000th_test_value = svc.predict(X_test)[1000]
print "1000th test prediction: ", guess_1000th_test_value


# Use IMSHOW to display the 1000th test image, so you can
# visually check if it was a hard image, or an easy image
plt.imshow(X_test.ix[1000,:].values.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# Were you able to beat the USPS advertised accuracy score
# of 98%? If so, STOP and answer the lab questions. But if you
# weren't able to get that high of an accuracy score, go back
# and change your SVC's kernel to 'poly' and re-run your lab
# again.

# Were you able to beat the USPS advertised accuracy score
# of 98%? If so, STOP and answer the lab questions. But if you
# weren't able to get that high of an accuracy score, go back
# and change your SVC's kernel to 'rbf' and re-run your lab
# again.

# Were you able to beat the USPS advertised accuracy score
# of 98%? If so, STOP and answer the lab questions. But if you
# weren't able to get that high of an accuracy score, go back
# and tinker with your gamma value and C value until you're able
# to beat the USPS. Don't stop tinkering until you do. =).




