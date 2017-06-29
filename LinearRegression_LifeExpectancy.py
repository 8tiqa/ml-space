import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot') 

def drawLine(model, X_test, y_test, title):
  # function to plot test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c='g', marker='o')
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)

  print "Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0]
  print "Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0]
  print "Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0]

  score = model.score(X_test, y_test)
  title += " R2: " + str(score)
  ax.set_title(title)


  plt.show()


#Load Dataset
X = pd.read_csv('E:\Study\MPPData\Python\DAT210x-master\Module5\Datasets\life_expectancy.csv', delimiter = "\t")
X.describe()

# Create linear regression model 
from sklearn import linear_model
model = linear_model.LinearRegression()

# Slice out Data: Set X_train to be year values
# LESS than 1986, and y_train to be corresponding WhiteMale age values.
X_train = X[X.Year < 1986].Year
y_train = X[X.Year < 1986].WhiteMale
X_train = X_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)


# Train model then pass it into drawLine with training
# set and labels. Title it "WhiteMale". drawLine will output
# to the console a 2014 extrapolation / approximation for what it
# believes the WhiteMale's life expectancy in the U.S. will be...
# given the pre-1986 data you trained it with. It'll also produce a
# 2030 and 2045 extrapolation.
#
model.fit(X_train, y_train)
drawLine(model,X_train,y_train, "WhiteMale")

# Print the actual 2014 WhiteMale life expectancy from your
# loaded dataset
print X[X.Year == 1986].WhiteMale

 
# Repeat the process, but instead of for WhiteMale, this time
# select BlackFemale. Create a slice for BlackFemales, fit the
#model, and then call drawLine. Lastly, print out the actual 2014
# BlackFemale life expectancy
X_train = X[X.Year < 1986].Year
y_train = X[X.Year < 1986].BlackFemale
X_train = X_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)

model.fit(X_train, y_train)
drawLine(model,X_train,y_train, "BlackFemale")

print X[X.Year == 1986].BlackFemale


# print out a correlation matrix for your entire
# dataset, and display a visualization of the correlation
# matrix

print X.corr()

plt.imshow(X.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
plt.show()




#
# INFO + HINT On Fitting, Scoring, and Predicting:
#
# Here's a hint to help you complete the assignment without pulling
# your hair out! When you use .fit(), .score(), and .predict() on
# your model, SciKit-Learn expects your training data to be in
# spreadsheet (2D Array-Like) form. This means you can't simply
# pass in a 1D Array (slice) and get away with it.
#
# To properly prep your data, you have to pass in a 2D Numpy Array,
# or a dataframe. But what happens if you really only want to pass
# in a single feature?
#
# If you slice your dataframe using df[['ColumnName']] syntax, the
# result that comes back is actually a *dataframe*. Go ahead and do
# a type() on it to check it out. Since it's already a dataframe,
# you're good -- no further changes needed.
#
# But if you slice your dataframe using the df.ColumnName syntax,
# OR if you call df['ColumnName'], the result that comes back is
# actually a series (1D Array)! This will cause SKLearn to bug out.
# So if you are slicing using either of those two techniques, before
# sending your training or testing data to .fit / .score, do a
# my_column = my_column.reshape(-1,1). This will convert your 1D
# array of [n_samples], to a 2D array shaped like [n_samples, 1].
# A single feature, with many samples.
#
# If you did something like my_column = [my_column], that would produce
# an array in the shape of [1, n_samples], which is incorrect because
# SKLearn expects your data to be arranged as [n_samples, n_features].
# Keep in mind, all of the above only relates to your "X" or input
# data, and does not apply to your "y" or labels.

