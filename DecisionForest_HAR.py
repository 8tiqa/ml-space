import pandas as pd
import time

# DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


# Load up the dataset into dataframe 'X'
path = 'E:\Study\MPPData\Python\DAT210x-master\Module6\Datasets\\dataset-har-PUC-Rio-ugulino.csv'
X = pd.read_csv(path, delimiter = ';')
X.head()


# Encode the gender column, 0 as male, 1 as female
X.gender.unique()
X.gender = X.gender.map({'Man':0,'Woman':1 })


# Clean up any column with commas in it
# so that they're properly represented as decimals instead
X = X.replace(',', '.',regex = True)

# Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic
# If you find any problematic records, drop them before calling the
# to_numeric methods above...

print X.dtypes
X = X.drop(X.index[[122076]], axis = 0)
X.z4 = pd.to_numeric(X.z4, errors = "raise")
X.body_mass_index = pd.to_numeric(X.body_mass_index, errors = "raise")
X.how_tall_in_meters = pd.to_numeric(X.how_tall_in_meters, errors = "raise")


# Encode your 'y' value as a dummies version of your dataset's "class" column
y = pd.get_dummies(X['class'])
X.drop(['user','class'], inplace = True, axis=1)

print X.describe()


# Remove missing values
X = X.replace('?', np.nan)
print X[pd.isnull(X).any(axis=1)]
X.dropna(axis = 0, inplace=True)



# Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30, max_depth =10, oob_score=True, random_state=0)


# Split your data into test / train sets
# Your test size can be 30% with random_state 7
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size = 0.30, random_state = 7)

#train your model on your training set
print "Fitting..."
s = time.time()
model.fit(X_train, y_train)
print "Fitting completed in: ", time.time() - s


# Display the OOB Score of your data
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)

# score your model on your test set
print "Scoring..."
s = time.time()
score = model.score(X_test, y_test)
print "Score: ", round(score*100, 3)
print "Scoring completed in: ", time.time() - s

