import pandas as pd

#Dataset source:
#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names
#1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
#2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
#3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
#4. bruises?: bruises=t,no=f 
#5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
#6. gill-attachment: attached=a,descending=d,free=f,notched=n 
#7. gill-spacing: close=c,crowded=w,distant=d 
#8. gill-size: broad=b,narrow=n 
#9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
#10. stalk-shape: enlarging=e,tapering=t 
#11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
#12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
#13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
#14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
#15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
#16. veil-type: partial=p,universal=u 
#17. veil-color: brown=n,orange=o,white=w,yellow=y 
#18. ring-number: none=n,one=o,two=t 
#19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
#20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
#21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
#22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

# Load up the mushroom dataset into dataframe 'X'
path ='E:\Study\MPPData\Python\DAT210x-master\Module6\Datasets\\agaricus-lepiota.data'
X = pd.read_csv(path, header = None)
X.columns = ['label','cap-shape','cap-surface','cap-color',' bruises',' odor','gill-attachment','gill-spacing', 'gill-size','gill-color',
           'stalk-shape','stalk-root','stalksa','stalkbr','color_above','color_below','veil-type','veil-color','ring-nuber','ring-type','spore','population','habitat']
X.head()

# Remove rows containing missing values 
X = X.replace('?', np.nan)
print X[pd.isnull(X).any(axis=1)]
X.dropna(axis = 0, inplace=True)

print X.shape

# Split data and label
y = X["label"]
X = X.drop("label", axis = 1)
y = y.map({ 'e':0 , 'p':1})


# Encode the entire dataset as binary vectors
X = pd.get_dummies(X)


# Split your data into test / train sets
from sklearn.model_selection import train_test_split
X_train, X_test,  y_train, y_test = train_test_split(X,y, test_size=0.30, random_state = 7)


#Create an DT classifier. 
from sklearn import tree
model = tree.DecisionTreeClassifier()


#train the classifier on the training data / labels:
model.fit(X_train, y_train)
    
#score the classifier on the testing data / labels:
score = model.score(X_test, y_test)
print "High-Dimensionality Score: ", round((score*100), 3)


# Render the .DOT and use: http://webgraphviz.com/ to view the tree
tree.export_graphviz(model.tree_, out_file='tree.dot', feature_names=X.columns)


