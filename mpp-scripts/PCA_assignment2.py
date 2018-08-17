import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

plt.style.use('ggplot')
scaleFeatures = False

df = pd.read_csv("E:\Study\MPPData\Python\DAT210x-master\Module4\Datasets\kidney_disease.csv", index_col=0)
df = df.dropna(axis=0) 

labels = ['red' if i=='ckd' else 'green' for i in df.classification]

df = df[ ['bgr','wc','rc']]

df.dtypes
df.wc = pd.to_numeric(df.wc,errors='coerce')
df.rc = pd.to_numeric(df.rc,errors='coerce')


# PCA Operates based on variance. The variable with the greatest
# variance will dominate. Peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results.
df.describe()

if scaleFeatures: df = helper.scaleFeatures(df)

# Run PCA to reduce df to 2 components
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(df)
T = pca.transform(df)
  


# Plot the transformed data as a scatter plot. 
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


