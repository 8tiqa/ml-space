import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans


# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

def doKMeans(df):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', alpha=0.3)

  df = df[['Longitude', 'Latitude']]

  model = KMeans(n_clusters=7)
  model.fit(df)

  centroids = model.cluster_centers_
  ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
  print centroids


# Data source: GAMBLING data from Chicago crimes
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2/data
df = pd.read_csv("E:\Study\MPPData\Python\DAT210x-master\Module5\Datasets\Crimes_-_2001_to_present.csv", index_col=0)
df.dropna(axis=0,inplace=True)

# Coerce the 'Date' feature (which is currently a string object) into real date,
df.Date = pd.to_datetime(df.Date, errors = 'coerce')


doKMeans(df)

#Filter out the data so that it only contains samples that have
# a Date > '2011-01-01
df_2011 = df[df.Date > '2011-01-01']
doKMeans(df_2011)


plt.show()


