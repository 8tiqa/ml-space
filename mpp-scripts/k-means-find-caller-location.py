
#Challenge:This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live and work at!

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import KMeans


matplotlib.style.use('ggplot') 

def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  exit()



df = pd.read_csv("E:\Study\MPPData\Python\DAT210x-master\Module5\Datasets\CDR.csv")
df.head()

df.dropna(axis=0,inplace=True)
df.CallDate = pd.to_datetime(df.CallDate, errors = 'coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors = 'coerce')

# distinct list of "In" phone numbers (users) 
distinct_In = df.In.unique().tolist()

# user1 that filters to only include dataset records where the
# "In" feature (user phone number) is equal to the first number on your unique list above;
user1 = df[df.In == distinct_In[0]]

# Plot all the call locations
user1.plot.scatter(x='TowerLon', y='TowerLat', c='gray', alpha=0.1, title='Call Locations')
showandtell()  

#
# INFO: The locations map above should be too "busy" to really wrap your head around. This
# is where domain expertise comes into play. Your intuition tells you that people are likely
# to behave differently on weekends:
#
# On Weekends:
#   1. People probably don't go into work
#   2. They probably sleep in late on Saturday
#   3. They probably run a bunch of random errands, since they couldn't during the week
#   4. They should be home, at least during the very late hours, e.g. 1-4 AM
#
# On Weekdays:
#   1. People probably are at work during normal working hours
#   2. They probably are at home in the early morning and during the late night
#   3. They probably spend time commuting between work and home everyday


# More filters to the user1. Add bitwise logic so that you're
# only examining records that came in on weekends (sat/sun).
#
user1 = user1[(user1.DOW == 'Sat')  |  (user1.DOW == 'Sun')]



#
# TODO: Further filter it down for calls that came in either before 6AM OR after 10pm (22:00:00).
# You can use < and > to compare the string times, just make sure you code them as military time
# strings, eg: "06:00:00", "22:00:00": https://en.wikipedia.org/wiki/24-hour_clock
#
# You might also want to review the Data Manipulation section for this. Once you have your filtered
# slice, print out its length:
#
user1 = user1[(user1.CallTime < "06:00:00") | (user1.CallTime > "22:00:00")]


# Visualize the dataframe with a scatter plot as a sanity check. 
# At this point, you don't yet know exactly where the user is located just based off the cell
# phone tower position data; but considering the below are for Calls that arrived in the twilight
# hours of weekends, it's likely that wherever they are bunched up is probably near where the
# caller's residence:
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(user1.TowerLon,user1.TowerLat, c='g', marker='o', alpha=0.2)
ax.set_title('Weekend Calls (<6am or >10p)')
#showandtell() 


#
# TODO: Run K-Means with a K=1. There really should only be a single area of concentration. If you
# notice multiple areas that are "hot" (multiple areas the usr spends a lot of time at that are FAR
# apart from one another), then increase K=2, with the goal being that one of the centroids will
# sweep up the annoying outliers; and the other will zero in on the user's approximate home location.
# Or rather the location of the cell tower closest to their home.....
#
# Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
# data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
# no feature scaling is required. Print out the centroid locations and add them onto your scatter
# plot. Use a distinguishable marker and color.
#
# Hint: Make sure you graph the CORRECT coordinates. This is part of your domain expertise.
#
user1 = user1[['TowerLon','TowerLat']]

kmeans = KMeans(n_clusters=2)
kmeans.fit(user1)
centroids = kmeans.cluster_centers_
print centroids
ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='r', alpha=0.5, linewidths=3, s=169)
#showandtell() 

# Repeat the above steps for all 10 individuals, being sure to record their approximate home
# locations. You might want to use a for-loop, unless you enjoy typing.'


def findloc(user):
    user = df[df.In == user]
    user = user[(user.DOW == 'Sat')  |  (user.DOW == 'Sun')]
    user = user[(user.CallTime < "06:00:00") | (user.CallTime > "22:00:00")]  
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(user.TowerLon,user.TowerLat, c='g', marker='o', alpha=0.2)
    ax.set_title('Weekend Calls (<6am or >10p)' + str(user))
    
    user = user[['TowerLon','TowerLat']]
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(user)    
    centroids = kmeans.cluster_centers_
    print centroids
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='r', alpha=0.9)
    showandtell() 

   

i =0
for user in distinct_In:
    findloc(user)
    i += 1
