#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# read the file

file_path='C:\\Users\\gouthami\\Downloads\\Cognify Dataset.csv'
df=pd.read_csv(file_path)
df


# In[3]:


# first 5 rows
df.head()


# In[4]:


# last 5 rows
df.tail()


# **Rows and columns**

# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


# seperate categorical and numerical column
cat=df.select_dtypes(include=['object']).columns
num=df.select_dtypes(exclude=['object']).columns
print('cat:',cat)
print('num:',num)


# In[8]:


# fill the missing values with mode because it is categorical column
df['Cuisines']=df['Cuisines'].fillna(df['Cuisines'].mode()[0])


# In[9]:


# again check any missing values are available or not
df.isnull().sum()


# **LEVEL :- 2**

# **Task :- 1**

# $Restaurant$ $Ratings$:-

# - Analyze the distribution of aggregate ratings and determine the most common rating range.

# In[12]:


# save the rating in one variable
rating=df['Aggregate rating'].head()
rating


# In[13]:


# Histogram of Aggregate rating among the restaurents
plt.figure(figsize=(5,5))
plt.hist(rating,bins=5)
plt.xlabel("Rating Range")
plt.ylabel("Number of restaurents")
plt.title('Distribution of aggregate rating among the restaurents')
plt.show()


# In[14]:


rating_ranges=pd.cut(df['Aggregate rating'],bins=[0,1,2,3,4,5], labels=['0-1','1-2','2-3','3-4','4-5'])


# In[15]:


rating_ranges.head()


# - Calculate the average number of votes received by restaurants.

# In[16]:


# Average votes received by the restaurent
avg_votes=df['Votes'].mean()
avg_votes


# **Task :- 2**

# $Cuisine$ $Combination$:-

# - identify the most common combinations of cuisines in the Dataset.

# In[17]:


# value counts of Cuisines
df['Cuisines'].value_counts()


# In[18]:


# Cuisines 
df['Cuisines']


# In[19]:


# Combinations of cuisines in the dataset
import itertools

df['Cuisines'] = df['Cuisines'].str.split(',')
unique_combinations = []
for i in df['Cuisines']:
    unique_combinations.extend(set(combo) for combo in itertools.combinations(i, 2))
combination_counts = pd.Series(unique_combinations).value_counts()
print(combination_counts.head())


# - Determine if certain cuisine combinations tend to have higher ratings.

# In[20]:


df=df.dropna(subset=['Cuisines','Aggregate rating'])


# In[21]:


df.head(2)


# In[22]:


# save the cuisines in a variable
rat=df['Cuisines']
rat


# In[23]:


import pandas as pd

# Assuming 'df' is your DataFrame
df['Cuisines'] = df['Cuisines'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Display the updated DataFrame
print(df['Cuisines'])


# In[24]:


avg_rating=df.groupby('Cuisines')['Aggregate rating'].mean()
avg_rating


# In[25]:


# Average rating in descending order
avg_rating=avg_rating.sort_values(ascending=False)
avg_rating


# In[26]:


# Combination of Cuisines
print('The Cuisines Combination that have higher ratings:')
print(avg_rating.head())


# **Task :- 3**

# $Geographic$ $Analysis$:-

# - plot the locations of restaurants on a map using longitude and latitude coordinates.

# In[27]:


# import the packages
import plotly.express as px


# In[28]:


# plot the restaurents on the map
fig = px.scatter_mapbox(
     df,
     lat='Latitude',
     lon='Longitude',
     hover_name='Restaurant Name',
     hover_data=['Cuisines'],
     color_discrete_sequence=['green'],
     zoom=5,
)


# In[29]:


fig.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 0, "l": 0, "b":0},
)


# - identify any patterns or clusters of restaurants in specific areas.

# In[30]:


# import the package
from sklearn.cluster import KMeans


# In[31]:


X=df[['Latitude','Longitude']]
num_cluster=5


# In[32]:


# k mean clustering
kmeans=KMeans(n_clusters=num_cluster,n_init=10,random_state=42)
df['cluster']=kmeans.fit_predict(X)


# In[33]:


# plot on the map
fig=px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    hover_name='Restaurant Name',
    hover_data=['Cuisines','Country Code'],
    color='cluster',
    color_continuous_scale='reds',
    zoom=5,
)


# In[34]:


fig.update_layout(
    mapbox_style='carto-positron',
    margin={'r':0,"t":0,"l":0,'b':0}
)
fig.show()


# In[35]:


# clustering

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = df[['Latitude', 'Longitude']]

num_clusters = 5

kmeans = KMeans(n_clusters=num_clusters,n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Plotting the clusters
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='rainbow')
plt.title('Restaurant Clusters')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


# **Task :- 4**

# $Restaurant$ $Chains$:-

# - identify if there are any restaurant chains present in the dataset.

# In[36]:


df.head(2)


# In[37]:


res_count=df['Restaurant Name'].value_counts()


# In[38]:


potential_chains=res_count[res_count > 1].index


# In[39]:


print("Potential restaurant chains:")
for chain in potential_chains:
    print(f"-{chain}")


# - Analyze the ratings and popularity of different restaurant chains.

# In[40]:


df=df[df['Aggregate rating'].notnull()]


# In[41]:


chain_stats=df.groupby('Restaurant Name').agg({
    'Aggregate rating':'mean',
    'Votes':'sum',
    'Cuisines':'count'
}).reset_index()


# In[42]:


chain_stats.columns=['Restaurant Name','Average rating','Total Votes','Number of Location']


# In[43]:


chain_stats=chain_stats.sort_values(by='Total Votes',ascending=False)


# In[44]:


print("Restaurant Chain Rating and Popularity Analysis (Sorted by Total Votes):")
print(chain_stats.to_string(index=False,justify='center'))


# In[ ]:




