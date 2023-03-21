#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df_tracks = pd.read_csv('tracks.csv')
df_tracks.head()


# In[6]:


## identify missing or null values

pd.isnull(df_tracks).sum()


# In[7]:


df_tracks.info()


# In[8]:


# identify the least top 10 popularity songs

sorted_df = df_tracks.sort_values('popularity', ascending=True).head(10)
sorted_df


# In[9]:


# describes the statistical properties

df_tracks.describe().transpose()


# In[10]:


# identify the top 5 popularity songs more than 90

most_popular = df_tracks.query('popularity>90', inplace = False).sort_values('popularity', ascending=False)
most_popular[:5]


# In[11]:


# using release_date as index

df_tracks.set_index('release_date', inplace=True)
df_tracks.index=pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[12]:


# using lambda function to divide 1000 for duration_ms

df_tracks['duration'] = df_tracks['duration_ms'].apply(lambda x: round(x/1000))
df_tracks.drop('duration_ms', inplace=True, axis=1)


# In[13]:


df_tracks.duration.head()


# In[14]:


# plotting correlation heatmap, using drop method to delete useless columns.
# Y axis means the height of correlation from -1 to 1, which means how one variable affect with another comparison variable.

corr_df = df_tracks.drop(['key', 'mode', 'explicit'],axis=1).corr(method='pearson')
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='inferno', linewidths=1, linecolor='Black')
heatmap.set_title('Correlation HeatMap Between Variable')


# In[15]:


# This can be useful for performing analyses or modeling on a smaller, representative subset of the data, 
# especially when the original DataFrame is very large.

# creates a new DataFrame called sample_df by randomly sampling 0.4% (0.004) 
# of the rows from an existing DataFrame called df_tracks.

sample_df = df_tracks.sample(int(0.004*len(df_tracks)))
print(len(sample_df))


# In[16]:


# linear regression model of Loudness and Energy from sample dataset.

plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y='loudness', x='energy', color='c', line_kws={"color": "green"}).set(title='Loudness VS. Energy Correlation')


# In[17]:


# linear regression model of Popularity and Acousticness from sample dataset.

plt.figure(figsize=(10,6))
sns.regplot(data = sample_df, y='popularity', x='acousticness', color='b', line_kws={"color": "green"}).set(title='Popularity VS. Acousticness Correlation')


# In[18]:


# get_level_values to retrieve the values of the 'release_date' level from the index of a DataFrame called df_tracks. 
# It then creates a new column called 'dates' in the df_tracks DataFrame and assigns the retrieved values to this column.

df_tracks['dates'] = df_tracks.index.get_level_values('release_date')
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year


# In[19]:


# Number of songs per year
# Distribution plot to represent data in histogram form

sns.displot(years, discrete=True, aspect=2, height=5, kind='hist').set(title="Number of songs per year")


# In[20]:


# Duration of the song during years in barplot

total_dr = df_tracks.duration
fig, ax = plt.subplots(figsize=(18, 7))
fig = sns.barplot(x=years, y=total_dr, ax=ax, errwidth=False).set(title="Year VS Duration")


# In[21]:


df_genre = pd.read_csv("SpotifyFeatures.csv")
df_genre.head()


# In[22]:


# Duration of the Songs in Different Genres

plt.title('Duration of the Songs in Different Genres')
sns.color_palette('rocket', as_cmap=True)
sns.barplot(y='genre', x='duration_ms', data=df_genre)
plt.xlabel('Duration in milli seconds')
plt.ylabel('Benres')


# In[23]:


# Top 5 Genres by Popularity

sns.set_style(style='darkgrid')
plt.figure(figsize=(10,5))
famous = df_genre.sort_values('popularity', ascending=False).head(10)
sns.barplot(y='genre', x='popularity', data=famous).set(title='Top 5 Genres by Popularity')


# In[ ]:




