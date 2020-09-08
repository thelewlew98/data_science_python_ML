import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


columns_names = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('u.data',sep='\t',names=columns_names)
print(df.head())
movie_titles = pd.read_csv('Movie_Id_Titles')
print(movie_titles.head())

df = pd.merge(df,movie_titles,on='item_id')
print(df.head())

a=df.groupby('title')['rating'].mean().sort_values(ascending=False).head()
print(a)
b=df.groupby('title')['rating'].count().sort_values(ascending=False).head()
print(b)
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
plt.figure(1)
ratings['num of ratings'].hist(bins=70)
plt.xlim(0,600)
plt.show()
plt.figure(2)
ratings['rating'].hist(bins=70)
plt.show()
plt.figure(3)
sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)
plt.show()

print(df.head())

moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings['num of ratings'])
a=corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation'
               ,ascending=False)

print(a.head())
print('\n')

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
b=corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation'
               ,ascending=False)
print(b.head())