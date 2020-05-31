import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from math import pow, sqrt
import re

df = pd.read_csv("/Users/Mana/Downloads/Impressions-train.csv", sep=',')
df.head()

#print(df.describe())

#f = open("/Users/Mana/Downloads/movie-codes.txt", "r")
#print(f.read())

ratings = pd.DataFrame(df.groupby('movie-code')['rating'].mean())
print(ratings.head())

ratings['number_of_ratings'] = df.groupby('movie-code')['rating'].count()
print(ratings.head())

#import matplotlib.pyplot as plt
#%matplotlib inline
ratings['rating'].hist(bins=50)
ratings['number_of_ratings'].hist(bins=60)

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)

movie_matrix = df.pivot_table(index='reviewerid', columns='movie-code', values='rating')
print(movie_matrix.head())

print(ratings.sort_values('number_of_ratings', ascending=False).head(10))

zootopia_user_rating = movie_matrix[200]
johnwick_user_rating = movie_matrix[77]

print(zootopia_user_rating.head())
print(johnwick_user_rating.head())

#similar_to_zootopia = movie_matrix.corrwith(zootopia_user_rating)
#print(similar_to_zootopia.head())

similar_to_johnwick = (movie_matrix.corrwith(johnwick_user_rating.dropna()))
print((similar_to_johnwick).head())

corr_johnwick = pd.DataFrame(similar_to_johnwick, columns=['Correlation'])
corr_johnwick.dropna(inplace=True)
corr_johnwick.head()

corr_johnwick = corr_johnwick.join(ratings['number_of_ratings'])
print(corr_johnwick.head())
corr_johnwick[corr_johnwick['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(20)
