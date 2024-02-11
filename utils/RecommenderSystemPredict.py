#!/usr/bin/env python
# coding: utf-8

# # Movie Recommender System

# In[1]:


get_ipython().system('pip freeze')


# In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.
# 
# With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.
# 
# Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.

# ![43e0db2f-fea0-4308-bfb9-09f2a88f6ee4_what_is_netflix_1_en.png](attachment:43e0db2f-fea0-4308-bfb9-09f2a88f6ee4_what_is_netflix_1_en.png)

# With this context, EA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.

# What value is achieved through building a functional recommender system?
# Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.
# 
# 
# ![recommendation-system.webp](attachment:recommendation-system.webp)

# <a id="cont"></a>
# ## Table of Content
# 
# <a href=#one>1. Problem Statement</a>
#        
# <a href=#two>2. Importing Packages</a>
# 
# <a href=#three>3. Loading Data</a>
# 
# <a href=#four>4. Exploratory Data Analysis (EDA)<a>
#     
# <a href=#five>5. Data Processing</a>
# 
# <a href=#six>6. Feature Engineering</a>
# 
# <a href=#seven>7. Modelling</a>
# 
# <a href=#eight>8. Model Performance</a>
# 
# <a href=#nine>9. Saving & Exporting Model</a>
# 
# <a href=#ten>10. Conclusion</a>
# 
# <a href=#eleven>11. Recommendation</a>
# 
# <a href=#ref>Reference Document Links</a>

# 
# <a id="one"></a>
# # 1. Problem Statement
# <a href=#cont>Back to Table of Contents</a>

# In this challenge we have been tasked with proudcing an Unsupervised Learning algorithm that can provide personalized movie suggestions to users based on their preferences, ratings, and viewing history.The system should use a combination of collaborative filtering and content-based filtering techniques to generate recommendations, and evaluate the performance of the system using appropriate metrics.

# <a id="two"></a>
# # 2. Importing packages
# <a href=#cont>Back to Table of Contents</a>

# In[20]:


# Import our regular old heroes 
import numpy as np
import pandas as pd
import scipy as sp # <-- The sister of Numpy, used in our code for numerical efficientcy. 
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Entity featurization and similarity computation
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# Libraries used during sorting procedures.
import operator # <-- Convienient item retrieval during iteration 
import heapq # <-- Efficient sorting of large lists

# Imported for our sanity
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import sys
for p in sys.path:
       print(p)


# 
# <a id="three"></a>
# # 3. Loading the data
# <a href=#cont>Back to Table of Contents</a>

# In[4]:


gen_scores = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/genome_scores.csv')
gen_tgs = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/genome_tags.csv')
imd = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/imdb_data.csv')
links = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/links.csv')
mvs = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/movies.csv')
ss = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/sample_submission.csv')
tgs = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/tags.csv')
test = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/test.csv')
train = pd.read_csv('C:/Users/PSCadmin/Downloads/ea-movie-recommendation-predict-2023-2024/train.csv')


# ### Description of the Dataset

# Data Overview
# This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!
# 
# For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.
# 
# Source
# The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB
# 
# Supplied Files
# genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here
# genome_tags.csv - user assigned tags for genome-related scores
# imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.
# links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.
# sample_submission.csv - Sample of the submission format for the hackathon.
# tags.csv - User assigned for the movies within the dataset.
# test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.
# train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.

# 
# <a id="four"></a>
# # 4. Exploratory Data Analysis
# <a href=#cont>Back to Table of Contents</a>

# In[5]:


gen_scores.head()


# In[6]:


gen_tgs.head()


# In[7]:



imd.head()


# In[8]:



links.head()


# In[9]:



mvs.head()


# In[10]:



ss.head()


# In[11]:



tgs.head()


# In[12]:



test.head()


# In[13]:



train.head(10000)


# In[ ]:





# 
# <a id="five"></a>
# # 5. Date Pre-Processing
# <a href=#cont>Back to Table of Contents</a>

# In[14]:



file_names = ['gen_scores', 'genome_tags', 'imd', 'links', 'mvs', 'tgs', 'test', 'train']
files = [gen_scores, gen_tgs, imd, links, mvs, tgs, test, train]
for name, file in zip(file_names,files):
    print(name)
    print("Count of Null Values")
    print(file.isnull().sum())


# 
# <a id="six"></a>
# # 6. Feature Engineering
# <a href=#cont>Back to Table of Contents</a>

# In[15]:


#fill all empty cells of imdb_data with relevant data type
imd['title_cast'] = imd['title_cast'].fillna('')
imd['director'] = imd['director'].fillna('')
imd['runtime'] = imd['runtime'].fillna(0)
imd['budget'] = imd['budget'].fillna('')
imd['plot_keywords'] = imd['plot_keywords'].fillna('')


# In[16]:


#fill all empty cells of links with relevant data type
links['tmdbId'] = links['tmdbId'].fillna(0)


# In[17]:


#fill all empty cells of tags with relevant data type
tgs['tag'] = tgs['tag'].fillna('')


# In[18]:


#split all genres
mvs['genres'] = mvs['genres'].apply(lambda x: ' '.join(x.split('|')))


# In[24]:


#extract the year of all movies from title
mvs['year'] = mvs['title'].apply(lambda x: re.findall(r'\((.[\d]+)\)',x))
mvs['year'] = mvs['year'].str[-1]
mvs['year'] = mvs['year'].fillna(0)
mvs['year'] = mvs['year'].astype('int')
mvs.head()


# In[25]:


#split all titles
mvs['title_new'] = mvs['title'].apply(lambda x: x.split('('))
mvs['title_new'] = mvs['title_new'].apply(lambda x: x[0])
mvs.head()


# In[26]:


#split all title_cast in imdb_data
imd['title_cast'] = imd['title_cast'].apply(lambda x: ' '.join(x.split('|')))
#split all plot_keywords in imdb_data
imd['plot_keywords'] = imd['plot_keywords'].apply(lambda x: ' '.join(x.split('|')))


# In[27]:


imd.head()


# 
# <a id="seven"></a>
# # 7. Modelling
# <a href=#cont>Back to Table of Contents</a>

# In[31]:


# the function to convert from index to title_year
def get_title_year_from_index(index):

      return movies[movies.index == index]['title_year'].values[0]

# the function to convert from title to index
def get_index_from_title(title):

      return movies[movies.title == title].index.values[0]


# In[32]:


# create a function to find the closest title
def matching_score(a,b):

      return fuzz.ratio(a, b)


# In[33]:


# the function to convert from index to title
def get_title_from_index(index):

      return movies[movies.index == index]['title'].values[0]


# In[34]:


# the function to return the most similar title to the words a user types
def find_closest_title(title):

    leven_scores = list(enumerate(movies['title'].apply(matching_score, b=title)))
    sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
    closest_title = get_title_from_index(sorted_leven_scores[0][0])
    distance_score = sorted_leven_scores[0][1]

    return closest_title, distance_score


# In[35]:


def contents_based_recommender(movie_user_likes, how_many):
    # Get closest Title & Dist. score from Inputed Title
    closest_title, distance_score = find_closest_title(movie_user_likes)

    if distance_score == 100:
        # Get movie idex using declared fuunction
        movie_index = get_index_from_title(closest_title)
        # Apply index to similarity matrix and obtain list of similar movie index
        movie_list = list(enumerate(sim_matrix[int(movie_index)]))
        # Return a list of similar movies
        similar_movies = list(filter(lambda x:x[0] != int(movie_index), 
                                     sorted(movie_list, key=lambda x:x[1], reverse=True))) 
        print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
        
        for i, s in similar_movies[: how_many]: print(get_title_year_from_index(i))
    
    else:
        print('Did you mean '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n')
        movie_index = get_index_from_title(closest_title)
        movie_list = list(enumerate(sim_matrix[int(movie_index)]))
        similar_movies = list(filter(lambda x:x[0] != int(movie_index), 
                                     sorted(movie_list,key=lambda x:x[1], reverse=True)))
        print('Here\'s the list of movies similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')
        
        for i,s in similar_movies[:how_many]: print(get_title_year_from_index(i))


# In[ ]:





# In[40]:


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD, accuracy

# Load the MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')
# data

# Split the data into train and test sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
# trainset, testset

# Build and train a SVD model
model = SVD()
model.fit(trainset)

# Predict the ratings for the test set
predictions = model.test(testset)
# predictions

# Evaluate the model performance using RMSE
rmse = accuracy.rmse(predictions)
print(f'The RMSE of the model is {rmse:.2f}')


# In[ ]:





# In[ ]:





# In[ ]:




