#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from html import unescape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[2]:


data = pd.read_pickle('final_data.pkl')


# In[3]:


print(data.head(5))


# ### EDA

# In[4]:


##################To get Unique descriptions################
df2 = data.explode("description").drop_duplicates(subset=["description"])
df2.reset_index(drop=True, inplace=True)


# In[5]:


total_reviews = len(data)
print("Number of reviews:",total_reviews)
print()
###Number of unique reviewers
no_unique_reviewers=len(data.reviewerID.unique())
print("Number of unique reviewers:",no_unique_reviewers)
print()
print("Proportion of unique reviewers:",float(no_unique_reviewers/total_reviews)*100)
print()
print ("Average rating score: ",round(data.overall.mean(),3))
#Number of positive reviews
pos=data[data['rating_category']=='positive']
neg=data[data['rating_category']=='negative']
neutral=data[data['rating_category']=='neutral']
print("Number of positive reviews:",len(pos))
#Number of negative reviews
print("Number of negative reviews:",len(neg))
#Number of neutral reviews
print("Number of neutral reviews:",len(neutral))


##########################################
## DISTRIBUTION OF RATING SCORE
########################################## 
#data=data.rename(columns={'overall':'rating'})
class_counts = data.groupby('overall').size()
print(class_counts)


# In[6]:


##########################################
## PLOT NUMBER OF REVIEWS FOR TOP 20 BRANDS  
##########################################

brands = data["brand"].value_counts()
plt.figure(figsize=(10,6))
brands[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 Brands")
plt.xlabel('Brand Name')
plt.ylabel('Number of Reviews')


# In[7]:


##########################################
## PLOT NUMBER OF REVIEWS FOR TOP 20 PRODUCTS  
##########################################

products = data["title"].value_counts()
plt.figure(figsize=(10,6))
products[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 50 Products")
plt.xlabel('Product Name')
plt.ylabel('Number of Reviews')


# In[ ]:


year=data['time'].dt.year

##################################################################
# Total review for every year
#####################################################################
plt.figure(figsize = (10,6))
sns.countplot(year)
plt.title('Total Review Numbers for Each Year', color='r')
plt.xlabel('year')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
data[year].value_counts()


# In[9]:


data['year']=year
print(data['year'])
# How many unique customers in each year?
unique_cust = data.groupby('year')['reviewerID'].nunique()

# Plot unique customer numbers in each year
plt.figure(figsize = (10,6))
unique_cust.plot(kind='bar', rot = 0, color = 'maroon')
plt.title('Unique Customers in Each Year', color='g', size = 14)
plt.xlabel('Year')
plt.ylabel('Unique Customer Numbers')
plt.show()

# Print unique customer numbers in each year
print(unique_cust)


# In[10]:


# How many unique products in each year?
unique_prod = data.groupby('year')['asin'].nunique()

# Plot unique product numbers in each year
plt.figure(figsize = (10,6))
unique_prod.plot(kind='bar', color = 'darkslategrey', rot =0)
plt.title('Unique Products in Each Year', color = 'g', size = 14)
plt.xlabel('Year')
plt.ylabel('Unique Product Numbers')
plt.show()

# Print unique product numbers in each year
print(unique_prod)


# In[11]:


# Total numbers of ratings in the home and kitchen product reviews
plt.figure(figsize = (10,6))
sns.countplot(data['overall'])
plt.title('Total Review Numbers for Each Rating', color='r')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()

# Customer totals for each rating class
data['overall'].value_counts()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10,6))

data.groupby('overall').overall.count()
data.groupby('overall').overall.count().plot(kind='pie',autopct='%1.1f%%',startangle=90,explode=(0,0.1,0,0,0),)


# ## Calcuating co-relation between products

# In[16]:


relevant_reviewer_df=pd.read_pickle('relevant_reviewer_modified_also_buy.pkl')


# In[18]:


product_group = relevant_reviewer_df.groupby('asin').size()
product_group[product_group>100].sort_values(ascending=False)


# In[19]:


df3 = relevant_reviewer_df.groupby('asin').agg({'overall': ['mean']})
df3[df3[('overall', 'mean')]>2].sort_values(by=[('overall', 'mean')], ascending=True)


# In[20]:


relevant_reviewer_df[relevant_reviewer_df['asin'].isin(['B00WDARWRY', 'B004M8SSZK', 'B000W9PD6A', 'B0019HL8Q8', 'B000BQ7GW8', 'B0015DYMVO', 'B0043T7FXE', 'B000VS4HDM'])].groupby('asin').agg({'overall': ['mean']})


# In[21]:


review_per_asin_df = relevant_reviewer_df[relevant_reviewer_df['asin'].isin(['B00WDARWRY', 'B000BQ7GW8', 'B0015DYMVO', 'B0043T7FXE', 'B000VS4HDM'])].groupby(['asin', 'overall']).size().reset_index(name='counts')


# In[22]:


also_bought_df = pd.DataFrame()
also_viewed_df = pd.DataFrame()
for product in relevant_reviewer_df.head(100000).asin.unique():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    also_bought = []
    also_viewed = []
    for value in relevant_reviewer_df[relevant_reviewer_df['asin'] == product].also_buy:
        also_bought += value
    for value in relevant_reviewer_df[relevant_reviewer_df['asin'] == product].also_view:
        also_viewed += value
    df1['also_buy'] = also_bought
    df1['asin'] = product
    also_bought_df = pd.concat([also_bought_df, df1], axis=0)
    df2['also_buy'] = also_bought
    df2['asin'] = product
    also_viewed_df = pd.concat([also_viewed_df, df2], axis=0)


# In[23]:


bought_grouped_df = also_bought_df.groupby(['also_buy', 'asin']).size().to_frame().reset_index()


# In[24]:


bought_grouped_df.rename(columns={0:'freq'},inplace=True)


# In[36]:


divisor_df = also_bought_df.groupby('asin').size().to_frame().reset_index()


# In[37]:


divisor_df.rename(columns={0:'divisor'},inplace=True)


# In[38]:


merged_bought_df=pd.merge(bought_grouped_df, divisor_df,on="asin",how="inner")


# In[39]:


merged_bought_df['prob_also_buy_after_asin'] = merged_bought_df['freq']/merged_bought_df['divisor']


# In[40]:


merged_bought_df[merged_bought_df['asin'].isin(['9831691113', '9831591534',])]


# In[41]:


merged_bought_size = merged_bought_df.groupby('also_buy').size()
merged_bought_size[merged_bought_size>5]


# In[42]:


merged_bought_df['prob_also_buy_after_asin'] = merged_bought_df['freq']/merged_bought_df['divisor']


# In[44]:


df = merged_bought_df[merged_bought_df['also_buy'].isin(['9831691113', '9831591534', 'B00005V52C', 'B0000BVYT3', 'B0000VYJRY', 'B00066HL50', 'B00066HOWK'])]
pivot_df = df.pivot(columns='also_buy', values='prob_also_buy_after_asin')
pivot_df.columns
pivot_df.replace(np.nan, 0, inplace=True)
array = np.where(pivot_df>0.0, 1, 0)
pivot_df = pd.DataFrame(data=array)
pivot_df
corr_df = pivot_df.corr()
corr_df


# In[45]:


colors = ['orange', 'g', 'b', 'c', 'y', 'm', 'r']
# legend = ['9831691113', '9831591534', 'B00005V52C', 'B0000BVYT3', 'B0000VYJRY', 'B00066HL50', 'B00066HOWK']
legend = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']
for i, c, l in zip(corr_df.index, colors, legend):
    plt.scatter(legend, corr_df.loc[i,:], color=c, label=l)
plt.legend()
plt.xlabel('Product')
plt.ylabel('Correlation')
plt.title('Correlation of Top 7 Bought Products')


# ## Word Cloud

# In[49]:


df_pos=relevant_reviewer_df[(relevant_reviewer_df['overall']==5) & (relevant_reviewer_df['asin']=='B000BQ7GW8')].sample(10000, replace=True)


# In[50]:


df_neg=relevant_reviewer_df[(relevant_reviewer_df['overall']==1) & (relevant_reviewer_df['asin'].isin(['B00005141S']))].sample(10000, replace=True)


# In[54]:


df_pos['reviewText']=df_pos['reviewText'].astype('str')


# In[55]:


stopwords = set(STOPWORDS)

text = " ".join(review for review in df_pos.reviewText)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("positivereview.png")


plt.show()


# In[56]:


df_neg['reviewText']=df_neg['reviewText'].astype('str')


# In[57]:


stopwords = set(STOPWORDS)

text = " ".join(review for review in df_neg.reviewText)
print ("There are {} words in the combination of all review.".format(len(text)))

# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig("positivereview.png")


plt.show()


# In[ ]:




