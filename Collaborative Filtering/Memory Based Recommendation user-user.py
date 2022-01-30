#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ujson
import gzip
import pandas as pd
import gc
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise import SVD
from collections import defaultdict
import operator
from surprise.model_selection import KFold
from surprise import accuracy
from surprise import KNNWithMeans


# In[2]:


data = pd.read_pickle("/Users/sidhantarora/work/ALDA/Project/data_nov_8.pkl")




newData = data[['asin', 'reviewerID', 'overall']].copy()




# In[9]:


newData = newData.rename(columns={'asin': 'itemID', 'reviewerID': 'userID','overall': 'rating' })





reader = Reader(rating_scale=(1, 5))

# Loads Pandas dataframe
surprise_data = Dataset.load_from_df(newData[["userID", "itemID","rating"]], reader)


# ### Finding the best model for item-item filtering

# In[12]:


param = {
    "name": ["cosine"],
    "min_support": [3],
    "user_based": [True],
}
param_grid = {"sim_options": param}


# In[13]:


memory_based = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse"], cv=5)


# In[ ]:


memory_based.fit(surprise_data)


# In[ ]:


print(memory_based.best_score["rmse"])
print(memory_based.best_params["rmse"])


# In[ ]:


evaluation = pd.DataFrame.from_dict(memory_based.cv_results)


# ### Testing on Common Data

# In[ ]:


sim_options = {'name': 'msd', 'min_support': 12, 'user_based': False}


# In[ ]:


test_model = KNNWithMeans(sim_options=sim_options)


# In[ ]:


kf = KFold(n_splits=5)

for trainset, testset in kf.split(surprise_data):

    # train and test algorithm.
    test_model .fit(trainset)
    predictions = test_model.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# ### Using Best Parameters

# In[ ]:


best = memory_based.best_estimator['rmse']


# In[ ]:


best.fit(surprise_data.build_full_trainset())


# In[ ]:


best.predict("A28T6TZRAJF7J5","B01HIY64XM")


# ### Generating Recommendation

# In[ ]:


#No. of all unique items
all_items = list(set(list(newData.itemID)))
len(all_items)


# In[ ]:


def get_rating_predictions(user_id):
    
    item_rating = defaultdict(int)
    
    for item in all_items:
        item_rating[item] = best.predict(user_id, item).est
        
    return item_rating


# In[ ]:


len(set(list(data['reviewerID'])))


# In[ ]:


len(set(list(data['asin'])))


# In[ ]:


len(data)


# In[ ]:


def reviewed_items(user_id):
    
    items = set()
    
    for idx in range(len(data)):
        if data.iloc[idx]['reviewerID'] == user_id:
            items.add(data.iloc[idx]['asin'])
            
    return items


# In[ ]:


reviewed_items("A28T6TZRAJF7J5")


# ###  Mapping Item id to Product Name

# In[ ]:


itemID_to_name = defaultdict(str)

for idx in range(len(data)):
    itemID_to_name[data.iloc[idx]['asin']] = data.iloc[idx]['title']


# In[ ]:


def get_recommendation(user_id):
    item_rating = get_rating_predictions(user_id)
    already_bought = reviewed_items(user_id)
    sorted_items = sorted(item_rating.items(), key=operator.itemgetter(1), reverse = True)
    print(sorted_items[:10])
    
    items_to_suggest = []
    
    #Removing already bought items
    count = 0
    for item in sorted_items:
        
        if count == 10:
            break
        
        item_id = item[0]
        
        if item not in already_bought:
            items_to_suggest.append(itemID_to_name[item[0]])
            count += 1
    
    return items_to_suggest


# ####  Items purchased by user A3AKVALGT4Y02G

# In[ ]:


bought_items = reviewed_items("A3AKVALGT4Y02G")
for i in bought_items:
    print(itemID_to_name[i]) 


# In[ ]:


get_recommendation("A3AKVALGT4Y02G")

