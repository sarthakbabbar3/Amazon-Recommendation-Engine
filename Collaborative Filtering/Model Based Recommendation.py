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


# In[2]:


data = pd.read_pickle("/Users/sidhantarora/work/ALDA/Project/data_nov_8.pkl")



newData = data[['asin', 'reviewerID', 'overall']].copy()


newData = newData.rename(columns={'asin': 'itemID', 'reviewerID': 'userID','overall': 'rating' })


reader = Reader(rating_scale=(1, 5))

# Loads Pandas dataframe
surprise_data = Dataset.load_from_df(newData[["userID", "itemID","rating"]], reader)


get_ipython().run_cell_magic('time', '', 'tuning_parameters = {\n    \'n_epochs\': [5, 10, 20 ], \'lr_all\': [0.001, 0.002, 0.005],\n    \'reg_all\': [0.2, 0.4, 0.6]\n}\nSVD_model = GridSearchCV(SVD, tuning_parameters, measures=["rmse","mae"], cv= 5)\n\nSVD_model.fit(surprise_data)')


evaluation = pd.DataFrame.from_dict(SVD_model.cv_results)




# In[11]:


print(SVD_model.best_score["rmse"])
print(SVD_model.best_params["rmse"])


# ### Testing on Common Data

# In[48]:


test_model = SVD(n_epochs =  20, lr_all =  0.005, reg_all = 0.2)


# In[52]:


kf = KFold(n_splits=5)

for trainset, testset in kf.split(surprise_data):

    # train and test algorithm.
    test_model .fit(trainset)
    predictions = test_model.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)


# ### Using Best Parameters

# In[12]:


best = SVD_model.best_estimator['rmse']
best.fit(surprise_data.build_full_trainset())


# In[13]:


best.predict("A28T6TZRAJF7J5","B01HIY64XM")


# ### Generating Recommendation

# In[14]:


#No. of all unique items
all_items = list(set(list(newData.itemID)))
len(all_items)


# In[15]:


def get_rating_predictions(user_id):
    
    item_rating = defaultdict(int)
    
    for item in all_items:
        item_rating[item] = best.predict(user_id, item).est
        
    return item_rating


# In[16]:


len(set(list(data['reviewerID'])))


# In[17]:


len(set(list(data['asin'])))


# In[18]:


len(data)


# In[19]:


def reviewed_items(user_id):
    
    items = set()
    
    for idx in range(len(data)):
        if data.iloc[idx]['reviewerID'] == user_id:
            items.add(data.iloc[idx]['asin'])
            
    return items


# In[20]:


reviewed_items("A28T6TZRAJF7J5")


# ###  Mapping Item id to Product Name

# In[21]:


itemID_to_name = defaultdict(str)

for idx in range(len(data)):
    itemID_to_name[data.iloc[idx]['asin']] = data.iloc[idx]['title']




# In[37]:


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


# ### Finding a user with more than 5 reviews

# In[43]:


for user in data.reviewerID:
    count = len(reviewed_items(user))
    if count >= 5:
        break
        
print(user)


# ####  Items purchased by user A3AKVALGT4Y02G

# In[44]:


bought_items = reviewed_items("A3AKVALGT4Y02G")
for i in bought_items:
    print(itemID_to_name[i]) 


# In[38]:


print(get_recommendation("A3AKVALGT4Y02G"))

