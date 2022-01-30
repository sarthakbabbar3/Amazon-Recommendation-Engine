
import os
import ujson
import gzip
import pandas as pd
import numpy as np
import gc
from collections import Counter


# ### Load Electronics Meta Data


data = []
with open('meta_Electronics.json') as f:
    for l in f:
        data.append(ujson.loads(l.strip()))

print(len(data))

df_reviews = pd.DataFrame.from_dict(data)

print(len(df_reviews))

df_reviews.shape


# ### Clearing data variable to free up space
del data
gc.collect()


# ### Columns for Review Data

### finding unique attribute values of main_cat(main category)
pd.unique(df_reviews['main_cat'])

main_cat_counter = Counter(df_reviews.main_cat)

main_cat_counter.most_common()

print("% of data selected = {}.".format(len(All_electronics)*100/len(df_reviews)))

selected_columns_values = ['Camera &amp; Photo', 'All Electronics',
       'Home Audio & Theater', 'Computers', 'Home Audio &amp; Theater',
       'Portable Audio &amp; Accessories', 'Portable Audio & Accessories',
       'Cell Phones &amp; Accessories', 'GPS &amp; Navigation', 'GPS & Navigation',
       'Camera & Photo', 'Cell Phones & Accessories','Car Electronics',
       'Amazon Devices','Video Games','Appliances', 'Apple Products',
       'Beats by Dr. Dre', 'Amazon Fire TV']
filtered_df = df_reviews[df_reviews.main_cat.isin(selected_columns_values)]

len(filtered_df)

All_electronics = filtered_df[filtered_df.main_cat == 'All Electronics']

All_electronics.head(10)

total = len(All_electronics)

for category in All_electronics.columns:
    count = 0

    print(category, len(All_electronics.category))

    for val in All_electronics[category]:
        if val == [] or i == '' or len(i)<=2:
            count += 1

    print(category,(count/total)*100)



# ### Finding Null values

All_electronics.to_json('All_Electronics_file1.json', orient = "records")


# ### EDA

# In[99]:


count = 0
for i in All_electronics.feature:
    if i == '':
        count +=1

print(count*100/len(All_electronics))

All_electronics.head(10)


All_electronics.replace('[]', np.NaN)


All_electronics.to_pickle("Filtered_Electronics.pkl")
