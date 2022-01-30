


import os
import ujson
import gzip
import pandas as pd
import gc


# ### Load Review Data
data = []
with open('Electronics Dataset/Electronics.json') as f:
    for l in f:
        data.append(ujson.loads(l.strip()))

print(len(data))


df_reviews = pd.DataFrame.from_dict(data)

print(len(df_reviews))


del data
gc.collect()


df_reviews.shape

len(df_reviews.columns)

for i in df_reviews.columns:
    print(i)

df_reviews.head(110)


# ### Extracting Verified Reviews Only

verified_count = 0
for i in df_reviews['verified']:
    if i:
        verified_count += 1

print("% of verified reviews = {}%".format((verified_count/len(df_reviews))*100))

df_verified_reviews = df_reviews[df_reviews.verified == True]

len(df_verified_reviews)

# ### Load Product MetaData

metadata = pd.read_pickle('Electronics Dataset/Filtered_Electronics.pkl')

len(metadata)

metadata.head()


# ### Merge Review and Product Dataset

merged_data = pd.merge(df_reviews, metadata, on='asin',how='inner')

merged_data.to_picklele("merged_data.pkl")



for i in merged_data.columns:
    print(i)

merged_data.head()


merged_data.iloc[0]


# ### Filtering Verified Reviews

merged_verified = merged_data[merged_data.verified == True]

merged_verified.to_pickle("verified_merged_reviews.pkl")
