
import pandas as pd

data = pd.read_pickle('/Users/sarthakbabbar/Desktop/DataSet/relevant_reviewer_modified_also_buy.pkl')

len(data)

data.head(10)


# ### Handling Missing Data

data.isnull().sum()


index = []

for col in data.columns:
    count = 0
    for idx,value in enumerate(data[col]):
        if value in ['', [], 'Nan', 'NaN', 'nan'] or value is None:
            count += 1
            index.append(idx)
    print(col, count*100/3368235,"%")



len(set(index))

data['brand']


# ####  Dropping rows where brand is empty string

data = data[data.brand != '']

len(data)


# #### Drop Reviewer Names since we have ReviewerID. Also dropping Vote, Image Columns, also_view since majority of there rows are empty.

data.drop(['reviewerName'],axis = 1,inplace = True)

data.drop(['image','vote'],axis = 1,inplace = True)

data.drop(['imageURL','imageURLHighRes'],axis = 1,inplace = True)

data.drop(['also_view'],axis = 1,inplace = True)

data.columns


# #### Dropping verified since all reviews are verified

data.drop(['verified'],axis = 1,inplace = True)

data.drop(['similar_item'],axis = 1,inplace = True)

data.drop(['rank'],axis = 1,inplace = True)


# #### Dropping Rows with missing values in price, description, review text, sales rank. Decided not to drop price column since price is an important factor.

def drop_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


data = drop_rows_by_values(data,'price',['', [], 'Nan', 'NaN', 'nan',None])

len(data)


data = drop_rows_by_values(data,'reviewText',['', [], 'Nan', 'NaN', 'nan',None])

len(data)

data = drop_rows_by_values(data,'description',['', [], 'Nan', 'NaN', 'nan',None])

len(data)

data.dropna(subset=['reviewText'],inplace = True)

data = drop_rows_by_values(data,'also_buy',['', [], 'Nan', 'NaN', 'nan',None])

# data = drop_rows_by_values(data,'similar_item',['', [], 'Nan', 'NaN', 'nan',None])

data = drop_rows_by_values(data,'feature',['', [], 'Nan', 'NaN', 'nan',None])

data = drop_rows_by_values(data,'date',['', [], 'Nan', 'NaN', 'nan',None])

data.isnull().sum()

len(data)

index = []
data_len = len(data)

for col in data.columns:
    count = 0
    for idx,value in enumerate(data[col]):
        if value in ['', [], 'Nan', 'NaN', 'nan'] or value is None:
            count += 1
            index.append(idx)
    print(col, count*100/data_len,"%")


# #### Merge Review and Summary Columns

data.summary.iloc[0]

data.reviewText.iloc[0]

data['review_summary_combined'] = data['summary'] + ' ' + data['reviewText']


data.review_summary_combined.iloc[0]


# #### Drop review and summary

data.drop(['reviewText'],axis = 1,inplace = True)

data.drop(['summary'],axis = 1,inplace = True)

data.head(2)


# #### Categorize ratings as positive, negative, and neutral

categorize = []

for rating in data.overall:

    if rating < 3:
        categorize.append('negative')


    if rating == 3:
        categorize.append('neutral')


    if rating > 3:
        categorize.append('positive')


data['rating_category'] = categorize

data.head(2)


# #### Transform Review Time Column to datetime format


data.reviewTime.iloc[0]

data['time'] = data.reviewTime.str.replace(',', "")


data.time.iloc[0]


data['time'] = pd.to_datetime(data['time'], format = '%m %d %Y')


data.time.iloc[0]


data.drop(['reviewTime'],axis = 1,inplace = True)

data.head(5)
len(data)


data.to_pickle('data_nov_8.pkl')
