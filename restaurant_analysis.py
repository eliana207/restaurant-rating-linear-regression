import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Aim: Build a linear regression model to understand which factors affect a restaurant's rating 


businesses = pd.read_json('yelp_business.json', lines = True)
checkins = pd.read_json('yelp_checkin.json', lines = True)
photos = pd.read_json('yelp_photo.json', lines = True)
review = pd.read_json('yelp_review.json', lines = True)
tips = pd.read_json('yelp_tip.json', lines = True)
users = pd.read_json('yelp_user.json', lines = True)

dataframes = [businesses, checkins, photos, review, tips, users]
#check which columns the datasets have in common: 

common_column = set(businesses.columns.to_list()) & set(checkins.columns.to_list()) & set(photos.columns.to_list()) & set(review.columns.to_list()) & set(tips.columns.to_list()) & set(users.columns.to_list()) #TODO: must be better way to handle this

common_column_name = next(iter(common_column))
print(common_column_name)

#joining the datasets into one big dataset
df = businesses
for i in range(1, len(dataframes) - 1): 
    df = pd.merge(df, dataframes[i+1], how='left', on=common_column_name)



columns = df.columns.to_list()
for i in range(len(columns)):
    print(i, columns[i])


#Cleaning dataset:


features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']

print(df.shape)
#dropping columns that are not binary or continuous variables
df.drop(features_to_remove, axis=1, inplace=True)
print(df.shape)

#replacing missing values with 0 
df.fillna(0)






