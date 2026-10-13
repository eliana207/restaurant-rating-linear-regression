import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Aim: Build a linear regression model to understand which factors affect a restaurant's rating 


businesses = pd.read_json('yelp_business.json', lines = True)
checkins = pd.read_json('yelp_checkin.json', lines = True)
data = pd.read_json('yelp_data.json', lines = True)
photos = pd.read_json('yelp_photo.json', lines = True)
review = pd.read_json('yelp_review.json', lines = True)
tips = pd.read_json('yelp_tip.json', lines = True)
users = pd.read_json('yelp_user.json', lines = True)

dataframes = [businesses, checkins, data, photos, review, tips, users]
#check which columns the datasets have in common: 

common_column = set(businesses.columns.to_list()) & set(checkins.columns.to_list()) & set(data.columns.to_list()) & set(photos.columns.to_list()) & set(review.columns.to_list()) & set(tips.columns.to_list()) & set(users.columns.to_list())

common_column_name = next(iter(common_column))

print("len common: ", len(common_column))

#joining the datasets into one big dataset
df = businesses
for i in range(len(dataframes) - 1): 
    df = pd.merge(df, dataframes[i+1], how='outer', on=common_column_name)

columns = df.columns.to_list()
for i in range(len(columns)):
    print(i, columns[i])
