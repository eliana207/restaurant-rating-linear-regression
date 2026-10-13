import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Aim: Build a linear regression model to understand which factors affect a restaurant's rating 


yelp_business = pd.read_json('yelp_business.json', lines = True)
yelp_checkin = pd.read_json('yelp_checkin.json', lines = True)
yelp_data = pd.read_json('yelp_data.json', lines = True)
yelp_photo = pd.read_json('yelp_photo.json', lines = True)
yelp_review = pd.read_json('yelp_review.json', lines = True)
yelp_tip = pd.read_json('yelp_tip.json', lines = True)
yelp_user = pd.read_json('yelp_user.json', lines = True)

file_prefixes = ['yelp_business', 'yelp_checkin', 'yelp_data', 'yelp_photo', 'yelp_review', 'yelp_tip', 'yelp_user']
pd_files = [yelp_business, yelp_checkin, yelp_data, yelp_photo, yelp_review, yelp_tip, yelp_user]

        










