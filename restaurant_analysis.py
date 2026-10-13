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








