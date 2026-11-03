import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#Aim: Build a linear regression model to understand which factors affect a restaurant's rating 


#LOADING THE DATA -- 

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

#JOINING DATASETS -- 

df = businesses
for i in range(len(dataframes) - 1): 
    df = pd.merge(df, dataframes[i+1], how='left', on=common_column_name)



#CLEANING DATA -- 

features_to_remove = ['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time']

#dropping columns that are not binary or continuous variables
df.drop(features_to_remove, axis=1, inplace=True)

#replacing missing values with 0 
df.fillna(0, inplace=True)


#CORELLATION ANALYSIS --

columns = df.columns.to_list()

cols_star_corr = {}
#checking each feature compared to rating (we will use correlation rounded to 3 df. )
for parameter in columns:
    cols_star_corr[parameter] = round(df[parameter].corr(df['stars']), 3)
#print("parameter:", parameter, "give us a correlation of:", df[parameter].corr(df['stars']))


#checking which parameters have highest correlation to ratings:
cols_star_corr_descending = {k: v for k, v in sorted(cols_star_corr.items(), key=lambda item: item[1], reverse=True)}


important_columns = []

i = 0
for column in cols_star_corr_descending:
    if i < 5 and column != 'stars':
        important_columns.append(column)
        i += 1 

        
#Plot stars vs important features
for column in important_columns:
    x = np.array(df[column])
    y = np.array(df['stars'])
    plt.scatter(x, y, alpha=0.3)
    plt.title(f"Stars vs {column}")
    #plt.show()
    #plt.close()


features = df[important_columns]
target = df['stars']

#SPLITTING DATA INTO TRAINING AND TESTING SETS -- 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=1)

print("head of y_test", y_test.head())

#TRAINING THE ML MODEL --
model = LinearRegression()
model.fit(X_train, y_train)

#EVALUATING THE MODEL -- 

#seeing score of model:
print("model score (value of coefficient of determination R^2:", model.score(X_test, y_test))

#plotting y_test against y_predicted:
plt.close('all')
y_predicted = model.predict(X_test)
plt.scatter(y_test, y_predicted, color='red', alpha=0.3)
plt.xlabel("y_test")
plt.ylabel("y_predicted")
plt.title(f"y_predicted against y_predicted")
plt.show()
plt.close()



