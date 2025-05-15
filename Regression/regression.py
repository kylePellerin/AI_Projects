"""
Project 4 - CSCI 2400 - Regression

Name: Kyle Pellerin

python3 /Users/kpellerin/Downloads/project_4/regression.py

"""
import pandas as pd
import numpy as np  
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

housing = fetch_california_housing(as_frame=True)
df = housing.data
df['target'] = housing.target
#print(df.columns)



def get_most_important_feature(housing_df: pd.DataFrame) -> str:
    """
    This function takes an input of a dataframe and finds the most improntat feature, or the one
    that has the highest correlation with the target. To do this, we call the helper function
    betlow that gets all the r2 scores and we find the oen with the highest value and return 
    its key, in the case that there are no values we simply return "None". 
    """
    input = get_r2_scores(housing_df) #call get_r2 to get all r2 scores
    max_val = float("-inf")
    max_key = "None"
    for key, val_row in input.iterrows(): #iterate through each obtianed score looking for highest val
        val = val_row['r2_score']
        if val > max_val:
            max_val = val
            max_key = key
    return max_key

def get_r2_scores(housing_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in a pandas dataframe, seperates it by target and non target cateogires and then preforms
    a linear regression on each category comparing it to the target category. The comparison is based on R2 values 
    with higher ones indicating more correlation between a given cateogry and the target dataset. 
    """
    X = housing_df.drop("target", axis=1)
    y = housing_df["target"]

    scores = {} #output as dictionary to being with, map it later

    for column in X.columns:
        input = X[column] #reshape into one column matrix with the values
        to_array = np.array(input)
        X_column = to_array.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X_column, y, test_size=0.2, random_state=42) #seperate the data as we did in slides
        model = LinearRegression() #linear regresion modle 
        model.fit(X_train, y_train) #train it on the training data
        y_prediction = model.predict(X_test) 
        score = r2_score(y_test, y_prediction) #get R2 score
        scores[column] = score

    return pd.DataFrame.from_dict(scores, orient='index', columns=['r2_score'])




    return output



print("Most important feature: ", get_most_important_feature(df))
print(" ")
print(get_r2_scores(df))