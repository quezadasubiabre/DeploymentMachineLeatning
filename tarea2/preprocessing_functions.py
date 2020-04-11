import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    return pd.read_csv(df_path)



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(target, axis=1),  # predictors
        df[target],  # target
        test_size=0.2,  # percentage of obs in test set
        random_state=0)  # seed to ensure reproducibility
    return X_train, X_test, y_train, y_test




def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0] # captures the first letter
    return df 



def add_missing_indicator(df, var):
    # function adds a binary missing value indicator
    # add missing indicator
    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)

    # replace NaN by median
    median_val = df[var].median()
    print(var, median_val)

    df[var].fillna(median_val, inplace=True)

    return df


    
def impute_na(df,var,word=None):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    if word==None:
        df[var] = df[var].fillna('Missing')
    else:
        df[var] = df[var].fillna(word)

    return df



def remove_rare_labels(df,var,frequents_ls):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    df[var] = np.where(df[var].isin(frequents_ls), df[var], 'Rare')
    return df



def encode_categorical(df, vars_cat):
    # adds the variables and removes original categorical variable
    
    df = df.copy()

    for var in vars_cat:
    
    # to create the binary variables, we use get_dummies from pandas
    
        df = pd.concat([df,pd.get_dummies(df[var], prefix=var, drop_first=True)], axis=1)  

    

    df.drop(labels=vars_cat, axis=1, inplace=True)

    return df 



    
    



    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
  
  
    

def scale_features(df, output_path):
    # load scaler and transform data
    scaler = joblib.load(output_path) # with joblib probably
    return scaler.transform(df)    



def train_model(df, target, output_path):
    # train and save model
        # initialise the model
    model = LogisticRegression(C=0.0005, random_state=0)
    
    # train the model
    model.fit(df, target)
    
    # save the model
    joblib.dump(model, output_path)
    
    return None



def predict(df, model_path):
    # load model and get predictions
    model = joblib.load(model_path)
    class_ = model.predict(df)
    pred = model.predict_proba(df)[:,1]
    return class_, pred
