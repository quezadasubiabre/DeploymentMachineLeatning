import preprocessing_functions as pf
import config

# ================================================
# TRAINING STEP - IMPORTANT TO PERPETUATE THE MODEL

# Load data


data = pf.load_data(config.PATH_TO_DATASET)


# divide data set

X_train, X_test, y_train, y_test = pf.divide_train_test(data, config.TARGET)


# get first letter from cabin variable

pf.extract_cabin_letter(X_train, 'cabin')

# impute NA categorical
for var in ['age', 'fare']:
    pf.add_missing_indicator(X_train, var)



# impute NA numerical

for var in config.CATEGORICAL_VARS:
    pf.impute_na(X_train,var)

# Group rare labels
for var in config.CATEGORICAL_VARS:
    pf.remove_rare_labels(X_train,var,config.FREQUENT_LABELS)

# encode variables
X_train = pf.encode_categorical(X_train, config.CATEGORICAL_VARS)


    

# scale variables


pf.train_scaler(X_train, config.OUTPUT_SCALER_PATH)
pf.scale_features(X_train, config.OUTPUT_SCALER_PATH)
pf.train_model(X_train, y_train, config.OUTPUT_MODEL_PATH)









# train scaler and save



# scale train set



# train model and save



print('Finished training')