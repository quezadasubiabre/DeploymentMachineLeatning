# ====   PATHS ===================

PATH_TO_DATASET = "titanic.csv"
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'logistic_regression.pkl'


# ======= PARAMETERS ===============

# imputation parameters



# encoding parameters
FREQUENT_LABELS = {
	'sex' : ['female', 'male'],
	'cabin': ['C', 'Missing'],
	'embarked': ['C', 'Q', 'S'],
	'title': ['Miss', 'Mr', 'Mrs']
}





# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_TO_IMPUTE = ['pclass', 'age', 'sibsp', 'parch', 'fare']