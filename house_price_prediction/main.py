import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_excel("house_price_prediction.xlsx")
pd.set_option("display.max_rows", None)   # show all rows
pd.set_option("display.max_columns", None)
# print(dataset.head(5))
print(dataset.shape)

#--------------Data Preprocessing-----------------------------

object = (dataset.dtypes == 'object')
object_cols = list(object[object].index)
print("Categorical variables:" , len(object_cols))

Integer = (dataset.dtypes == 'int')
integer_cols = list(Integer[Integer].index)
print("Integer variables:", len(integer_cols))

float = (dataset.dtypes == 'float')
float_cols = list(float[float].index)
print("float variables:", len(float_cols))

#--------------------Exploratory Data Analysis---------------------
numerical_dataset = dataset.select_dtypes(include=["number"])
plt.figure(figsize=(12,7))
sns.heatmap(numerical_dataset.corr(),
            cmap= 'BrBG',
            fmt='.2f',
            linewidths=2,
            annot=True)
plt.show()

unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize = (10,7))
plt.title('No. Unique values of Categorical  Featured')
plt.xticks(rotation = 90)
sns.barplot(x=object_cols, y=unique_values)

plt.show()

plt.figure(figsize=(18, 12))   # wider instead of tall
plt.suptitle('Categorical Feature: Distribution', fontsize=16)

index = 1
for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(4, 4, index)
    sns.barplot(x=list(y.index), y=y.values)
    plt.title(col)
    plt.xticks(rotation=90)
    index += 1

plt.tight_layout(rect=[0, 0, 1, 0.96])  # fixes overlapping and fits title
plt.show()

#-------------------------Data Cleaning--------------------------------

dataset.drop(['Id'],
             axis=1,
             inplace=True)

dataset['SalePrice']= dataset['SalePrice'].fillna(
    dataset['SalePrice'].mean()
)

new_dataset = dataset.dropna()

a = new_dataset.isnull().sum()
print(a)

#--------------------------OneHotEncoder - For Label categorical features------------------

from sklearn.preprocessing import OneHotEncoder
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("categorical variables : ")
print(object_cols)
print("No. of  categorical features: ", len(object_cols))

#----------------------------------OneHotEncoding to the whole list.-------------------------------

OH_encoder = OneHotEncoder(sparse_output= False, handle_unknown='ignore')
# sparse_output=False → returns a dense DataFrame/array instead of a sparse matrix (easier to see in Pandas).
#
# handle_unknown='ignore' → if in the future you see a category that wasn’t in training
#it won’t crash. It will just ignore it.

OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
# fit → learns the unique categories in each categorical column.
#
# transform → creates one-hot encoded columns.

OH_cols.index = new_dataset.index
# Ensures the new one-hot DataFrame has the same row indexes as the original dataset.
#
# This is needed so you can concatenate them later without mismatch.

OH_cols.columns = OH_encoder.get_feature_names_out()
# Renames the new one-hot encoded columns with meaningful names

df_final = new_dataset.drop(object_cols, axis=1)
# Drops the original categorical columns because now you’ve converted them into numeric one-hot columns.

df_final = pd.concat([df_final, OH_cols], axis=1)
# Joins the numerical features (still present in new_dataset) with the one-hot encoded categorical features.
#
# Final DataFrame (df_final) is all numeric ✅.



#-----------------------------------Splitting Dataset into Training and Testing------------------------------

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# mean_absolute_error → A metric used to check how good your regression model is (smaller is better).
#
# train_test_split → A utility that splits your dataset into training data (for model learning)
# and validation/test data (for checking how well the model performs on unseen data).

X = df_final.drop(['SalePrice'], axis = 1)
Y = df_final['SalePrice']

# SalePrice is the target variable (the thing you want to predict, i.e., house price).
#
# X = all other columns → the features (things like LotArea, Neighborhood, HouseStyle, etc.).
#
# Y = SalePrice → the label/output we want to predict.
#
# So here:
#
# X = Input
#
# Y = Output

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size= 0.8, test_size= 0.2, random_state= 0)
#
#
# train_test_split splits the dataset into two parts:
#
# Training set (80%) → (X_train, Y_train) → used by the model to learn patterns.
#
# Validation set (20%) → (X_valid, Y_valid) → used to evaluate model performance on data it hasn’t seen before.
#
# train_size=0.8 → 80% data goes to training.
#
# test_size=0.2 → 20% data goes to validation.
#
# random_state=0 → fixes the random shuffling so every time you run the code, you get the same split (important for reproducibility).

#---------------------------------------------------------------------------------------------
                                #------------------------------------------
#------------------------------------ Model Training and Accuracy------------------------------


# SVM-Support Vector Machine
# Random Forest Regressor
# Linear Regressor
#


#--------------------------------------------SVM - Support vector Machine----------------------------

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

# svm → Support Vector Machine algorithms (can be used for classification or regression).
#
# SVC → Support Vector Classifier (for classification tasks, not used here).
#
# mean_absolute_percentage_error (MAPE) → A metric that tells you how far off your predictions are from the actual values, in percentage terms.

model_SVR = svm.SVR()
model_SVR.fit(X_train, Y_train)
#
# Train the model using the training data (80% we split earlier).
#
# The model looks at features (X_train) and tries to learn the relationship to predict SalePrice (Y_train).

Y_pred = model_SVR.predict(X_valid)
# Uses the trained model to predict house prices for the validation set (X_valid).
#
# Y_pred contains predicted prices.

print(mean_absolute_percentage_error(Y_valid, Y_pred
                                     ))

# This number is the average percentage error between your actual house prices (Y_valid) and the predicted prices (Y_pred) made by your Support Vector Regression (SVR) model.
#
# 0.18705129 = 18.7% (approx)
#
# It means:
#
# On average, your model’s predicted house prices are 18.7% off from the actual house prices.

#---------------------------------------Random Forest Regression----------------------------------------

from sklearn.ensemble import RandomForestRegressor
# Random Forest is an ensemble learning method: it builds many decision trees and averages their results to make predictions more accurate and stable.
model_RFR = RandomForestRegressor(n_estimators= 10)
# Creates a Random Forest Regressor model.
#
# n_estimators=10 → the number of decision trees to build in the forest.
#
# Each tree makes its own prediction.
#
# The Random Forest takes the average of all trees’ predictions.
#
# More trees (n_estimators=100 or 500) usually = better accuracy, but slower training.
model_RFR.fit(X_train, Y_train)

#Imagine you ask 10 real estate experts to estimate house prices. Each has different experience and biases. The average of their answers is usually closer to reality than relying on just one expert.
Y_pred = model_RFR.predict(X_valid)

#Uses the trained Random Forest model to make predictions on unseen validation data (X_valid).

# Y_pred = the predicted house prices for that 20% validation set.
print(mean_absolute_percentage_error(Y_valid, Y_pred))

#------------------------------------Linear Regression-------------------------------------------

from sklearn.linear_model import LinearRegression

model_LR = LinearRegression()
# This model assumes there is a linear relationship between your features (X) and the target (Y).
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
#Uses the trained linear equation to predict house prices for the validation set (X_valid).

# Y_pred = predicted house prices.
print(mean_absolute_percentage_error (Y_valid, Y_pred))











