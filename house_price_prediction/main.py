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