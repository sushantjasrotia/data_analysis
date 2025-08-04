import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_frame = pd.read_csv("E:\data_analysis\zomato_data_analysis\content\zomato_data.csv")
# None mean it Show all columns (no truncation)
#here it changes the global display option
pd.set_option('display.max_columns', None)



# print(data_frame.head())

def handleRate(value):
    value = str(value).split('/')
    value = value[0]
    return float(value)

data_frame['rate'] = data_frame['rate'].apply(handleRate)
print(data_frame.head())

data_frame.info()

# print(data_frame.isnull().sum())#print sum of values that are none

# ======================
# Restaurant Type Analysis
# ======================

sns.countplot(x=data_frame['listed_in(type)'])
plt.xlabel("Types of Restaurant") #effects the most recent created plot
plt.show()

grouped_data = data_frame.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes' : grouped_data})
plt.plot(result, c="green", marker='o')
plt.xlabel('Types of Restaurant')
plt.ylabel('Votes')
plt.show()

max_votes = data_frame['votes'].max()
restaurant_max_votes = data_frame.loc[data_frame['votes'] == max_votes, 'name']
print('Restaurant(s) with maximum votes:')
print(restaurant_max_votes)


sns.countplot(x = data_frame['online_order'])
plt.show()


plt.hist(data_frame['rate'],bins=5)
plt.title('Rating Distribution')
plt.show()

couple_data = data_frame['approx_cost(for two people)']
sns.countplot(x = couple_data)
plt.show()

plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order',
            y = 'rate',
            data = data_frame)

plt.show()

pivot_table = data_frame.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap')
plt.xlabel('Online Order')
plt.ylabel('Listed In (Type)')
plt.show()













