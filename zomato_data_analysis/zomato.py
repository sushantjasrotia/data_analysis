import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_frame = pd.read_csv("E:\data_analysis\zomato_data_analysis\content\zomato_data.csv")
# Show all columns (no truncation)
pd.set_option('display.max_columns', None)


#
# print(data_frame.head())

def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)

data_frame['rate'] = data_frame['rate'].apply(handleRate)
print(data_frame.head())