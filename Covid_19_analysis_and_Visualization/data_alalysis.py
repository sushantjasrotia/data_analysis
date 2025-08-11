#data analysis
import plotly.graph_objs  as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

#  data visualization

import matplotlib.pyplot as plt

#import plotly

import plotly.offline as py

pio.renderers.default = 'browser'
#initializing plotly
# pio.renderers.default = 'colab'

pd.set_option('display.max_columns', None)

dataset1 = pd.read_csv("covid.csv")
# print(covid_csv)

# print(covid_csv.shape)
# print(covid_csv.size)
# covid_csv.info()

dataset2 = pd.read_csv("covid_grouped.csv")
# print(covid_grouped_csv)
# print(dataset2.shape)
# print(dataset2.size)
# dataset2.info()

a = dataset1.columns
print(a)

dataset1.drop(['NewCases', 'NewDeaths', 'NewRecovered'], axis=1, inplace=True)
b=dataset1.sample(5)
print(b)

from plotly.figure_factory import create_table

colorscale = [[0, '#4d004c'],[.5,'#f2c5ff'],[1,'#ffffff']]
table = create_table(dataset1.head(15), colorscale=colorscale)
c = table.show()
print(c)