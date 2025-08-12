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
table.show()


infected_country_vs_total_cases = px.bar(dataset1.head(15),x='Country/Region', y='TotalCases', color='TotalCases',
       height=500, hover_data=['Country/Region', 'Continent'])
infected_country_vs_total_cases.show()

infected_country_vs_total_death = px.bar(dataset1.head(15),x='Country/Region', y='TotalCases', color='TotalDeaths',
                                         height=500, hover_data=['Country/Region', 'Continent'])
infected_country_vs_total_death.show()

infected_country_vs_total_recovered = px.bar(dataset1.head(15),x='Country/Region', y='TotalCases', color='TotalRecovered',
                                             height=500, hover_data=['Country/Region', 'Continent'])
infected_country_vs_total_recovered.show()

infected_country_vs_total_test = px.bar(dataset1.head(15),x='Country/Region', y='TotalCases', color='TotalTests',
                                        height=500, hover_data=['Country/Region', 'Continent'])
infected_country_vs_total_test.show()

total_test_wrt_continent = px.bar(dataset1.head(15),x='TotalTests', y='Continent', color='TotalTests',
                                  orientation='h', height=500, hover_data=['Country/Region','Continent'])
total_test_wrt_continent.show()


scatter_graph_totalcases = px.scatter(dataset1.head(57), x='Continent', y='TotalCases', hover_data=['Country/Region', 'Continent'],
           color='TotalCases', size='TotalCases', size_max=80, log_y=True) # if change y-axis from linear to logarithmic
scatter_graph_totalcases.show()

scatter_graph_total_tests = px.scatter(dataset1.head(54), x='Continent', y='TotalTests', hover_data=['Country/Region', 'Continent'],
           color='TotalTests', size='TotalTests', size_max=80,  log_y=True )
scatter_graph_total_tests.show()

bubble_chart_totalCases = px.scatter(dataset1.head(30), x='Country/Region', y='TotalCases', hover_data=['Country/Region','Continent'],
           color='Country/Region', size='TotalCases', size_max=80,log_y=True)
bubble_chart_totalCases.show()

bubble_chart_total_death = px.scatter(dataset1.head(10), x='Country/Region', y= 'TotalDeaths',
           hover_data=['Country/Region', 'Continent'],
           color='Country/Region', size= 'TotalDeaths', size_max=80)

bubble_chart_total_death.show()

bar_chart_death = px.bar(dataset2, x="Date", y="Deaths", color="Deaths",
       hover_data=["Confirmed", "Date", "Country/Region"],
       log_y=False, height=400)
bar_chart_death.show()

dataset2_bar_chart = px.bar(dataset2, x="Date", y="Confirmed", color="Confirmed",
       hover_data=["Confirmed", "Date", "Country/Region"], height=400)
dataset2_bar_chart.show()

df_india = dataset2.loc[dataset2["Country/Region"]=="India"]
india_confirmed = px.bar(df_india, x="Date", y="Confirmed", color="Confirmed", height=400)
india_confirmed.show()


recovered_line = px.line(df_india,x="Date", y="Recovered", height=400)
recovered_line.show()

new_cases_line = px.line(df_india,x="Date", y="New cases", height=400)
new_cases_line.show()

animation  =  px.choropleth(dataset2,
              locations="iso_alpha",
              color="Confirmed",
              hover_name="Country/Region",
              color_continuous_scale="Blues",
              animation_frame="Date")
animation.show()




