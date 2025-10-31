🍽️ Zomato Data Analysis using Python
📖 Overview

This project explores and analyzes restaurant data from Zomato to uncover patterns and insights about restaurant types, customer preferences, and online ordering behavior.

It uses Python’s data visualization and analysis libraries to clean data, handle missing values, and visualize various trends such as restaurant categories, ratings, and cost distributions.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
⚙️ Features

🧹 Data Cleaning – Handles missing and inconsistent values such as ratings and cost fields.

🔢 Feature Transformation – Converts Zomato’s rating values into numeric format for analysis.

📊 Exploratory Data Analysis (EDA) – Visualizes restaurant types, votes, ratings, and cost for two people.

🔥 Insights Extraction – Identifies restaurants with the highest votes and explores online ordering trends.

🌡️ Heatmap Visualization – Displays correlation between restaurant types and online ordering availability.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧰 Tech Stack
Component	Technology
Language	Python
Libraries Used	Pandas, NumPy, Matplotlib, Seaborn
Dataset Format	CSV

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔍 Workflow
1. Load and Inspect Data

Import dataset using Pandas

Display summary statistics and column details

Configure Pandas to show all columns using:
  pd.set_option('display.max_columns', None)

2. Data Cleaning

Defined a function to clean the “rate” column:

def handleRate(value):
    value = str(value).split('/')[0]
    return float(value)

    Applied transformation using:

data_frame['rate'] = data_frame['rate'].apply(handleRate)


Checked missing values with:

data_frame.isnull().sum()

3. Restaurant Type Analysis

Visualized distribution of restaurants by type:

sns.countplot(x=data_frame['listed_in(type)'])
plt.show()


Grouped votes by restaurant type and plotted line chart:

grouped_data = data_frame.groupby('listed_in(type)')['votes'].sum()
plt.plot(grouped_data, c="green", marker='o')

4. Find Top-Rated Restaurant

Extracted restaurant(s) with the maximum votes:

max_votes = data_frame['votes'].max()
restaurant_max_votes = data_frame.loc[data_frame['votes'] == max_votes, 'name']

5. Visualizations

Online Order Count Plot

sns.countplot(x=data_frame['online_order'])
plt.show()


Rating Distribution

plt.hist(data_frame['rate'], bins=5)
plt.title('Rating Distribution')
plt.show()


Cost Distribution

sns.countplot(x=data_frame['approx_cost(for two people)'])
plt.show()


Boxplot of Ratings vs Online Orders

sns.boxplot(x='online_order', y='rate', data=data_frame)
plt.show()


Heatmap (Online Order vs Restaurant Type)

pivot_table = data_frame.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap')
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

▶️ How to Run

1. Install dependencies:

pip install pandas numpy matplotlib seaborn

2. Update the dataset path:
In the script, replace:

data_frame = pd.read_csv("E:\\data_analysis\\zomato_data_analysis\\content\\zomato_data.csv")

3. Run the program:

python main.py

4. Outputs:

Bar charts and histograms of restaurant trends

Line chart of total votes by restaurant type

Heatmap showing restaurant type vs online order preference

Console output with restaurant(s) having maximum votes

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📄 License

This project is open-source and available under the MIT License.
