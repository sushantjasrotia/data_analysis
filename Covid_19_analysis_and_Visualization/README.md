🦠 COVID-19 Data Analysis and Visualization
📖 Overview

The COVID-19 Data Analysis project is a Python-based data visualization and analysis tool that explores the global impact of COVID-19 using real-world datasets.
It demonstrates data preprocessing, interactive visualizations, and trend analysis using powerful Python libraries such as Plotly, Pandas, and Matplotlib.

Through this project, users can visualize infection rates, deaths, recoveries, and testing data across different countries and continents using interactive charts and maps.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ Features

📊 Interactive Charts – Dynamic bar, scatter, and bubble charts built with Plotly.

🌍 Animated World Map – Choropleth map showing COVID-19 spread over time.

📈 Trend Lines – Line graphs depicting confirmed, recovered, and new cases.

🇮🇳 Country-Specific Analysis – Detailed visualization for India’s COVID-19 trends.

🧮 Data Cleaning – Removal of unnecessary columns and sampling for display.

🎨 Custom Color Themes – Enhanced data readability with color-coded plots.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧰 Tech Stack
Component	Technology
Programming Language	Python
Libraries Used	Pandas, Plotly, Matplotlib
Visualization Mode	Interactive (browser rendering)
Data Format	CSV (covid.csv, covid_grouped.csv)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧩 How It Works

Data Loading:
Two CSV files (covid.csv and covid_grouped.csv) are read using Pandas.

Data Cleaning:
Unnecessary columns like NewCases, NewDeaths, and NewRecovered are removed.

Visualization:
Various Plotly Express visualizations are created to analyze the data:

Bar charts for top infected countries

Scatter and bubble charts for comparison

Line charts showing trend over time

Animated choropleth map for global spread visualization

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

▶️ How to Run

1. Install dependencies:
pip install pandas plotly matplotlib

2. Ensure dataset files are in the same directory:

covid.csv

covid_grouped.csv

3. Run the script:

python main.py

4. The visualizations will automatically open in your web browser (Plotly’s default renderer).

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This project is open-source and available under the MIT License
.
