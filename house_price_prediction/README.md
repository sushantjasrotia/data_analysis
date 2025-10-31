🏠 House Price Prediction using Machine Learning
📖 Overview

The House Price Prediction project is a machine learning model designed to estimate property prices based on various features such as area, location, number of rooms, and other property attributes.

This project demonstrates a complete end-to-end data science pipeline — from data preprocessing and feature encoding to model training, evaluation, and comparison.

------------------------------------------------------------------------------------------------------------------

⚙️ Features

🧹 Data Cleaning & Preprocessing – Handles missing values, drops irrelevant columns, and encodes categorical features.

📊 Exploratory Data Analysis (EDA) – Visualizes relationships and correlations using Seaborn and Matplotlib.

🔢 Feature Engineering – Converts categorical data into numerical form using One-Hot Encoding.

🤖 Model Training – Trains three regression models:

Support Vector Regression (SVR)

Random Forest Regressor

Linear Regression

📈 Model Evaluation – Compares models using Mean Absolute Percentage Error (MAPE).

🧠 Performance Insights – Understands how each algorithm performs on real-world property datasets.

-------------------------------------------------------------------------------------------------------------------

🧰 Tech Stack
Component	Technology
Programming Language	Python
Libraries Used	Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
ML Models	SVR, RandomForestRegressor, LinearRegression
Dataset Format	Excel (.xlsx)

-------------------------------------------------------------------------------------------------------------------
🔍 Workflow
1. Data Preprocessing

Load dataset using Pandas

Identify numeric and categorical columns

Handle missing values (fill mean for SalePrice)

Drop unnecessary columns (Id)

2. Exploratory Data Analysis

Correlation Heatmap for numerical variables

Bar charts showing categorical feature distributions

Count of unique categorical values

3. Feature Encoding

Apply OneHotEncoder to transform categorical columns into numeric form.

Merge encoded columns with the main dataset.

4. Model Training

Trains and evaluates three models:

Support Vector Regression (SVR)

Random Forest Regressor

Linear Regression

Each model is evaluated on the validation set using Mean Absolute Percentage Error (MAPE).

5. Model Evaluation

Lower MAPE = better performance.
Example:

SVR Model MAPE: 0.1870  → ~18.7% average prediction error
Random Forest MAPE: 0.0921  → ~9.2% error (better)
Linear Regression MAPE: 0.1056  → ~10.5% error

---------------------------------------------------------------------------------------------------------------------

▶️ How to Run

1. Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

2. Ensure the dataset file is in the same directory:

house_price_prediction.xlsx

3. Run the script:

python main.py

4. Outputs:

Correlation heatmap

Bar plots and distributions

Model performance comparison (printed in console)

---------------------------------------------------------------------------------------------------------------------

📄 License

This project is open-source and available under the MIT License.
