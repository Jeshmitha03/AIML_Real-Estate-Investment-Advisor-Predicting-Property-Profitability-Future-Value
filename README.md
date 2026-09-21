# 🏠 Real Estate Investment Advisor: Predicting Property Profitability & Future Value

## 📌 Project Overview

The **Real Estate Investment Advisor** is a Machine Learning project designed to analyze Indian real estate properties and provide data-driven insights for property investment decisions.

The project focuses on two main prediction tasks:

1. **Good Investment Classification** – predicts whether a property is considered a good investment.
2. **5-Year Future Price Prediction** – predicts the estimated future property price after five years.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, Machine Learning model development, model evaluation, and an interactive **Streamlit dashboard**.

---

## 🎯 Project Objectives

* Analyze Indian real estate property data.
* Clean and preprocess raw housing data.
* Perform exploratory data analysis to identify important patterns.
* Engineer useful features for Machine Learning.
* Classify properties based on investment potential.
* Predict estimated property value after five years.
* Compare multiple Machine Learning algorithms.
* Evaluate models using appropriate performance metrics.
* Build an interactive dashboard using Streamlit.
* Track and manage ML experiments using MLflow.

---

## 🧩 Problem Statement

Real estate investment decisions depend on several factors such as:

* Property price
* Location
* Property type
* Size
* Number of bedrooms
* Property age
* Infrastructure
* Nearby schools and hospitals
* Security
* Amenities
* Investment potential

Analyzing all these factors manually can be difficult.

This project uses Machine Learning and data analytics to process these factors and provide predictions that can support real estate analysis.

---

# 📊 Dataset

The project uses an Indian housing price dataset containing **250,000 property records**.

### Main Dataset

```text
data/raw/india_housing_prices.csv
```

### Processed Dataset

```text
data/processed/feature_engineered_housing_prices.csv
```

### Example Features

| Feature                        | Description                         |
| ------------------------------ | ----------------------------------- |
| State                          | State where the property is located |
| City                           | City of the property                |
| Locality                       | Local area                          |
| Property_Type                  | Type of property                    |
| BHK                            | Number of bedrooms                  |
| Size_in_SqFt                   | Property size                       |
| Price_in_Lakhs                 | Property price                      |
| Price_per_SqFt                 | Price per square foot               |
| Year_Built                     | Construction year                   |
| Age_of_Property                | Age of the property                 |
| Nearby_Schools                 | Nearby school information           |
| Nearby_Hospitals               | Nearby hospital information         |
| Public_Transport_Accessibility | Public transportation accessibility |
| Security                       | Security-related information        |
| Amenities                      | Available amenities                 |
| Facing                         | Property facing direction           |
| Owner_Type                     | Type of owner                       |
| Availability_Status            | Property availability               |
| Infrastructure_Score           | Infrastructure score                |
| Investment_Potential_Score     | Investment potential score          |

---

# 🎯 Target Variables

## 1. Good_Investment

`Good_Investment` is the classification target.

It represents whether a property satisfies the project's defined investment criteria.

```text
1 → Good Investment
0 → Not a Good Investment
```

### Important Note

The `Good_Investment` target was **created using predefined project rules based on the available property features**. Therefore, the classification model learns to reproduce those rules from the dataset; it should not be interpreted as an independent real-world investment recommendation.

---

## 2. Future_Price_5Y

`Future_Price_5Y` is the regression target.

It represents an estimated property price after five years.

### Important Note

The five-year future price is based on an **assumed growth scenario** used for this project. It is therefore a scenario-based estimate rather than a guaranteed real-world future market price.

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Data Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Model Saving
     ↓
Streamlit Dashboard
```

---

# 🧹 Data Preprocessing

The preprocessing stage includes:

* Handling missing values
* Checking duplicate records
* Handling incorrect data types
* Encoding categorical variables
* Scaling numerical features where required
* Detecting and handling data inconsistencies
* Preparing data for Machine Learning

---

# 📈 Exploratory Data Analysis

EDA was performed to understand the structure and relationships within the dataset.

### EDA includes:

* Univariate analysis
* Bivariate analysis
* Multivariate analysis
* Distribution analysis
* Correlation analysis
* Outlier analysis
* Property price analysis
* Location-based analysis
* Investment potential analysis

### Visualizations

The project uses:

* Histograms
* Box plots
* Bar charts
* Scatter plots
* Correlation heatmaps
* Count plots

---

# ⚙️ Feature Engineering

Feature engineering was performed to create useful Machine Learning features from the original dataset.

The feature engineering stage prepares the final dataset for classification and regression models.

The processed dataset is stored at:

```text
data/processed/feature_engineered_housing_prices.csv
```

---

# 🤖 Machine Learning Models

## Classification

The project evaluates multiple classification algorithms.

### Models Used

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Gradient Boosting

### Classification Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

## 📊 Classification Results

The classification models achieved very high performance on the test data.

Example results:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |     1.00 |      1.00 |   1.00 |     1.00 |    1.00 |
| Decision Tree       |     1.00 |      1.00 |   1.00 |     1.00 |    1.00 |
| Random Forest       |     1.00 |      1.00 |   1.00 |     1.00 |    1.00 |
| Gradient Boosting   |     1.00 |      1.00 |   1.00 |     1.00 |    1.00 |
| KNN                 |    0.935 |         — |      — |    0.936 |   0.983 |

### Interpretation

The perfect classification results should be interpreted carefully because the `Good_Investment` target was created using predefined rules from the available property features. This can make the target highly predictable and may result in unusually high model performance.

---

# 💰 Regression

The regression task predicts:

```text
Future_Price_5Y
```

### Regression Models

* Linear Regression
* Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

### Regression Metrics

Models were evaluated using:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² Score

---

## 📊 Regression Results

| Model             |      MAE |     RMSE |       R² |
| ----------------- | -------: | -------: | -------: |
| Linear Regression |   ~0.000 |   ~0.000 |    1.000 |
| Ridge Regression  | 0.001989 | 0.002495 |        — |
| Random Forest     | 0.002034 | 0.003250 |        — |
| Decision Tree     | 0.002409 | 0.006367 |        — |
| Gradient Boosting |    1.075 |    1.421 | 0.999953 |

### Interpretation

The extremely low errors and near-perfect R² should be interpreted in the context of how `Future_Price_5Y` was constructed. Because the target is based on an assumed growth scenario derived from property price information, the target has a strong mathematical relationship with the input data.

Therefore, these results demonstrate the model's ability to reproduce the project's defined scenario, rather than proving that it can accurately forecast actual real estate prices five years into the future.

---

# 💾 Saved Models

The trained models are stored in:

```text
dashboard/models/
```

### Files

```text
best_classification_model.pkl
best_regression_model.pkl
model_info.pkl
```

These saved models are loaded by the Streamlit dashboard to generate predictions.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

### Dashboard Sections

#### 1. Prediction

Users can enter property information and receive:

* Investment classification
* Investment probability
* Estimated future property price

#### 2. Market Insights

Provides visual insights into:

* Property prices
* Locations
* Property types
* Investment patterns

#### 3. Price Analysis

Provides charts for analyzing:

* Current property prices
* Price distributions
* Price-related relationships

#### 4. Model Performance

Displays:

* Classification metrics
* Regression metrics
* Confusion matrix
* ROC curve
* Model comparison

---

# 📁 Project Structure

```text
GUVI_my_third_project/
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv
│   │
│   └── processed/
│       └── feature_engineered_housing_prices.csv
│
├── notebook/
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── feature.ipynb
│
├── dashboard/
│   ├── app.py
│   │
│   └── models/
│       ├── best_classification_model.pkl
│       ├── best_regression_model.pkl
│       └── model_info.pkl
│
├── .gitignore
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Dashboard

* Streamlit

### Experiment Tracking

* MLflow

### Model Serialization

* Joblib / Pickle

### Development Environment

* Jupyter Notebook
* VS Code

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Jeshmitha03/AIML_Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value.git
```

Move into the project directory:

```bash
cd AIML-Real-Estate-Investment-Advisor-Predicting-Property-Profitability-Future-Value
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit mlflow joblib openpyxl
```

---

# ▶️ Running the Dashboard

From the project root directory, run:

```bash
streamlit run dashboard/app.py
```

The Streamlit application will open in your browser.

---

# 📓 Running the Notebooks

The notebooks are available in:

```text
notebook/
```

### EDA

```text
notebook/eda.ipynb
```

### Preprocessing

```text
notebook/preprocessing.ipynb
```

### Feature Engineering

```text
notebook/feature.ipynb
```

Run them sequentially to understand the complete data preparation workflow.

---

# 📊 MLflow

MLflow was used for experiment tracking and model management.

The dashboard can connect to the MLflow tracking server using:

```text
http://127.0.0.1:5000
```

MLflow can be started with:

```bash
mlflow ui --port 5000
```

---

# 🔍 Key Skills Demonstrated

This project demonstrates practical knowledge of:

* Python
* Pandas
* NumPy
* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Categorical Encoding
* Feature Scaling
* Classification
* Regression
* Hyperparameter Tuning
* Model Evaluation
* Confusion Matrix
* ROC-AUC
* Model Comparison
* Model Serialization
* Streamlit
* MLflow
* Data Visualization

---

# ⚠️ Project Limitations

* The `Good_Investment` target is generated using predefined project rules.
* `Future_Price_5Y` is based on an assumed growth scenario.
* The model predictions should not be treated as financial advice.
* Real-world property prices are influenced by many external factors such as economic conditions, interest rates, government policies, infrastructure development, supply and demand, and local market conditions.
* The dataset may not represent all Indian real estate markets equally.
* Very high model scores should be interpreted in the context of target construction and feature relationships.

---

# 🚀 Future Improvements

Possible improvements include:

* Use real historical property transaction data.
* Create investment labels using independently verified investment outcomes.
* Use time-based train/test splitting for future price forecasting.
* Incorporate historical price trends.
* Add location-based geospatial features.
* Include economic indicators such as interest rates and inflation.
* Add demand and supply indicators.
* Improve the dashboard with interactive maps.
* Deploy the application to a cloud platform.
* Perform more robust model validation on independent datasets.

---

# 👩‍💻 Author

**Jeshmitha J**

M.Tech Information Technology

Machine Learning | Artificial Intelligence | Data Science

GitHub: **Jeshmitha03**

---

# 📌 Disclaimer

This project is developed for **educational and Machine Learning demonstration purposes**.

The predictions generated by this application are based on the dataset, project-defined rules, and assumptions used during model development. They should not be considered professional financial, investment, valuation, or real estate advice.
