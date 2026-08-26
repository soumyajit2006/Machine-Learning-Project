# Machine-Learning-Project
# Account Receivables Payment Date Prediction System

A machine learning-based system for predicting invoice payment timelines from historical accounts receivable data. The project analyzes invoice, customer, payment-term, payment-mode, amount, and date-related information to estimate payment delay and support better cash-flow management and collection planning.

📌 Project Overview

Accounts receivable management is an important part of financial operations because delayed invoice payments can affect an organization's cash flow and working capital.

The objective of this project is to build a machine learning solution that can learn payment behavior from historical invoice records and predict the expected **payment delay in days**.

The predicted delay can then be combined with an invoice's due date to estimate the expected payment date.

### Problem Statement

Given historical invoice information such as:

* Customer identifier
* Invoice amount
* Due date
* Invoice creation date
* Baseline date
* Payment terms
* Payment mode
* Historical payment-related information

the system predicts the expected number of days by which an invoice is likely to be paid relative to its due date.

### Target Variable

The primary prediction target is:

delay = clearing_date - due_date

A negative delay indicates that the invoice was cleared before the due date, while a positive delay indicates payment after the due date.

For example:

Due Date       : 2021-01-29
Actual Payment : 2021-02-11
Delay          : 13 days

The predicted delay can subsequently be used to estimate the expected payment date.

🎯 Objectives

The major objectives of the project are:

* Analyze historical accounts receivable invoice data.
* Understand customer payment behavior and delay patterns.
* Perform exploratory data analysis on invoice and payment characteristics.
* Clean and preprocess date, numerical, and categorical variables.
* Engineer useful time-based and payment-related features.
* Encode categorical payment terms and payment modes.
* Select relevant predictive features.
* Train and compare regression-based machine learning models.
* Predict invoice payment delays.
* Convert predicted delays into expected payment dates.
* Evaluate predictions using both statistical and business-oriented metrics.
* Provide insights that can assist cash-flow forecasting and collection prioritization.

🗂️ Dataset

The project uses **AR Dummy Data** for experimentation and model development.

The dataset contains **8,052 invoice records and 10 original attributes**. The original dataset includes invoice/customer identifiers, invoice amount, multiple date fields, payment terms, and payment mode.

### Original Dataset Features

| Feature              | Description                               |
| -------------------- | ----------------------------------------- |
| `cms_doc_header_id`  | Invoice/document identifier               |
| `fk_customer_map_id` | Customer identifier                       |
| `amount`             | Invoice amount                            |
| `due_date`           | Invoice due date                          |
| `clearing_date`      | Actual invoice clearing/payment date      |
| `create_date`        | Invoice creation date                     |
| `baseline_date_norm` | Normalized baseline date                  |
| `mega_date`          | Date used for temporal analysis/splitting |
| `payment terms`      | Payment terms associated with the invoice |
| `payment mode`       | Payment method used for settlement        |

The dataset contains **136 unique customers**, **8 payment-term categories**, and **4 payment modes**.


## 🔄 Project Workflow

AR Dummy Data
      ↓
Data Loading
      ↓
Data Inspection & Validation
      ↓
Date Conversion
      ↓
Data Quality Checks
      ↓
Exploratory Data Analysis
      ↓
Target Creation
      ↓
Outlier Analysis
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Feature Selection
      ↓
Time-Based Train / Validation / Test Split
      ↓
Regression Model Training
      ↓
Payment Delay Prediction
      ↓
Expected Payment Date Estimation
      ↓
Model Evaluation

## 🧹 Data Preprocessing

### 1. Data Type Conversion

The date-related columns were initially stored as strings. They were converted into Pandas datetime objects to enable temporal analysis and feature engineering.

The following columns were converted:

due_date
clearing_date
create_date
baseline_date_norm
mega_date

This resulted in five datetime columns that could be used for date-based calculations.

### 2. Missing Value Analysis

Missing-value checks were performed across all original variables.

The dataset contained **no missing values** in the analyzed fields.

### 3. Data Validation

Several data-quality checks were performed, including:

* Checking date relationships.
* Checking for non-positive invoice amounts.
* Checking the relationship between baseline and clearing dates.
* Checking unique values and distributions.
* Checking the temporal range of the dataset.

The analyzed data covers invoice activity from **November 2019 through October 2021**.

📊 Exploratory Data Analysis

Extensive exploratory analysis was performed to understand payment behavior and identify useful patterns.

### Payment Delay

The payment delay was calculated as:

df['delay'] = (df['clearing_date'] - df['due_date']).dt.days

The project also investigated the distribution of payment delays using density plots and time-series visualizations.

### Delay Bucketing

Payment delays were grouped into business-friendly categories:

| Delay   | Category                   |
| ------- | -------------------------- |
| `<= 0`  | Paid on or before due date |
| `0–15`  | 1–15 days late             |
| `15–30` | 16–30 days late            |
| `>30`   | More than 30 days late     |

The dataset contained 4,056 invoices paid on/before their due dates, 3,730 invoices in the 0–15 day delay bucket, and 266 invoices in the 15–30 day bucket after the project's delay processing.

### Customer-Level Analysis

Customer-level analysis was performed based on:

* Total invoice amount
* Number of invoices
* Average invoice amount
* Payment delay patterns

The analysis also examined the top customers and their corresponding payment-delay distributions.

### Payment-Term Analysis

The dataset contains eight payment-term categories:

Z071
ZZ63
Z513
ZZ29
ZZ28
Z536
Z205
Z843

Z071 was the most frequent payment term in the dataset.

🧮 Feature Engineering

Several features were engineered from the original invoice and date information.

### Payment Time

Payment time was calculated as:

payment_time = due_date - baseline_date_norm

### Temporal Features

The project derived additional date-based features such as:

megaMonth
clearingWeekDay
closeDay
payment_day_of_week

These features capture monthly and day-level payment patterns.

### Categorical Encoding

Categorical variables were transformed using one-hot encoding.

Payment terms were encoded with features such as:

Code_Z071
Code_Z205
Code_Z513
Code_Z536
Code_Z843
Code_ZZ28
Code_ZZ29
Code_ZZ63

Payment modes were similarly encoded:

Mode_Cash
Mode_Cheque
Mode_Credit Card
Mode_Net banking

The encoded dataset contained the original variables plus these derived categorical features.

🚨 Outlier Analysis

Outlier analysis was performed using the **Interquartile Range (IQR)** method.

The calculated delay boundaries were:

Lower Boundary : -50 days
Upper Boundary : 48 days

No observations fell outside these boundaries for the delay variable, so the delay records were retained.

The invoice amount distribution was also examined for potential outliers.

🎯 Feature Selection

Recursive Feature Elimination (**RFE**) was used with a LightGBM regressor to identify relevant predictive variables.

The selected feature set included:

Code_ZZ63
Code_ZZ29
Code_ZZ28
Code_Z536
Code_Z513
Mode_Cash
Code_Z071
closeDay
clearingWeekDay
megaMonth
payment_time
amount
Mode_Cheque

The feature-selection process helped reduce the original feature space and focus the model on variables that contributed more strongly to payment-delay prediction.

⏳ Time-Based Data Splitting

Instead of randomly shuffling the invoice records, the project used a chronological split to preserve the temporal nature of payment prediction.

### Training Set

November 2019 – November 2020

### Validation Set

December 2020 – July 2021

### Test Set

July 31, 2021 – September 30, 2021

The resulting dataset sizes were:

| Dataset    | Records |
| ---------- | ------: |
| Training   |   5,129 |
| Validation |   2,518 |
| Test       |     401 |

This approach is more appropriate for a time-dependent prediction problem because future records are evaluated separately from historical records.

🤖 Machine Learning Models

The project explored multiple regression algorithms for predicting payment delay.

### Baseline Regression Models

The project was designed around regression-based payment prediction, including:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression

These models provide different levels of complexity and help establish baseline predictive performance.

### Additional Tree-Based Experiments

The notebook also contains experiments with more advanced gradient-boosting models, including:

* LightGBM
* XGBoost
* CatBoost

For example, LightGBM was trained using the selected feature set with 5,129 training observations and 13 selected features.

XGBoost was also explored with parameters including 200 estimators, maximum depth of 10, learning rate/eta of 0.1, subsampling, and column subsampling.

📏 Model Evaluation

The project evaluates predictions using both conventional regression metrics and payment-date tolerance metrics.

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted payment delays.

Lower MSE = Better

### Root Mean Squared Error (RMSE)

RMSE provides the prediction error in the same unit as the target variable — **days**.

RMSE = √MSE

This makes RMSE particularly intuitive for payment-date prediction.

### ± Day Accuracy

A custom scoring function was implemented to measure how frequently the predicted payment date falls within a specified number of business days of the actual payment date.

The project evaluates:

±0 days
±3 days
±5 days
±7 days
±9 days

The scoring logic derives actual and predicted payment dates from the invoice due date and predicted/actual delay, while excluding weekends from the business-day comparison.

📈 Example Model Performance

One of the LightGBM experiments produced the following test-set performance:

| Metric         | Test Result |
| -------------- | ----------: |
| MSE            |       82.52 |
| RMSE           |   9.08 days |
| Within ±3 days |      37.41% |
| Within ±5 days |      56.61% |
| Within ±7 days |      69.83% |
| Within ±9 days |      83.79% |

These values represent one of the model configurations explored in the notebook and should be interpreted as experimental results on the AR dummy dataset rather than production performance.

## A second LightGBM configuration using a lower learning rate, 500 estimators, 37 leaves, and an L1 regression objective achieved a test RMSE of approximately **9.23 days**.

📅 Payment Date Prediction

The model predicts the expected payment delay rather than directly predicting a calendar date.

The expected payment date can then be calculated as:

Predicted Payment Date
    = Invoice Due Date + Predicted Delay

For example:

Due Date              : 15-May-2021
Predicted Delay       : 7 days
Expected Payment Date : 22-May-2021

This approach makes the prediction easier to interpret for accounts receivable and collection teams.

💼 Business Applications

The system can potentially support organizations in:

### Cash Flow Forecasting

Estimate when outstanding invoices are likely to be converted into cash.

### Collection Prioritization

Identify invoices that are more likely to experience payment delays.

### Working Capital Management

Improve visibility into expected incoming cash flows.

### Customer Payment Analysis

Understand differences in payment behavior across customers and payment terms.

### Receivables Risk Monitoring

Use predicted payment delays to identify potentially late invoices earlier.

🛠️ Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* LightGBM
* XGBoost
* CatBoost

### Data Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook

📁 Project Structure

A recommended GitHub repository structure is:

Account-Receivables-Payment-Date-Prediction/
│
├── Project_SDP_Code.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   └── DummyARData.csv
│
├── images/
│   ├── delay_distribution.png
│   ├── correlation_heatmap.png
│   ├── customer_delay_analysis.png
│   └── model_performance.png
│
└── LICENSE

> **Note:** If you do not want to upload the AR dummy dataset, keep the data/ directory out of the repository and explain in the README that the notebook requires the corresponding AR dummy dataset.

🚀 How to Run the Project

1. Clone the Repository

git clone https://github.com/<your-username>/Account-Receivables-Payment-Date-Prediction.git

2. Navigate to the Project Directory

cd Account-Receivables-Payment-Date-Prediction

3. Create a Virtual Environment

python -m venv venv

4. Activate the Environment

# Windows

venv\Scripts\activate

# macOS/Linux

source venv/bin/activate

5. Install Dependencies

pip install -r requirements.txt

### 6. Launch Jupyter Notebook

jupyter notebook

Open:

Project_SDP_Code.ipynb

and execute the notebook cells sequentially.

📦 Requirements

numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
scipy
xgboost
lightgbm
catboost
tqdm

🔍 Key Insights from the Analysis

The exploratory analysis highlighted several useful characteristics of the AR dataset:

* The dataset contains 8,052 invoice records.
* There are 136 unique customer identifiers.
* Payment behavior varies across payment terms and customer segments.
* Invoice amounts show substantial variation between customers.
* Payment delays vary across the observed timeline.
* Payment terms and payment modes provide categorical information that can be incorporated into predictive models.
* Customer-level analysis can reveal differences in payment-delay patterns.
* Time-based splitting provides a more realistic evaluation setup for forecasting future payment behavior.

For example, the notebook found different average delay patterns across 2019, 2020, and 2021, demonstrating temporal variation in payment behavior.

⚠️ Limitations

This project is developed using **AR Dummy Data**, so the results should not be interpreted as production-level financial forecasting performance.

Important limitations include:

* The dataset is synthetic/dummy data.
* Real organizations may have additional variables affecting payment behavior.
* Customer payment behavior can change over time.
* Economic conditions, disputes, holidays, credit risk, and collection actions may influence actual payment dates.
* Model performance on dummy data may not generalize to real accounts receivable data.
* A production system would require continuous monitoring and periodic model retraining.

🔮 Future Improvements

Potential future enhancements include:

* Integrating real accounts receivable transaction data.
* Adding historical customer payment behavior features.
* Incorporating invoice dispute information.
* Adding customer credit-risk indicators.
* Including holidays and business calendars.
* Incorporating collection/contact history.
* Performing hyperparameter optimization.
* Comparing additional boosting and ensemble models.
* Building a prediction API using Flask or FastAPI.
* Creating an interactive dashboard using Streamlit or Power BI.
* Deploying the trained model using a cloud platform.
* Implementing model monitoring and periodic retraining.
* Providing confidence intervals or prediction ranges instead of only a point estimate.

📚 Learning Outcomes

Through this project, the following concepts were explored:

* Real-world-style accounts receivable analytics
* Data cleaning and validation
* Date/time feature engineering
* Exploratory Data Analysis
* Outlier detection using IQR
* Categorical variable encoding
* Feature selection using Recursive Feature Elimination
* Regression-based machine learning
* Tree-based ensemble models
* Time-based train/validation/test splitting
* Regression evaluation using MSE and RMSE
* Business-oriented prediction accuracy
* Payment-date prediction from predicted delay
* Customer segmentation and payment behavior analysis

👨‍💻 Author

**Soumyajit Kuila**

B.Tech Computer Science & Engineering — AI/ML
SOA University

Skills Demonstrated

Python
Pandas
NumPy
Scikit-learn
Machine Learning
Regression
Feature Engineering
EDA
Data Visualization
Jupyter Notebook

⭐ Project Highlights

* 📊 Analyzed **8,052 invoice records**
* 👥 Analyzed payment behavior across **136 customers**
* 📅 Used historical temporal information for payment-delay prediction
* 🧹 Performed data validation and outlier analysis
* ⚙️ Engineered date-based and payment-related features
* 🔤 Applied one-hot encoding to categorical variables
* 🎯 Used RFE for feature selection
* 🤖 Experimented with multiple regression and ensemble algorithms
* ⏳ Used chronological Train/Validation/Test splits
* 📏 Evaluated predictions using RMSE and business-day tolerance accuracy
* 💰 Designed the solution around cash-flow and collections use case

📄 License

This project is intended for educational and portfolio purposes.

If you use or modify this project, please provide appropriate attribution to the original author.

