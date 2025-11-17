# Electricity Consumption Analysis

This project performs a comprehensive analysis of household electricity usage using the **Individual Household Electric Power Consumption** dataset. It includes data cleaning, exploratory data analysis (EDA), forecasting evaluation, clustering, anomaly detection, and an AI-based usage categorization system.

---

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)  

**Columns Used:**

- `Date`
- `Time`
- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

---

## What This Notebook Does

### 1. Load and Prepare Data
- Reads the dataset
- Combines `Date` + `Time` into a single `Datetime` column
- Converts numeric columns to float
- Sorts by timestamp and sets `Datetime` as index
- Displays missing values

### 2. Exploratory Data Analysis (EDA)
- Generates:
  - Global Active Power time-series plot
  - Daily average power usage plot
- Identifies missing or abnormal values
- Helps understand trends and detect unusual behavior

### 3. Forecasting Evaluation (Error Metrics Only)
- Computes performance metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
- Plots:
  - Actual vs Predicted Global Active Power (first 500 samples)  

### 4. Clustering (Unsupervised Learning)
- Calculates daily consumption and groups using **K-Means (3 clusters)**:
  - Low-consumption days
  - Medium-consumption days
  - High-consumption days
- Scatter plot shows the daily consumption clusters

### 5. Anomaly Detection
- Applies **Isolation Forest** (contamination = 0.02)
- Plots anomalies vs normal days using daily consumption

### 6. AI Rule-Based Category System
- Classifies the latest Global Active Power into:
  - Low Usage → Use energy-efficient appliances
  - Medium Usage → Turn off unused devices
  - High Usage → Reduce AC/heater usage
- Outputs:
  - Latest consumption value
  - Assigned category
  - Suggestion

---

# How to Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Open the notebook

```bash
jupyter notebook
```

## Run the analysis notebook

```
notebooks/electricity_consumption.ipynb
```

## Place the dataset

```
data/household_power_consumption.txt
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

