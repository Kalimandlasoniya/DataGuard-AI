# DataGuard AI

**AI-powered Data Quality, Anomaly Detection & Root-Cause Analysis System**

DataGuard AI is an intelligent data quality application that automatically profiles datasets, identifies data quality issues, detects unusual records using statistical and machine learning techniques, performs data cleaning, and generates AI-assisted root-cause analysis.

## 🚀 Project Overview

Poor data quality can affect analytics, reporting, and machine learning results.

DataGuard AI provides an end-to-end workflow:

```text
CSV / Excel File
       ↓
Data Profiling
       ↓
Data Quality Checks
       ↓
Data Cleaning
       ↓
Statistical Outlier Detection
       ↓
ML Anomaly Detection
       ↓
AI Root-Cause Analysis
       ↓
Streamlit Dashboard
       ↓
Power BI Dashboard
```

## ✨ Key Features

* Automatic dataset profiling
* Missing-value detection
* Duplicate-record detection
* Invalid-value detection
* City/value inconsistency detection
* Statistical outlier detection using IQR
* Machine learning anomaly detection using Isolation Forest
* Automated data cleaning
* Median-based missing-value treatment
* Duplicate removal
* AI-assisted root-cause analysis using Google Gemini
* Data quality scoring
* Power BI export
* Interactive Streamlit interface

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Google Gemini API
* Power BI
* OpenPyXL

## 🤖 Machine Learning

DataGuard AI uses **Isolation Forest** to identify unusual records across multiple numerical features.

The model analyzes combinations of features such as:

* Age
* Quantity
* Sales
* Discount

An anomaly flag indicates that a record has an unusual multivariate pattern and should be investigated further.

**Note:** An anomaly is not automatically an error or fraud. It is a signal for further investigation.

## 📊 Data Quality Analysis

The application checks for:

| Quality Check        | Description                             |
| -------------------- | --------------------------------------- |
| Missing Values       | Identifies missing cells                |
| Duplicates           | Detects duplicate records               |
| Invalid Values       | Identifies values outside defined rules |
| Inconsistencies      | Detects inconsistent categorical values |
| Statistical Outliers | Uses IQR-based detection                |
| ML Anomalies         | Uses Isolation Forest                   |

## 🧹 Data Cleaning

The cleaning pipeline can:

1. Standardize inconsistent city values
2. Convert invalid numeric values to missing values
3. Fill missing numeric values using the median
4. Remove duplicate records
5. Generate a cleaning log

The application also provides before-and-after cleaning statistics.

## 📈 Power BI Dashboard

The cleaned and analyzed data can be exported for Power BI.

The dashboard contains:

* Executive Data Quality Overview
* Data Quality Issues by Type
* ML Anomaly Analysis
* Statistical Outlier Analysis
* Cleaning Impact
* Data Quality Metrics

## 🧠 AI Root-Cause Analysis

Google Gemini is used to provide an AI-assisted interpretation of detected data quality issues.

The AI analysis helps identify possible causes and suggests areas for investigation.

The generated explanations are treated as **possible root causes**, not confirmed facts.

## 📁 Project Structure

```text
DataGuard-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── powerbi/
    └── DataGuard_AI_PowerBI_Export.zip
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/Kalimandlasoniya/DataGuard-AI.git
```

Move into the project folder:

```bash
cd DataGuard-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🔐 Gemini API Configuration

The application uses the Google Gemini API for AI-assisted root-cause analysis.

Set your API key as an environment variable:

```bash
GEMINI_API_KEY="your_api_key_here"
```

**Never commit your API key to GitHub.**

## 📌 Example Results

The project was tested using a synthetic sales dataset containing intentionally introduced data-quality issues.

Example analysis included:

* Missing values
* Duplicate records
* Invalid age and quantity values
* City-name inconsistencies
* Sales outliers
* Multivariate ML anomalies

The system combines rule-based validation, statistical analysis, machine learning, and generative AI into a single data-quality workflow.

## 🎯 Project Objective

The objective of DataGuard AI is to demonstrate how data analytics, machine learning, and generative AI can work together to improve data quality and support reliable downstream analysis.

## 👩‍💻 Author

**Kalimandla Soniya**

B.Tech — Artificial Intelligence & Data Science
