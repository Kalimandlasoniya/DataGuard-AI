import io
import os
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


# =========================================================
# APP CONFIG
# =========================================================

APP_NAME = "DataGuard AI"
GEMINI_MODEL = "gemini-3.6-flash"

AGE_MIN = 18
AGE_MAX = 100

CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru",
}


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        text-align: center;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #eef8ee;
        border: 1px solid #b7ddb7;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #eef5ff;
        border: 1px solid #b8d2f2;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_to_datetime(series):
    """
    Safely convert a pandas Series to datetime.

    Handles mixed date formats and invalid values.
    """
    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )
    except Exception:
        try:
            return pd.to_datetime(
                series,
                errors="coerce"
            )
        except Exception:
            return pd.Series(pd.NaT, index=series.index)


def detect_datetime_columns(df):
    """
    Detect columns that appear to contain datetime values.
    """

    datetime_columns = []

    for column in df.columns:

        try:
            converted = safe_to_datetime(df[column])

            valid_count = converted.notna().sum()

            if len(df) > 0:
                valid_ratio = valid_count / len(df)

                if valid_ratio >= 0.70:
                    datetime_columns.append(column)

        except Exception:
            continue

    return datetime_columns


def classify_columns(df):
    """
    Classify columns into numeric, categorical and datetime.
    """

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    datetime_columns = detect_datetime_columns(df)

    return (
        numeric_columns,
        categorical_columns,
        datetime_columns
    )


# =========================================================
# DATA LOADING
# =========================================================

def load_uploaded_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):

            return pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):

            return pd.read_excel(uploaded_file)

        elif file_name.endswith(".xls"):

            return pd.read_excel(uploaded_file)

        else:

            st.error(
                "Unsupported file type. Please upload CSV or Excel."
            )

            return None

    except Exception as e:

        st.error(
            f"Unable to read the file: {e}"
        )

        return None


# =========================================================
# DATA QUALITY CHECKS
# =========================================================

def check_invalid_values(df):

    results = {}

    # Age validation
    if "Age" in df.columns:

        age = pd.to_numeric(
            df["Age"],
            errors="coerce"
        )

        invalid_age = (
            (age < AGE_MIN) |
            (age > AGE_MAX)
        ).sum()

        results["Invalid Age"] = int(invalid_age)

    # Quantity validation
    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        invalid_quantity = (
            quantity < 0
        ).sum()

        results["Invalid Quantity"] = int(
            invalid_quantity
        )

    return results


def check_city_inconsistencies(df):

    if "City" not in df.columns:

        return {
            "found": False,
            "count": 0,
            "values": []
        }

    city_series = (
        df["City"]
        .astype("string")
        .str.strip()
    )

    inconsistent_values = []

    for city in city_series.dropna().unique():

        if city in CITY_MAPPING:

            inconsistent_values.append(city)

    return {
        "found": len(inconsistent_values) > 0,
        "count": len(inconsistent_values),
        "values": inconsistent_values
    }


# =========================================================
# IQR OUTLIER DETECTION
# =========================================================

def detect_iqr_outliers(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    outlier_summary = {}
    outlier_masks = {}

    for column in numeric_columns:

        try:

            series = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if len(series) < 4:

                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            if iqr == 0:

                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            mask = (
                (df[column] < lower_bound) |
                (df[column] > upper_bound)
            )

            count = int(mask.sum())

            outlier_summary[column] = {
                "count": count,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }

            outlier_masks[column] = mask

        except Exception:
            continue

    return outlier_summary, outlier_masks


# =========================================================
# ML ANOMALY DETECTION
# =========================================================

def run_ml_anomaly_detection(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) == 0:

        return df.copy(), 0

    working_df = df.copy()

    features = working_df[numeric_columns].copy()

    # Convert all numeric columns safely
    for column in numeric_columns:

        features[column] = pd.to_numeric(
            features[column],
            errors="coerce"
        )

    # Fill missing values
    features = features.fillna(
        features.median()
    )

    # If median is still NaN
    features = features.fillna(0)

    # Need at least 2 rows
    if len(features) < 2:

        working_df["ML_Anomaly"] = 0

        return working_df, 0

    try:

        model = IsolationForest(
            contamination="auto",
            random_state=42
        )

        predictions = model.fit_predict(
            features
        )

        working_df["ML_Anomaly"] = np.where(
            predictions == -1,
            1,
            0
        )

        anomaly_count = int(
            (working_df["ML_Anomaly"] == 1).sum()
        )

        return working_df, anomaly_count

    except Exception:

        working_df["ML_Anomaly"] = 0

        return working_df, 0


# =========================================================
# ROOT CAUSE HYPOTHESES
# =========================================================

def generate_root_cause_hypotheses(
    df,
    quality_results,
    outlier_summary,
    anomaly_count
):

    hypotheses = []

    missing_total = int(
        df.isna().sum().sum()
    )

    duplicate_count = int(
        df.duplicated().sum()
    )

    if missing_total > 0:

        hypotheses.append(
            "Missing values may indicate incomplete data entry, "
            "optional fields, or failed data collection."
        )

    if duplicate_count > 0:

        hypotheses.append(
            "Duplicate records may have been introduced through "
            "repeated uploads, system retries, or data merging."
        )

    if "Invalid Age" in quality_results:

        if quality_results["Invalid Age"] > 0:

            hypotheses.append(
                "Invalid age values may be caused by data-entry "
                "errors or incorrect source-system values."
            )

    if "Invalid Quantity" in quality_results:

        if quality_results["Invalid Quantity"] > 0:

            hypotheses.append(
                "Negative or invalid quantity values may indicate "
                "transaction-entry or data-processing issues."
            )

    if outlier_summary:

        large_outliers = sum(
            item["count"]
            for item in outlier_summary.values()
        )

        if large_outliers > 0:

            hypotheses.append(
                "Extreme numeric values may represent genuine "
                "business events or potential data-quality problems."
            )

    if anomaly_count > 0:

        hypotheses.append(
            "Machine-learning anomalies may indicate unusual "
            "combinations of numeric attributes that require review."
        )

    city_info = check_city_inconsistencies(df)

    if city_info["found"]:

        hypotheses.append(
            "Different spellings or abbreviations of cities may "
            "come from inconsistent data-entry standards."
        )

    if not hypotheses:

        hypotheses.append(
            "No major automated root-cause signals were identified."
        )

    return hypotheses


# =========================================================
# AUTOMATED CLEANING
# =========================================================

def clean_dataset(df):

    cleaned = df.copy()

    # -----------------------------------------------------
    # Standardize City
    # -----------------------------------------------------

    if "City" in cleaned.columns:

        cleaned["City"] = (
            cleaned["City"]
            .astype("string")
            .str.strip()
            .replace(CITY_MAPPING)
        )

    # -----------------------------------------------------
    # Numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "Age",
        "Quantity",
        "Sales",
        "Discount"
    ]

    for column in numeric_columns:

        if column in cleaned.columns:

            # Convert to float to prevent nullable integer
            # assignment errors when median is decimal.
            cleaned[column] = (
                pd.to_numeric(
                    cleaned[column],
                    errors="coerce"
                )
                .astype("float64")
            )

    # -----------------------------------------------------
    # Invalid Age
    # -----------------------------------------------------

    if "Age" in cleaned.columns:

        invalid_age_mask = (
            (cleaned["Age"] < AGE_MIN) |
            (cleaned["Age"] > AGE_MAX)
        )

        cleaned.loc[
            invalid_age_mask,
            "Age"
        ] = np.nan

    # -----------------------------------------------------
    # Invalid Quantity
    # -----------------------------------------------------

    if "Quantity" in cleaned.columns:

        invalid_quantity_mask = (
            cleaned["Quantity"] < 0
        )

        cleaned.loc[
            invalid_quantity_mask,
            "Quantity"
        ] = np.nan

    # -----------------------------------------------------
    # Fill numeric missing values with median
    # -----------------------------------------------------

    for column in numeric_columns:

        if column in cleaned.columns:

            median_value = cleaned[column].median()

            if pd.notna(median_value):

                cleaned[column] = cleaned[column].fillna(
                    median_value
                )

    # -----------------------------------------------------
    # Fill categorical missing values
    # -----------------------------------------------------

    categorical_columns = cleaned.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns

    for column in categorical_columns:

        if cleaned[column].isna().any():

            cleaned[column] = cleaned[column].fillna(
                "Unknown"
            )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    cleaned = cleaned.drop_duplicates()

    # Reset index
    cleaned = cleaned.reset_index(
        drop=True
    )

    return cleaned


# =========================================================
# QUALITY SCORE
# =========================================================

def calculate_quality_score(df):

    if len(df) == 0:

        return 0

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:

        return 0

    missing_cells = int(
        df.isna().sum().sum()
    )

    missing_ratio = (
        missing_cells / total_cells
    )

    duplicate_ratio = (
        df.duplicated().sum() / len(df)
    )

    score = 100

    score -= missing_ratio * 40

    score -= duplicate_ratio * 30

    score = max(
        0,
        min(100, score)
    )

    return round(score, 1)


# =========================================================
# GEMINI AI
# =========================================================

def get_setting(name):

    value = os.getenv(name)

    if value:

        return value

    try:

        if name in st.secrets:

            return st.secrets[name]

    except Exception:

        pass

    return None


def generate_gemini_analysis(
    df,
    quality_score,
    anomaly_count,
    outlier_summary,
    root_causes
):

    api_key = get_setting(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "Gemini AI is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets "
            "to enable AI-generated analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        missing_total = int(
            df.isna().sum().sum()
        )

        duplicate_count = int(
            df.duplicated().sum()
        )

        prompt = f"""
You are a senior data analyst reviewing a dataset.

Provide a concise professional data-quality and business analysis.

Dataset rows: {len(df)}
Dataset columns: {len(df.columns)}
Quality score: {quality_score}
Missing cells: {missing_total}
Duplicate rows: {duplicate_count}
Numeric columns: {numeric_columns}
ML anomalies detected: {anomaly_count}

IQR outlier summary:
{outlier_summary}

Root-cause hypotheses:
{root_causes}

Give:
1. Overall assessment
2. Most important data-quality issues
3. Business risks
4. Recommended actions
5. Short executive summary

Keep the explanation practical and easy to understand.
"""

        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt
        )

        # Try common response formats
        if hasattr(response, "text"):

            return response.text

        if hasattr(response, "output_text"):

            return response.output_text

        return str(response)

    except Exception as e:

        return (
            f"Gemini AI analysis could not be generated: {e}"
        )


# =========================================================
# POWER BI-READY EXPORTS
# =========================================================

def create_powerbi_exports(df):

    """
    Create separate CSV files that can be imported into Power BI.
    """

    exports = {}

    # Main cleaned dataset
    exports["cleaned_data.csv"] = df.to_csv(
        index=False
    ).encode("utf-8")

    # Data dictionary
    data_dictionary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing Values": [
            int(df[column].isna().sum())
            for column in df.columns
        ],
        "Unique Values": [
            int(df[column].nunique())
            for column in df.columns
        ]
    })

    exports["data_dictionary.csv"] = (
        data_dictionary
        .to_csv(index=False)
        .encode("utf-8")
    )

    # Quality summary
    quality_summary = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Cells",
            "Duplicate Rows",
            "Quality Score"
        ],
        "Value": [
            len(df),
            len(df.columns),
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            calculate_quality_score(df)
        ]
    })

    exports["quality_summary.csv"] = (
        quality_summary
        .to_csv(index=False)
        .encode("utf-8")
    )

    # Numeric summary
    numeric_df = df.select_dtypes(
        include=np.number
    )

    if not numeric_df.empty:

        numeric_summary = (
            numeric_df
            .describe()
            .reset_index()
        )

    else:

        numeric_summary = pd.DataFrame({
            "Message": [
                "No numeric columns available."
            ]
        })

    exports["numeric_summary.csv"] = (
        numeric_summary
        .to_csv(index=False)
        .encode("utf-8")
    )

    # Missing value summary
    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": [
            int(df[column].isna().sum())
            for column in df.columns
        ]
    })

    exports["missing_summary.csv"] = (
        missing_summary
        .to_csv(index=False)
        .encode("utf-8")
    )

    return exports


def create_zip_file(files):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as zip_file:

        for file_name, file_data in files.items():

            zip_file.writestr(
                file_name,
                file_data
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ DataGuard AI"
    )

    st.markdown(
        """
        **AI-powered data quality and
        anomaly detection platform**
        """
    )

    st.divider()

    st.markdown(
        "### Pipeline"
    )

    st.markdown(
        """
        1. 📂 Upload Data  
        2. 🔎 Profile Data  
        3. 🧹 Quality Checks  
        4. 📊 Outlier Detection  
        5. 🤖 ML Anomalies  
        6. 🔍 Root Cause  
        7. 🧠 Gemini AI  
        8. ✨ Clean Data  
        9. 📈 Power BI Export  
        10. 📋 Final Report
        """
    )

    st.divider()

    st.info(
        "Power BI integration uses downloadable "
        "CSV files. No Azure account is required."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ DataGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered data quality, anomaly detection,
    cleaning and business insight platform.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📂 Upload Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel dataset to start the analysis."
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

df = load_uploaded_file(
    uploaded_file
)

if df is None:

    st.stop()


if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# =========================================================
# BASIC INFO
# =========================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )

with col2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}"
    )

with col3:

    st.metric(
        "Missing Cells",
        f"{df.isna().sum().sum():,}"
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )


# =========================================================
# DATA PREVIEW
# =========================================================

st.markdown(
    '<div class="section-title">👀 Data Preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(20),
    use_container_width=True
)


# =========================================================
# COLUMN PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Column Profiling</div>',
    unsafe_allow_html=True
)

numeric_columns, categorical_columns, datetime_columns = (
    classify_columns(df)
)

profile_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": [
        str(df[column].dtype)
        for column in df.columns
    ],
    "Missing Values": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
    "Unique Values": [
        int(df[column].nunique())
        for column in df.columns
    ]
})

st.dataframe(
    profile_df,
    use_container_width=True
)

st.write(
    f"**Numeric columns:** {len(numeric_columns)}"
)

st.write(
    f"**Categorical columns:** {len(categorical_columns)}"
)

st.write(
    f"**Datetime columns:** {len(datetime_columns)}"
)


# =========================================================
# QUALITY CHECKS
# =========================================================

st.markdown(
    '<div class="section-title">🧹 Data Quality Checks</div>',
    unsafe_allow_html=True
)

quality_results = check_invalid_values(
    df
)

city_info = check_city_inconsistencies(
    df
)

missing_total = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)


q1, q2, q3, q4 = st.columns(4)

with q1:

    st.metric(
        "Missing Cells",
        missing_total
    )

with q2:

    st.metric(
        "Duplicate Rows",
        duplicate_count
    )

with q3:

    invalid_total = sum(
        quality_results.values()
    )

    st.metric(
        "Invalid Values",
        invalid_total
    )

with q4:

    st.metric(
        "City Issues",
        city_info["count"]
    )


# ---------------------------------------------------------
# Missing Values
# ---------------------------------------------------------

if missing_total > 0:

    st.warning(
        f"Found {missing_total:,} missing cells."
    )

else:

    st.success(
        "No missing cells detected."
    )


# ---------------------------------------------------------
# Duplicates
# ---------------------------------------------------------

if duplicate_count > 0:

    st.warning(
        f"Found {duplicate_count:,} duplicate rows."
    )

else:

    st.success(
        "No duplicate rows detected."
    )


# ---------------------------------------------------------
# Invalid values
# ---------------------------------------------------------

if quality_results:

    invalid_df = pd.DataFrame(
        list(quality_results.items()),
        columns=[
            "Check",
            "Invalid Count"
        ]
    )

    st.dataframe(
        invalid_df,
        use_container_width=True
    )


# ---------------------------------------------------------
# City inconsistencies
# ---------------------------------------------------------

if city_info["found"]:

    st.warning(
        "Inconsistent city names detected:"
    )

    st.write(
        city_info["values"]
    )

else:

    if "City" in df.columns:

        st.success(
            "No known city naming inconsistencies detected."
        )


# =========================================================
# IQR OUTLIERS
# =========================================================

st.markdown(
    '<div class="section-title">📊 IQR Outlier Detection</div>',
    unsafe_allow_html=True
)

outlier_summary, outlier_masks = detect_iqr_outliers(
    df
)

if outlier_summary:

    outlier_table = []

    for column, details in outlier_summary.items():

        outlier_table.append({
            "Column": column,
            "Outliers": details["count"],
            "Lower Bound": round(
                details["lower_bound"],
                2
            ),
            "Upper Bound": round(
                details["upper_bound"],
                2
            )
        })

    outlier_df = pd.DataFrame(
        outlier_table
    )

    st.dataframe(
        outlier_df,
        use_container_width=True
    )

else:

    st.info(
        "No suitable numeric columns were available "
        "for IQR outlier analysis."
    )


# =========================================================
# BUSINESS VISUALIZATION
# =========================================================

st.markdown(
    '<div class="section-title">📈 Business Analysis</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Sales Distribution
# ---------------------------------------------------------

if "Sales" in df.columns:

    sales = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    st.markdown(
        "### Sales Distribution"
    )

    st.bar_chart(
        sales.dropna()
    )


# ---------------------------------------------------------
# Category Analysis
# ---------------------------------------------------------

possible_category_columns = [
    "Category",
    "Segment",
    "Region",
    "City"
]

available_category = None

for column in possible_category_columns:

    if column in df.columns:

        available_category = column

        break


if available_category:

    st.markdown(
        f"### {available_category} Distribution"
    )

    category_counts = (
        df[available_category]
        .astype("string")
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        category_counts
    )


# =========================================================
# DATE TREND
# =========================================================

if datetime_columns:

    date_column = datetime_columns[0]

    trend_df = df.copy()

    trend_df[date_column] = safe_to_datetime(
        trend_df[date_column]
    )

    trend_df = trend_df.dropna(
        subset=[date_column]
    )

    if not trend_df.empty:

        st.markdown(
            f"### 📅 Trend by {date_column}"
        )

        if "Sales" in trend_df.columns:

            trend_df["Sales"] = pd.to_numeric(
                trend_df["Sales"],
                errors="coerce"
            )

            trend = (
                trend_df
                .set_index(date_column)["Sales"]
                .resample("ME")
                .sum()
            )

            st.line_chart(
                trend
            )


# =========================================================
# ML ANOMALY DETECTION
# =========================================================

st.markdown(
    '<div class="section-title">🤖 Machine Learning Anomaly Detection</div>',
    unsafe_allow_html=True
)

ml_df, anomaly_count = run_ml_anomaly_detection(
    df
)

if anomaly_count > 0:

    st.warning(
        f"Isolation Forest detected "
        f"{anomaly_count:,} potential anomalous records."
    )

    anomaly_rows = ml_df[
        ml_df["ML_Anomaly"] == 1
    ]

    st.dataframe(
        anomaly_rows.head(50),
        use_container_width=True
    )

else:

    st.success(
        "No significant ML anomalies were detected."
    )


# =========================================================
# QUALITY SCORE
# =========================================================

st.markdown(
    '<div class="section-title">⭐ Data Quality Score</div>',
    unsafe_allow_html=True
)

quality_score = calculate_quality_score(
    df
)

score_col1, score_col2 = st.columns(2)

with score_col1:

    st.metric(
        "Quality Score",
        f"{quality_score}/100"
    )

with score_col2:

    if quality_score >= 90:

        st.success(
            "Excellent data quality"
        )

    elif quality_score >= 75:

        st.info(
            "Good data quality with some issues"
        )

    elif quality_score >= 50:

        st.warning(
            "Data quality needs improvement"
        )

    else:

        st.error(
            "Poor data quality"
        )


# =========================================================
# ROOT CAUSE ANALYSIS
# =========================================================

st.markdown(
    '<div class="section-title">🔍 Root-Cause Analysis</div>',
    unsafe_allow_html=True
)

root_causes = generate_root_cause_hypotheses(
    df,
    quality_results,
    outlier_summary,
    anomaly_count
)

for cause in root_causes:

    st.write(
        f"• {cause}"
    )


# =========================================================
# GEMINI AI
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Gemini AI Analysis</div>',
    unsafe_allow_html=True
)

with st.spinner(
    "Generating AI-powered analysis..."
):

    gemini_analysis = generate_gemini_analysis(
        df,
        quality_score,
        anomaly_count,
        outlier_summary,
        root_causes
    )

st.markdown(
    gemini_analysis
)


# =========================================================
# DATA CLEANING
# =========================================================

st.markdown(
    '<div class="section-title">✨ Automated Data Cleaning</div>',
    unsafe_allow_html=True
)

cleaned_df = clean_dataset(
    df
)

before_rows = len(df)
after_rows = len(cleaned_df)

before_missing = int(
    df.isna().sum().sum()
)

after_missing = int(
    cleaned_df.isna().sum().sum()
)

before_duplicates = int(
    df.duplicated().sum()
)

after_duplicates = int(
    cleaned_df.duplicated().sum()
)


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Rows Before",
        before_rows
    )

with c2:

    st.metric(
        "Rows After",
        after_rows
    )

with c3:

    st.metric(
        "Missing Before",
        before_missing
    )

with c4:

    st.metric(
        "Missing After",
        after_missing
    )


st.success(
    "Automated cleaning completed successfully."
)


with st.expander(
    "👀 Preview Cleaned Dataset"
):

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True
    )


# =========================================================
# DOWNLOAD CLEANED DATA
# =========================================================

st.markdown(
    '<div class="section-title">⬇️ Download Cleaned Data</div>',
    unsafe_allow_html=True
)

cleaned_csv = cleaned_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Cleaned CSV",
    data=cleaned_csv,
    file_name="dataguard_cleaned_data.csv",
    mime="text/csv"
)


# =========================================================
# POWER BI READY EXPORT
# =========================================================

st.markdown(
    '<div class="section-title">📊 Power BI-Ready Export</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

    DataGuard AI prepares clean, structured CSV files
    that can be imported directly into Power BI.

    No Azure account or cloud configuration is required.

    </div>
    """,
    unsafe_allow_html=True
)

powerbi_files = create_powerbi_exports(
    cleaned_df
)

powerbi_zip = create_zip_file(
    powerbi_files
)

st.download_button(
    label="📦 Download Power BI ZIP",
    data=powerbi_zip,
    file_name="dataguard_powerbi_export.zip",
    mime="application/zip"
)


st.markdown(
    "### Individual Power BI Files"
)

for file_name, file_data in powerbi_files.items():

    st.download_button(
        label=f"⬇️ {file_name}",
        data=file_data,
        file_name=file_name,
        mime="text/csv",
        key=f"download_{file_name}"
    )


st.info(
    "Import the downloaded CSV files into Power BI Desktop "
    "to create dashboards and reports."
)


# =========================================================
# FINAL REPORT
# =========================================================

st.markdown(
    '<div class="section-title">📋 Final DataGuard AI Report</div>',
    unsafe_allow_html=True
)

final_report = pd.DataFrame({
    "Metric": [
        "Original Rows",
        "Cleaned Rows",
        "Original Columns",
        "Missing Cells Before",
        "Missing Cells After",
        "Duplicate Rows Before",
        "Duplicate Rows After",
        "IQR Outlier Columns",
        "ML Anomalies",
        "Original Quality Score"
    ],
    "Value": [
        before_rows,
        after_rows,
        len(df.columns),
        before_missing,
        after_missing,
        before_duplicates,
        after_duplicates,
        len(outlier_summary),
        anomaly_count,
        quality_score
    ]
})

st.dataframe(
    final_report,
    use_container_width=True
)


# =========================================================
# FINAL SUCCESS MESSAGE
# =========================================================

st.markdown(
    """
    <div class="success-box">

    <h3>✅ DataGuard AI Analysis Completed</h3>

    <p>
    Your dataset has successfully passed through:
    </p>

    <p>
    📂 Data Upload →
    🔎 Profiling →
    🧹 Quality Checks →
    📊 Outlier Detection →
    🤖 ML Anomaly Detection →
    🔍 Root-Cause Analysis →
    🧠 Gemini AI →
    ✨ Data Cleaning →
    📈 Power BI Export
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "DataGuard AI • AI-powered Data Quality & Anomaly Detection Platform"
)
