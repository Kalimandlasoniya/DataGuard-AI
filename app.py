# ============================================================
# DataGuard AI
# Intelligent Data Quality & Root-Cause Analysis System
# ============================================================

import os
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DataGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

GEMINI_MODEL = "gemini-3.6-flash"

CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BANGALORE": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru"
}

EXPECTED_AGE_MIN = 0
EXPECTED_AGE_MAX = 120

# Isolation Forest screening sensitivity.
# This is NOT proof that exactly this percentage is erroneous.
ML_CONTAMINATION = 0.02

POWERBI_FOLDER = Path.cwd() / "powerbi_export"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-note {
        padding: 12px 16px;
        border-radius: 10px;
        background: #f1f5f9;
        border-left: 4px solid #2563eb;
        margin: 10px 0 20px 0;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ DataGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Data Quality & Root-Cause Analysis System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    Upload a CSV or Excel dataset and DataGuard AI will:

    **Profile → Detect Issues → Detect Outliers → Detect ML Anomalies
    → Analyze Root Causes → Clean Data → Export for Power BI**
    """
)


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        return None

    try:
        from google import genai

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None


client = get_gemini_client()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ DataGuard AI")

st.sidebar.markdown("### Pipeline")

pipeline_steps = [
    "1. Upload Data",
    "2. Profile Dataset",
    "3. Detect Quality Issues",
    "4. Detect Statistical Outliers",
    "5. Detect ML Anomalies",
    "6. Analyze Root Causes",
    "7. Clean Data",
    "8. Export for Power BI",
    "9. Generate Report"
]

for step in pipeline_steps:
    st.sidebar.write(step)

st.sidebar.markdown("---")

st.sidebar.markdown("### ML Settings")

st.sidebar.info(
    f"""
**Isolation Forest**

Configured screening sensitivity:
**{ML_CONTAMINATION * 100:.0f}%**

This controls how aggressively the model flags
potentially unusual records.

ML flags are not automatically data errors.
"""
)

st.sidebar.markdown("---")

if client:
    st.sidebar.success("Gemini AI: Connected")
else:
    st.sidebar.warning("Gemini AI: Not configured")

st.sidebar.caption(
    "Gemini API keys are read from environment variables "
    "or Streamlit Secrets and are never displayed."
)


# ============================================================
# SAFE DATETIME CONVERSION
# ============================================================

def safe_to_datetime(series):

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

            return pd.Series(
                pd.NaT,
                index=series.index
            )


# ============================================================
# DATETIME COLUMN DETECTION
# ============================================================

def detect_datetime_columns(df):

    datetime_columns = []

    date_keywords = (
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "modified",
        "dob",
        "birth",
        "day"
    )

    for column in df.columns:

        series = df[column]

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):

            datetime_columns.append(column)

            continue

        # Never interpret numeric IDs as dates
        if pd.api.types.is_numeric_dtype(series):

            continue

        # Only inspect object/string columns
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):

            continue

        converted = safe_to_datetime(series)

        if len(series) == 0:
            continue

        valid_ratio = converted.notna().mean()

        # Require strong evidence
        if valid_ratio < 0.85:
            continue

        column_name = str(column).lower()

        sample = (
            series
            .dropna()
            .astype(str)
            .head(50)
        )

        if len(sample) == 0:
            continue

        looks_date_like = (
            sample
            .str.contains(
                r"[-/:]",
                regex=True
            )
            .mean()
            >= 0.50
        )

        if (
            any(
                keyword in column_name
                for keyword in date_keywords
            )
            or looks_date_like
        ):

            datetime_columns.append(column)

    return datetime_columns


# ============================================================
# DATA LOADING
# ============================================================

def load_data(uploaded_file):

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        elif file_name.endswith(".xlsx"):

            df = pd.read_excel(
                uploaded_file
            )

        elif file_name.endswith(".xls"):

            df = pd.read_excel(
                uploaded_file
            )

        else:

            st.error(
                "Unsupported file format."
            )

            return None

        # Detect dates safely
        datetime_columns = detect_datetime_columns(
            df
        )

        for column in datetime_columns:

            converted = safe_to_datetime(
                df[column]
            )

            if converted.notna().mean() >= 0.85:

                df[column] = converted

        return df

    except Exception as e:

        st.error(
            f"Error loading file: {e}"
        )

        return None


# ============================================================
# COLUMN CLASSIFICATION
# ============================================================

def classify_columns(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    datetime_columns = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # Remove datetime columns from categorical list
    categorical_columns = [
        column
        for column in categorical_columns
        if column not in datetime_columns
    ]

    identifier_columns = []

    for column in df.columns:

        column_lower = str(
            column
        ).lower()

        if (
            column_lower.endswith("_id")
            or column_lower == "id"
            or "customer_id" in column_lower
            or "order_id" in column_lower
        ):

            identifier_columns.append(
                column
            )

    return (
        numeric_columns,
        categorical_columns,
        datetime_columns,
        identifier_columns
    )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

def check_missing_values(df):

    missing = df.isnull().sum()

    missing = missing[
        missing > 0
    ]

    if len(missing) == 0:

        return pd.DataFrame(
            columns=[
                "Column",
                "Missing_Count",
                "Missing_Percentage"
            ]
        )

    return pd.DataFrame({

        "Column":
            missing.index,

        "Missing_Count":
            missing.values,

        "Missing_Percentage":
            (
                missing.values
                /
                len(df)
                *
                100
            ).round(2)

    })


# ============================================================
# DUPLICATE CHECK
# ============================================================

def check_duplicates(df):

    return int(
        df.duplicated().sum()
    )


# ============================================================
# INVALID VALUE CHECK
# ============================================================

def check_invalid_values(df):

    issues = []

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    if "Age" in df.columns:

        age_numeric = pd.to_numeric(
            df["Age"],
            errors="coerce"
        )

        invalid_age = (
            (
                age_numeric
                < EXPECTED_AGE_MIN
            )
            |
            (
                age_numeric
                > EXPECTED_AGE_MAX
            )
        ).sum()

        if invalid_age > 0:

            issues.append({

                "Column":
                    "Age",

                "Issue":
                    "Invalid age",

                "Count":
                    int(invalid_age),

                "Severity":
                    "High"

            })

    # --------------------------------------------------------
    # QUANTITY
    # --------------------------------------------------------

    if "Quantity" in df.columns:

        quantity_numeric = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        invalid_quantity = (
            quantity_numeric < 0
        ).sum()

        if invalid_quantity > 0:

            issues.append({

                "Column":
                    "Quantity",

                "Issue":
                    "Negative quantity",

                "Count":
                    int(invalid_quantity),

                "Severity":
                    "High"

            })

    # --------------------------------------------------------
    # DISCOUNT
    # --------------------------------------------------------

    if "Discount" in df.columns:

        discount_numeric = pd.to_numeric(
            df["Discount"],
            errors="coerce"
        )

        invalid_discount = (
            (
                discount_numeric < 0
            )
            |
            (
                discount_numeric > 1
            )
        ).sum()

        if invalid_discount > 0:

            issues.append({

                "Column":
                    "Discount",

                "Issue":
                    "Discount outside 0–1 range",

                "Count":
                    int(invalid_discount),

                "Severity":
                    "Medium"

            })

    return pd.DataFrame(
        issues
    )


# ============================================================
# CITY CONSISTENCY CHECK
# ============================================================

def find_city_inconsistencies(df):

    if "City" not in df.columns:

        return pd.DataFrame(
            columns=[
                "Original_Value",
                "Standard_Value",
                "Count"
            ]
        )

    city_counts = (
        df["City"]
        .astype("string")
        .value_counts()
    )

    inconsistent_values = []

    for value in city_counts.index:

        value_str = str(
            value
        )

        if value_str in CITY_MAPPING:

            inconsistent_values.append({

                "Original_Value":
                    value_str,

                "Standard_Value":
                    CITY_MAPPING[value_str],

                "Count":
                    int(city_counts[value])

            })

    return pd.DataFrame(
        inconsistent_values
    )


# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

def detect_iqr_outliers(
    df,
    column
):

    numeric_series = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(numeric_series) < 4:

        return {

            "outliers":
                pd.DataFrame(),

            "Q1":
                np.nan,

            "Q3":
                np.nan,

            "IQR":
                np.nan,

            "Lower_Bound":
                np.nan,

            "Upper_Bound":
                np.nan,

            "Count":
                0

        }

    Q1 = numeric_series.quantile(
        0.25
    )

    Q3 = numeric_series.quantile(
        0.75
    )

    IQR = Q3 - Q1

    lower_bound = (
        Q1 - 1.5 * IQR
    )

    upper_bound = (
        Q3 + 1.5 * IQR
    )

    numeric_values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    mask = (
        (
            numeric_values
            < lower_bound
        )
        |
        (
            numeric_values
            > upper_bound
        )
    )

    outliers = df.loc[
        mask
    ].copy()

    return {

        "outliers":
            outliers,

        "Q1":
            Q1,

        "Q3":
            Q3,

        "IQR":
            IQR,

        "Lower_Bound":
            lower_bound,

        "Upper_Bound":
            upper_bound,

        "Count":
            len(outliers)

    }


# ============================================================
# ALL STATISTICAL OUTLIERS
# ============================================================

def detect_all_statistical_outliers(
    df
):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    results = []

    outlier_details = {}

    for column in numeric_columns:

        column_lower = str(
            column
        ).lower()

        # Skip identifier columns
        if (
            column_lower.endswith("_id")
            or column_lower == "id"
        ):

            continue

        result = detect_iqr_outliers(
            df,
            column
        )

        outlier_details[
            column
        ] = result

        results.append({

            "Column":
                column,

            "Q1":
                round(
                    result["Q1"],
                    4
                )
                if pd.notna(
                    result["Q1"]
                )
                else np.nan,

            "Q3":
                round(
                    result["Q3"],
                    4
                )
                if pd.notna(
                    result["Q3"]
                )
                else np.nan,

            "IQR":
                round(
                    result["IQR"],
                    4
                )
                if pd.notna(
                    result["IQR"]
                )
                else np.nan,

            "Lower_Bound":
                round(
                    result["Lower_Bound"],
                    4
                )
                if pd.notna(
                    result["Lower_Bound"]
                )
                else np.nan,

            "Upper_Bound":
                round(
                    result["Upper_Bound"],
                    4
                )
                if pd.notna(
                    result["Upper_Bound"]
                )
                else np.nan,

            "Outlier_Count":
                result["Count"]

        })

    return (
        pd.DataFrame(results),
        outlier_details
    )


# ============================================================
# ML ANOMALY DETECTION
# ============================================================

def detect_ml_anomalies(
    df
):

    preferred_features = [
        "Age",
        "Quantity",
        "Sales",
        "Discount"
    ]

    anomaly_features = [

        column

        for column in preferred_features

        if column in df.columns

    ]

    # Fallback
    if len(anomaly_features) < 2:

        numeric_columns = (
            df
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        anomaly_features = [

            column

            for column in numeric_columns

            if not (
                str(column)
                .lower()
                .endswith("_id")
                or
                str(column)
                .lower()
                == "id"
            )

        ]

    if len(anomaly_features) < 2:

        result_df = df.copy()

        return (
            result_df,
            [],
            0
        )

    ml_data = df[
        anomaly_features
    ].copy()

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    for column in anomaly_features:

        ml_data[column] = pd.to_numeric(
            ml_data[column],
            errors="coerce"
        )

        median_value = (
            ml_data[column]
            .median()
        )

        if pd.isna(
            median_value
        ):

            median_value = 0

        ml_data[column] = (
            ml_data[column]
            .fillna(median_value)
        )

    # --------------------------------------------------------
    # SMALL DATASET
    # --------------------------------------------------------

    if len(ml_data) < 10:

        result_df = df.copy()

        result_df[
            "ML_Anomaly"
        ] = 1

        return (
            result_df,
            anomaly_features,
            0
        )

    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    model = IsolationForest(

        n_estimators=200,

        contamination=
            ML_CONTAMINATION,

        random_state=42

    )

    predictions = model.fit_predict(
        ml_data
    )

    result_df = df.copy()

    result_df[
        "ML_Anomaly"
    ] = predictions

    anomaly_count = int(
        (
            result_df[
                "ML_Anomaly"
            ]
            == -1
        ).sum()
    )

    return (
        result_df,
        anomaly_features,
        anomaly_count
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    df,
    missing_count,
    duplicate_count,
    invalid_count,
    city_count
):

    if len(df) == 0:

        return 100.0

    total_cells = max(
        len(df)
        *
        max(
            len(df.columns),
            1
        ),
        1
    )

    missing_rate = (
        missing_count
        /
        total_cells
    )

    duplicate_rate = (
        duplicate_count
        /
        len(df)
    )

    invalid_rate = (
        invalid_count
        /
        len(df)
    )

    city_rate = (
        city_count
        /
        len(df)
    )

    # --------------------------------------------------------
    # Weighted penalties
    # --------------------------------------------------------

    score = 100 - (

        missing_rate * 25

        +

        duplicate_rate * 25

        +

        invalid_rate * 30

        +

        city_rate * 20

    )

    return round(
        max(
            0,
            min(
                100,
                score
            )
        ),
        2
    )


# ============================================================
# QUALITY LABEL
# ============================================================

def quality_label(
    score
):

    if score >= 90:

        return "🟢 Excellent"

    elif score >= 75:

        return "🟡 Good"

    elif score >= 60:

        return "🟠 Needs Improvement"

    else:

        return "🔴 Poor"


# ============================================================
# RULE-BASED ROOT CAUSE ANALYSIS
# ============================================================

def rule_based_root_cause(
    missing_count,
    duplicate_count,
    invalid_count,
    city_count,
    statistical_outlier_count,
    ml_anomaly_count
):

    causes = []

    if missing_count > 0:

        causes.append({

            "Issue":
                "Missing Values",

            "Possible Root Cause":
                "Incomplete data entry, unavailable source "
                "values, or missing information during data integration.",

            "Recommendation":
                "Investigate the source system and define an "
                "appropriate missing-value handling strategy."

        })

    if duplicate_count > 0:

        causes.append({

            "Issue":
                "Duplicate Records",

            "Possible Root Cause":
                "Repeated data entry, duplicate transactions, "
                "or repeated ingestion of the same records.",

            "Recommendation":
                "Check unique identifiers and ingestion processes "
                "before removing duplicates."

        })

    if invalid_count > 0:

        causes.append({

            "Issue":
                "Invalid Values",

            "Possible Root Cause":
                "Incorrect data entry, invalid business values, "
                "or source-system validation problems.",

            "Recommendation":
                "Add validation rules at the point where data is created."

        })

    if city_count > 0:

        causes.append({

            "Issue":
                "City Standardization",

            "Possible Root Cause":
                "Different spellings, abbreviations, or naming "
                "conventions across data sources.",

            "Recommendation":
                "Use a standardized master-data mapping."

        })

    if statistical_outlier_count > 0:

        causes.append({

            "Issue":
                "Statistical Outliers",

            "Possible Root Cause":
                "Legitimate unusual transactions, extreme business "
                "events, data-entry errors, or unusual customer behavior.",

            "Recommendation":
                "Investigate individual records before changing "
                "or deleting them."

        })

    if ml_anomaly_count > 0:

        causes.append({

            "Issue":
                "ML Anomalies",

            "Possible Root Cause":
                "Unusual combinations of multiple numeric features.",

            "Recommendation":
                "Review flagged records with domain context. "
                "ML anomalies are not automatically data errors or fraud."

        })

    return pd.DataFrame(
        causes
    )


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def ask_gemini(
    issue_summary
):

    if client is None:

        return (
            "### Gemini AI unavailable\n\n"

            "Gemini AI is not configured for this deployment.\n\n"

            "For local or Colab execution, configure the "
            "`GEMINI_API_KEY` environment variable.\n\n"

            "For Streamlit Community Cloud, add "
            "`GEMINI_API_KEY` under App Settings → Secrets.\n\n"

            "Never place your API key directly inside `app.py` "
            "or upload it to GitHub."
        )

    prompt = f"""

You are a senior data-quality analyst.

Analyze the following DataGuard AI quality results.

IMPORTANT RULES:

1. Root causes are hypotheses, not confirmed facts.
2. Do not invent facts that are not present in the supplied results.
3. If the exact cause cannot be determined, say:
   "The exact root cause requires further investigation."
4. Statistical outliers are not automatically errors.
5. Isolation Forest detects unusual multivariate patterns.
6. Gemini is explaining results produced by DataGuard AI.
7. ML anomalies are NOT proof of fraud.
8. Do not recommend automatically deleting anomalies.
9. Do not invent relationships that are not supplied.
10. Give practical recommendations.
11. Use professional but simple language.
12. Clearly separate confirmed quality issues from statistical
    observations and ML screening results.

DATA QUALITY SUMMARY:

{issue_summary}

Provide the analysis using these sections:

### 1. Overall Assessment

### 2. Confirmed Data Quality Issues

### 3. Missing Value Analysis

### 4. Duplicate Analysis

### 5. Invalid Value Analysis

### 6. City Standardization Analysis

### 7. Statistical Outlier Analysis

### 8. ML Anomaly Analysis

### 9. Possible Root Causes

### 10. Recommended Actions

### 11. Priority Order

### 12. Business Impact
"""

    try:

        response = client.interactions.create(

            model=GEMINI_MODEL,

            input=prompt,

            generation_config={
                "temperature": 0.2
            }

        )

        output = getattr(
            response,
            "output_text",
            None
        )

        if output and output.strip():

            return output.strip()

        return (
            "Gemini returned an empty response. "
            "Please try again."
        )

    except Exception as e:

        return (
            "### Gemini AI Error\n\n"

            "Gemini could not complete the analysis.\n\n"

            f"Error: {str(e)}"
        )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(
    df,
    identifier_columns
):

    cleaned_df = df.copy()

    cleaning_log = []

    # --------------------------------------------------------
    # CITY STANDARDIZATION
    # --------------------------------------------------------

    if "City" in cleaned_df.columns:

        city_before = (
            cleaned_df["City"].copy()
        )

        cleaned_df[
            "City"
        ] = (
            cleaned_df["City"]
            .replace(CITY_MAPPING)
        )

        changed = int(
            (
                city_before
                != cleaned_df["City"]
            )
            .fillna(False)
            .sum()
        )

        if changed > 0:

            cleaning_log.append({

                "Action":
                    "Standardized city values",

                "Column":
                    "City",

                "Affected_Rows":
                    changed

            })

    # --------------------------------------------------------
    # INVALID VALUE RULES
    # --------------------------------------------------------

    invalid_rules = {

        "Age":
            lambda x:
                (
                    (x < EXPECTED_AGE_MIN)
                    |
                    (x > EXPECTED_AGE_MAX)
                ),

        "Quantity":
            lambda x:
                x < 0,

        "Discount":
            lambda x:
                (
                    (x < 0)
                    |
                    (x > 1)
                )

    }

    for column, rule in invalid_rules.items():

        if column not in cleaned_df.columns:
            continue

        numeric_values = pd.to_numeric(
            cleaned_df[column],
            errors="coerce"
        )

        invalid_mask = rule(
            numeric_values
        )

        count = int(
            invalid_mask.sum()
        )

        if count > 0:

            cleaned_df.loc[
                invalid_mask,
                column
            ] = np.nan

            cleaning_log.append({

                "Action":
                    f"Corrected invalid {column} values",

                "Column":
                    column,

                "Affected_Rows":
                    count

            })

    # --------------------------------------------------------
    # FILL MISSING NUMERIC VALUES
    # --------------------------------------------------------

    numeric_columns = (
        cleaned_df
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    numeric_columns = [

        column

        for column in numeric_columns

        if column not in identifier_columns

    ]

    total_filled = 0

    for column in numeric_columns:

        before_missing = int(
            cleaned_df[column]
            .isnull()
            .sum()
        )

        if before_missing == 0:
            continue

        # Convert to float to avoid
        # nullable integer assignment errors
        cleaned_df[column] = (
            pd.to_numeric(
                cleaned_df[column],
                errors="coerce"
            )
            .astype("float64")
        )

        median_value = (
            cleaned_df[column]
            .median()
        )

        if pd.notna(
            median_value
        ):

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna(median_value)
            )

            after_missing = int(
                cleaned_df[column]
                .isnull()
                .sum()
            )

            filled = (
                before_missing
                -
                after_missing
            )

            total_filled += filled

            if filled > 0:

                cleaning_log.append({

                    "Action":
                        "Filled missing numeric values",

                    "Column":
                        column,

                    "Affected_Rows":
                        filled

                })

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    before_duplicates = len(
        cleaned_df
    )

    cleaned_df = (
        cleaned_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_duplicates
        -
        len(cleaned_df)
    )

    if duplicates_removed > 0:

        cleaning_log.append({

            "Action":
                "Removed duplicate records",

            "Column":
                "All Columns",

            "Affected_Rows":
                duplicates_removed

        })

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    if not cleaning_log:

        cleaning_log.append({

            "Action":
                "No cleaning required",

            "Column":
                "N/A",

            "Affected_Rows":
                0

        })

    cleaning_log_df = pd.DataFrame(
        cleaning_log
    )

    return (
        cleaned_df,
        cleaning_log_df,
        total_filled
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    df,
    duplicate_count,
    invalid_df,
    city_df,
    statistical_df,
    ml_anomaly_count,
    quality_score,
    cleaned_df
):

    invalid_total = (

        int(
            invalid_df["Count"].sum()
        )

        if not invalid_df.empty

        else 0

    )

    city_total = (

        int(
            city_df["Count"].sum()
        )

        if not city_df.empty

        else 0

    )

    statistical_total = (

        int(
            statistical_df[
                "Outlier_Count"
            ].sum()
        )

        if not statistical_df.empty

        else 0

    )

    lines = [

        "DATAGUARD AI - DATA QUALITY REPORT",

        "=" * 60,

        f"Rows: {len(df)}",

        f"Columns: {len(df.columns)}",

        f"Missing Cells: "
        f"{int(df.isnull().sum().sum())}",

        f"Duplicate Records: "
        f"{duplicate_count}",

        f"Invalid Values: "
        f"{invalid_total}",

        f"City Records to Standardize: "
        f"{city_total}",

        f"Statistical Outlier Flags: "
        f"{statistical_total}",

        f"ML Anomalies: "
        f"{ml_anomaly_count}",

        f"Data Quality Score: "
        f"{quality_score}/100",

        f"Quality Status: "
        f"{quality_label(quality_score)}",

        "",

        "INTERPRETATION",

        "-" * 60,

        "Confirmed quality issues include missing values, "
        "duplicates, rule-based invalid values, and "
        "standardization issues.",

        "Statistical outliers are unusual observations "
        "and are not automatically errors.",

        "ML anomalies represent unusual multivariate "
        "patterns and are not automatically fraud or errors.",

        "",

        "STATISTICAL OUTLIERS",

        "-" * 60

    ]

    if statistical_df.empty:

        lines.append(
            "No numeric statistical outlier analysis available."
        )

    else:

        for _, row in (
            statistical_df.iterrows()
        ):

            lines.append(

                f"{row['Column']}: "
                f"{int(row['Outlier_Count'])} outliers"

            )

    lines.extend([

        "",

        "CITY STANDARDIZATION",

        "-" * 60

    ])

    if city_df.empty:

        lines.append(
            "No city inconsistencies detected."
        )

    else:

        for _, row in (
            city_df.iterrows()
        ):

            lines.append(

                f"{row['Original_Value']} -> "
                f"{row['Standard_Value']} "
                f"({int(row['Count'])} records)"

            )

    lines.extend([

        "",

        "CLEANED DATA",

        "-" * 60,

        f"Cleaned Rows: "
        f"{len(cleaned_df)}",

        f"Rows Removed: "
        f"{len(df) - len(cleaned_df)}"

    ])

    return "\n".join(
        lines
    )


# ============================================================
# POWER BI EXPORT
# ============================================================

def create_powerbi_export(
    original_df,
    cleaned_df,
    missing_df,
    invalid_df,
    city_df,
    statistical_df,
    ml_df,
    cleaning_log,
    quality_score
):

    POWERBI_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # QUALITY SUMMARY
    # --------------------------------------------------------

    quality_summary = pd.DataFrame({

        "Metric": [

            "Original Rows",

            "Original Columns",

            "Missing Cells",

            "Duplicate Records",

            "Invalid Values",

            "City Records to Standardize",

            "Statistical Outliers",

            "ML Anomalies",

            "Cleaned Rows",

            "Rows Removed",

            "Data Quality Score"

        ],

        "Value": [

            len(original_df),

            len(original_df.columns),

            int(
                original_df
                .isnull()
                .sum()
                .sum()
            ),

            int(
                original_df
                .duplicated()
                .sum()
            ),

            (
                int(
                    invalid_df["Count"].sum()
                )
                if not invalid_df.empty
                else 0
            ),

            (
                int(
                    city_df["Count"].sum()
                )
                if not city_df.empty
                else 0
            ),

            (
                int(
                    statistical_df[
                        "Outlier_Count"
                    ].sum()
                )
                if not statistical_df.empty
                else 0
            ),

            (
                int(
                    (
                        ml_df[
                            "ML_Anomaly"
                        ]
                        == -1
                    ).sum()
                )
                if "ML_Anomaly"
                in ml_df.columns
                else 0
            ),

            len(cleaned_df),

            (
                len(original_df)
                -
                len(cleaned_df)
            ),

            quality_score

        ]

    })

    # --------------------------------------------------------
    # QUALITY ISSUES
    # --------------------------------------------------------

    quality_issues = []

    if not missing_df.empty:

        for _, row in (
            missing_df.iterrows()
        ):

            quality_issues.append({

                "Issue_Type":
                    "Missing Value",

                "Column":
                    row["Column"],

                "Count":
                    row["Missing_Count"],

                "Severity":
                    "Medium"

            })

    if not invalid_df.empty:

        for _, row in (
            invalid_df.iterrows()
        ):

            quality_issues.append({

                "Issue_Type":
                    "Invalid Value",

                "Column":
                    row["Column"],

                "Count":
                    row["Count"],

                "Severity":
                    row["Severity"]

            })

    if not city_df.empty:

        for _, row in (
            city_df.iterrows()
        ):

            quality_issues.append({

                "Issue_Type":
                    "City Standardization",

                "Column":
                    "City",

                "Count":
                    row["Count"],

                "Severity":
                    "Low"

            })

    if quality_issues:

        quality_issues_df = pd.DataFrame(
            quality_issues
        )

    else:

        quality_issues_df = pd.DataFrame(

            columns=[
                "Issue_Type",
                "Column",
                "Count",
                "Severity"
            ]

        )

    # --------------------------------------------------------
    # ANOMALY RESULTS
    # --------------------------------------------------------

    anomaly_results = ml_df.copy()

    (
        _,
        original_statistical_details
    ) = detect_all_statistical_outliers(
        original_df
    )

    for column, result in (
        original_statistical_details.items()
    ):

        flag_column = (
            f"{column}_IQR_Outlier"
        )

        numeric_values = pd.to_numeric(
            anomaly_results[column],
            errors="coerce"
        )

        anomaly_results[
            flag_column
        ] = (

            numeric_values.lt(
                result["Lower_Bound"]
            )

            |

            numeric_values.gt(
                result["Upper_Bound"]
            )

        )

    # --------------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------------

    files = {

        "quality_summary.csv":
            quality_summary,

        "quality_issues.csv":
            quality_issues_df,

        "anomaly_results.csv":
            anomaly_results,

        "cleaning_log.csv":
            cleaning_log,

        "cleaned_data.csv":
            cleaned_df

    }

    for filename, dataframe in (
        files.items()
    ):

        dataframe.to_csv(
            POWERBI_FOLDER / filename,
            index=False
        )

    # --------------------------------------------------------
    # CREATE ZIP
    # --------------------------------------------------------

    zip_path = (
        Path.cwd()
        /
        "DataGuard_AI_PowerBI_Export.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for filename in files:

            file_path = (
                POWERBI_FOLDER
                /
                filename
            )

            zip_file.write(
                file_path,
                arcname=filename
            )

    return (
        zip_path,
        files
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(

    "Upload your CSV or Excel file",

    type=[
        "csv",
        "xlsx",
        "xls"
    ]

)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    df_original = load_data(
        uploaded_file
    )

    if df_original is None:
        st.stop()

    if df_original.empty:

        st.error(
            "The uploaded dataset is empty."
        )

        st.stop()

    df = df_original.copy()

    st.success(
        f"Successfully uploaded: "
        f"{uploaded_file.name}"
    )


    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    st.header(
        "1️⃣ Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # ========================================================
    # BASIC PROFILE
    # ========================================================

    st.header(
        "2️⃣ Dataset Profile"
    )

    (
        numeric_columns,
        categorical_columns,
        datetime_columns,
        identifier_columns
    ) = classify_columns(
        df
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
            len(df.columns)
        )

    with col3:

        st.metric(
            "Missing Cells",
            f"{int(df.isnull().sum().sum()):,}"
        )

    with col4:

        st.metric(
            "Duplicates",
            f"{int(df.duplicated().sum()):,}"
        )

    st.subheader(
        "Column Classification"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            "**Numeric Columns**"
        )

        if numeric_columns:

            st.write(
                numeric_columns
            )

        else:

            st.write("None")

        st.write(
            "**Categorical Columns**"
        )

        if categorical_columns:

            st.write(
                categorical_columns
            )

        else:

            st.write("None")

    with c2:

        st.write(
            "**Date/Time Columns**"
        )

        if datetime_columns:

            st.success(
                ", ".join(
                    map(
                        str,
                        datetime_columns
                    )
                )
            )

        else:

            st.write("None")

        st.write(
            "**Identifier Columns**"
        )

        if identifier_columns:

            st.write(
                identifier_columns
            )

        else:

            st.write("None")


    # ========================================================
    # QUALITY ISSUES
    # ========================================================

    st.header(
        "3️⃣ Data Quality Checks"
    )

    missing_df = check_missing_values(
        df
    )

    duplicate_count = check_duplicates(
        df
    )

    invalid_df = check_invalid_values(
        df
    )

    city_df = find_city_inconsistencies(
        df
    )

    missing_count = int(
        df.isnull()
        .sum()
        .sum()
    )

    invalid_count = (

        int(
            invalid_df["Count"].sum()
        )

        if not invalid_df.empty

        else 0

    )

    city_count = (

        int(
            city_df["Count"].sum()
        )

        if not city_df.empty

        else 0

    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.metric(
            "Missing Cells",
            f"{missing_count:,}"
        )

    with q2:

        st.metric(
            "Duplicates",
            f"{duplicate_count:,}"
        )

    with q3:

        st.metric(
            "Invalid Values",
            f"{invalid_count:,}"
        )

    with q4:

        st.metric(
            "City Records to Standardize",
            f"{city_count:,}"
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    st.subheader(
        "Missing Values"
    )

    if missing_df.empty:

        st.success(
            "No missing values detected."
        )

    else:

        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    st.subheader(
        "Duplicate Records"
    )

    if duplicate_count == 0:

        st.success(
            "No duplicate records detected."
        )

    else:

        st.warning(
            f"{duplicate_count:,} "
            "duplicate records detected."
        )

    # --------------------------------------------------------
    # Invalid
    # --------------------------------------------------------

    st.subheader(
        "Invalid Values"
    )

    if invalid_df.empty:

        st.success(
            "No rule-based invalid values detected."
        )

    else:

        st.dataframe(
            invalid_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # City
    # --------------------------------------------------------

    st.subheader(
        "City Consistency"
    )

    if city_df.empty:

        st.success(
            "No city inconsistencies detected."
        )

    else:

        st.dataframe(
            city_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # STATISTICAL OUTLIERS
    # ========================================================

    st.header(
        "4️⃣ Statistical Outlier Detection"
    )

    st.info(
        """
**IQR outliers** identify observations that are statistically unusual.
They are **not automatically errors** and should be investigated before
removing or modifying them.
"""
    )

    (
        statistical_df,
        statistical_details
    ) = detect_all_statistical_outliers(
        df
    )

    if statistical_df.empty:

        st.warning(
            "No numeric columns were available for IQR analysis."
        )

    else:

        total_statistical_outliers = int(

            statistical_df[
                "Outlier_Count"
            ].sum()

        )

        st.metric(
            "Total Statistical Outlier Flags",
            f"{total_statistical_outliers:,}"
        )

        st.dataframe(
            statistical_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Outlier Details by Column"
        )

        for column, result in (
            statistical_details.items()
        ):

            st.markdown(
                f"#### {column}"
            )

            st.write(
                f"**Outliers detected:** "
                f"{result['Count']:,}"
            )

            if result["Count"] > 0:

                st.write(

                    f"Q1 = {result['Q1']:.4f} | "
                    f"Q3 = {result['Q3']:.4f} | "
                    f"IQR = {result['IQR']:.4f}"

                )

                st.write(

                    f"Lower Bound = "
                    f"{result['Lower_Bound']:.4f} | "

                    f"Upper Bound = "
                    f"{result['Upper_Bound']:.4f}"

                )

                st.dataframe(

                    result["outliers"].head(20),

                    use_container_width=True,

                    hide_index=True

                )

            else:

                st.success(
                    "No IQR outliers detected."
                )


    # ========================================================
    # ML ANOMALIES
    # ========================================================

    st.header(
        "5️⃣ Machine Learning Anomaly Detection"
    )

    st.markdown(
        f"""
<div class="section-note">

<strong>Isolation Forest screening sensitivity:</strong>
{ML_CONTAMINATION * 100:.0f}%

The model searches for unusual multivariate patterns.
A flagged record is a candidate for investigation — not
automatically a data error, fraud case, or invalid transaction.

</div>
""",
        unsafe_allow_html=True
    )

    (
        ml_df,
        anomaly_features,
        ml_anomaly_count
    ) = detect_ml_anomalies(
        df
    )

    if anomaly_features:

        st.write(
            "**ML Features:**",
            ", ".join(
                anomaly_features
            )
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "ML Anomalies",
                f"{ml_anomaly_count:,}"
            )

        with m2:

            st.metric(
                "Normal Records",
                f"{len(df) - ml_anomaly_count:,}"
            )

        with m3:

            actual_rate = (
                ml_anomaly_count
                /
                len(df)
                *
                100
            )

            st.metric(
                "Flagged Rate",
                f"{actual_rate:.2f}%"
            )

        anomaly_display = (
            ml_df[
                ml_df[
                    "ML_Anomaly"
                ] == -1
            ]
            .copy()
        )

        if not anomaly_display.empty:

            st.subheader(
                "Detected ML Anomalies"
            )

            st.dataframe(
                anomaly_display.head(100),
                use_container_width=True,
                hide_index=True
            )

    else:

        st.warning(
            "Not enough numeric features "
            "for ML anomaly detection."
        )


    # ========================================================
    # QUALITY SCORE
    # ========================================================

    st.header(
        "6️⃣ Overall Data Quality Score"
    )

    quality_score = calculate_quality_score(

        df,

        missing_count,

        duplicate_count,

        invalid_count,

        city_count

    )

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        st.metric(
            "Data Quality Score",
            f"{quality_score}/100"
        )

    with score_col2:

        st.metric(
            "Status",
            quality_label(
                quality_score
            )
        )

    st.caption(
        """
The score considers confirmed data-quality issues:
missing values, duplicate records, rule-based invalid values,
and city-standardization issues.

Statistical outliers and ML anomalies are reported separately
because unusual observations are not automatically errors.
"""
    )


    # ========================================================
    # ROOT CAUSE ANALYSIS
    # ========================================================

    st.header(
        "7️⃣ Root-Cause Analysis"
    )

    total_statistical_outliers = (

        int(
            statistical_df[
                "Outlier_Count"
            ].sum()
        )

        if not statistical_df.empty

        else 0

    )

    rca_df = rule_based_root_cause(

        missing_count,

        duplicate_count,

        invalid_count,

        city_count,

        total_statistical_outliers,

        ml_anomaly_count

    )

    if rca_df.empty:

        st.success(
            "No major rule-based quality issues detected."
        )

    else:

        st.dataframe(
            rca_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # GEMINI AI
    # ========================================================

    st.header(
        "8️⃣ Gemini AI Analysis"
    )

    issue_summary = f"""

Rows: {len(df)}
Columns: {len(df.columns)}

Missing cells:
{missing_count}

Duplicate records:
{duplicate_count}

Invalid values:
{invalid_count}

City records to standardize:
{city_count}

Statistical outlier flags:
{statistical_df.to_string(index=False)}

ML features:
{", ".join(anomaly_features)}

Isolation Forest configured contamination:
{ML_CONTAMINATION * 100:.0f}%

ML anomalies detected:
{ml_anomaly_count}

Actual ML anomaly rate:
{(ml_anomaly_count / len(df) * 100):.2f}%

Data quality score:
{quality_score}/100
"""

    if st.button(
        "🤖 Analyze Root Causes with Gemini AI",
        type="primary"
    ):

        with st.spinner(
            "Gemini is analyzing the dataset..."
        ):

            gemini_result = ask_gemini(
                issue_summary
            )

        st.markdown(
            gemini_result
        )

        st.download_button(

            label=
                "⬇️ Download AI Analysis",

            data=
                gemini_result,

            file_name=
                "DataGuard_AI_Root_Cause_Analysis.txt",

            mime=
                "text/plain"

        )


    # ========================================================
    # DATA CLEANING
    # ========================================================

    st.header(
        "9️⃣ Automated Data Cleaning"
    )

    (
        cleaned_df,
        cleaning_log_df,
        total_filled
    ) = clean_data(

        df,

        identifier_columns

    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Original Rows",
            f"{len(df):,}"
        )

    with c2:

        st.metric(
            "Cleaned Rows",
            f"{len(cleaned_df):,}"
        )

    with c3:

        st.metric(
            "Rows Removed",
            f"{len(df) - len(cleaned_df):,}"
        )

    with c4:

        st.metric(
            "Values Filled",
            f"{total_filled:,}"
        )

    st.subheader(
        "Cleaning Actions"
    )

    st.dataframe(
        cleaning_log_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Cleaned Dataset Preview"
    )

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # CLEANED CSV
    # --------------------------------------------------------

    cleaned_csv = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    original_name = Path(
        uploaded_file.name
    ).stem

    cleaned_filename = (
        f"{original_name}_cleaned.csv"
    )

    st.download_button(

        label=
            "⬇️ Download Cleaned CSV",

        data=
            cleaned_csv,

        file_name=
            cleaned_filename,

        mime=
            "text/csv"

    )


    # ========================================================
    # POWER BI EXPORT
    # ========================================================

    st.header(
        "🔟 Power BI Export"
    )

    st.info(
        """
DataGuard AI exports Power BI-ready CSV files.

The actual `.pbix` dashboard should be created in Power BI Desktop
using these exported files.
"""
    )

    if st.button(
        "📊 Create Power BI Export"
    ):

        with st.spinner(
            "Creating Power BI export files..."
        ):

            (
                zip_path,
                exported_files
            ) = create_powerbi_export(

                original_df=df,

                cleaned_df=cleaned_df,

                missing_df=missing_df,

                invalid_df=invalid_df,

                city_df=city_df,

                statistical_df=statistical_df,

                ml_df=ml_df,

                cleaning_log=cleaning_log_df,

                quality_score=quality_score

            )

        st.success(
            "Power BI export created successfully!"
        )

        st.write(
            "**Files created:**"
        )

        for filename in exported_files:

            st.write(
                f"✅ {filename}"
            )

        with open(
            zip_path,
            "rb"
        ) as file:

            st.download_button(

                label=
                    "⬇️ Download Power BI Export ZIP",

                data=
                    file.read(),

                file_name=
                    "DataGuard_AI_PowerBI_Export.zip",

                mime=
                    "application/zip"

            )


    # ========================================================
    # DATA QUALITY REPORT
    # ========================================================

    st.header(
        "1️⃣1️⃣ Data Quality Report"
    )

    report_text = generate_report(

        df=df,

        duplicate_count=
            duplicate_count,

        invalid_df=
            invalid_df,

        city_df=
            city_df,

        statistical_df=
            statistical_df,

        ml_anomaly_count=
            ml_anomaly_count,

        quality_score=
            quality_score,

        cleaned_df=
            cleaned_df

    )

    st.text_area(

        "Report Preview",

        report_text,

        height=450

    )

    st.download_button(

        label=
            "⬇️ Download Data Quality Report",

        data=
            report_text,

        file_name=
            "DataGuard_AI_Data_Quality_Report.txt",

        mime=
            "text/plain"

    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.header(
        "📋 Final Summary"
    )

    summary_data = pd.DataFrame({

        "Metric": [

            "Original Rows",

            "Original Columns",

            "Missing Cells",

            "Duplicate Records",

            "Invalid Values",

            "City Records to Standardize",

            "Statistical Outlier Flags",

            "ML Anomalies",

            "Cleaned Rows",

            "Rows Removed",

            "Values Filled",

            "Quality Score"

        ],

        "Result": [

            f"{len(df):,}",

            f"{len(df.columns):,}",

            f"{missing_count:,}",

            f"{duplicate_count:,}",

            f"{invalid_count:,}",

            f"{city_count:,}",

            f"{total_statistical_outliers:,}",

            f"{ml_anomaly_count:,}",

            f"{len(cleaned_df):,}",

            f"{len(df) - len(cleaned_df):,}",

            f"{total_filled:,}",

            f"{quality_score}/100"

        ]

    })

    st.dataframe(

        summary_data,

        use_container_width=True,

        hide_index=True

    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown("---")

    st.markdown(

        '<div class="footer">'
        'DataGuard AI | Data Quality • Statistical Analysis • '
        'Machine Learning • Root-Cause Analysis • Power BI'
        '</div>',

        unsafe_allow_html=True

    )


else:

    st.info(
        "👆 Upload a CSV or Excel dataset to start "
        "the DataGuard AI pipeline."
    )
    
