import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DataGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #0b1020;
        color: #e8ecf5;
    }

    [data-testid="stSidebar"] {
        background: #0f1629;
        border-right: 1px solid #202a42;
    }

    [data-testid="stSidebar"] * {
        color: #e8ecf5;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #f4f7fb !important;
    }

    p, span, label {
        color: #aeb8cc;
    }

    /* ---------- HEADER ---------- */

    .topbar {
        padding: 18px 22px;
        border: 1px solid #202a42;
        background: #11192d;
        border-radius: 16px;
        margin-bottom: 22px;
    }

    .brand-title {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
    }

    .brand-subtitle {
        font-size: 13px;
        color: #8995ad;
        margin-top: 3px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: #11192d;
        border: 1px solid #202a42;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .card-title {
        color: #f4f7fb;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .card-subtitle {
        color: #8995ad;
        font-size: 12px;
    }

    .metric-card {
        background: #11192d;
        border: 1px solid #202a42;
        border-radius: 16px;
        padding: 18px;
        min-height: 125px;
    }

    .metric-label {
        color: #8e9ab2;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 27px;
        font-weight: 800;
    }

    .metric-help {
        color: #64718a;
        font-size: 11px;
        margin-top: 6px;
    }

    /* ---------- STATUS ---------- */

    .status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #10271f;
        border: 1px solid #1e563f;
        border-radius: 20px;
        padding: 6px 12px;
        font-size: 12px;
        color: #70e0a6;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #3ddc84;
    }

    /* ---------- SCORE ---------- */

    .score-box {
        text-align: center;
        background: #11192d;
        border: 1px solid #202a42;
        border-radius: 18px;
        padding: 30px 20px;
    }

    .score-number {
        font-size: 54px;
        line-height: 1;
        font-weight: 900;
        color: #ffffff;
    }

    .score-label {
        margin-top: 10px;
        color: #8e9ab2;
        font-size: 13px;
    }

    /* ---------- WORKFLOW ---------- */

    .workflow {
        display: flex;
        align-items: center;
        gap: 8px;
        overflow-x: auto;
        padding: 4px 0 18px 0;
    }

    .workflow-step {
        background: #151e34;
        border: 1px solid #26324e;
        border-radius: 10px;
        padding: 9px 13px;
        white-space: nowrap;
        color: #78859d;
        font-size: 12px;
    }

    .workflow-step.active {
        background: #172d50;
        border-color: #3d78d8;
        color: #ffffff;
    }

    .workflow-step.done {
        background: #12251f;
        border-color: #255a43;
        color: #7ee2ab;
    }

    .workflow-arrow {
        color: #46526b;
    }

    /* ---------- ISSUE ---------- */

    .issue {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 13px 15px;
        border-bottom: 1px solid #202a42;
    }

    .issue:last-child {
        border-bottom: none;
    }

    .issue-name {
        color: #dfe5f0;
        font-size: 13px;
    }

    .issue-value {
        color: #ffffff;
        font-weight: 700;
    }

    /* ---------- UPLOAD ---------- */

    .upload-box {
        border: 1px dashed #3a4868;
        border-radius: 18px;
        background: #10182b;
        padding: 35px;
        text-align: center;
        margin-bottom: 20px;
    }

    .upload-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }

    /* ---------- BADGES ---------- */

    .badge-success {
        background: #12251f;
        border: 1px solid #255a43;
        color: #7ee2ab;
        padding: 4px 9px;
        border-radius: 12px;
        font-size: 11px;
    }

    .badge-warning {
        background: #2b2414;
        border: 1px solid #66501e;
        color: #e7c56a;
        padding: 4px 9px;
        border-radius: 12px;
        font-size: 11px;
    }

    .badge-danger {
        background: #301b21;
        border: 1px solid #69323e;
        color: #f18b9b;
        padding: 4px 9px;
        border-radius: 12px;
        font-size: 11px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #56627a;
        font-size: 11px;
        padding: 30px 0 10px;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .metric-card {
            margin-bottom: 10px;
        }

        .workflow {
            padding-bottom: 10px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "df": None,
    "cleaned_df": None,
    "business_df": None,
    "chart_df": None,
    "file_name": None,
    "analysis_complete": False,
    "analysis": None,
    "gemini_analysis": None,
    "contamination_pct": 5,
    "page": "Dashboard",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CONSTANTS
# ============================================================

CITY_MAPPING = {
    "blr": "Bengaluru",
    "blr.": "Bengaluru",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bengalooru": "Bengaluru",
    "madras": "Chennai",
    "chennai": "Chennai",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "hyderabad": "Hyderabad",
    "delhi": "Delhi",
}


# ============================================================
# HELPERS
# ============================================================

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=series.index)


def detect_datetime_columns(df):
    result = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(col)
            continue

        name = str(col).lower()

        if any(
            token in name
            for token in [
                "date",
                "time",
                "timestamp",
                "datetime",
            ]
        ):
            parsed = safe_to_datetime(df[col])

            if parsed.notna().mean() >= 0.60:
                result.append(col)

    return result


def detect_identifier_columns(df):

    result = []

    for col in df.columns:

        name = str(col).lower()

        unique_ratio = (
            df[col].nunique(dropna=True) / max(len(df), 1)
        )

        if (
            any(
                token in name
                for token in [
                    "id",
                    "code",
                    "identifier",
                ]
            )
            and unique_ratio >= 0.50
        ):
            result.append(col)

    return result


def profile_dataset(df):

    rows = []

    for col in df.columns:

        non_null = int(df[col].notna().sum())
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))

        non_null_values = df[col].dropna()

        if len(non_null_values) > 0:
            sample = str(non_null_values.iloc[0])
        else:
            sample = "—"

        rows.append(
            {
                "Column": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": non_null,
                "Missing": missing,
                "Missing %": round(
                    missing / max(len(df), 1) * 100,
                    2,
                ),
                "Unique Values": unique,
                "Sample Value": sample,
            }
        )

    return pd.DataFrame(rows)


def detect_invalid_values(df):

    invalid_count = 0
    invalid_by_column = {}

    keywords = [
        "age",
        "quantity",
        "sales",
        "revenue",
        "amount",
        "price",
        "profit",
    ]

    for col in df.columns:

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):

            negative_count = int(
                (series < 0).sum()
            )

            if negative_count > 0:
                invalid_by_column[col] = negative_count
                invalid_count += negative_count

        else:

            name = str(col).lower()

            if any(k in name for k in keywords):

                converted = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                invalid = (
                    series.notna()
                    & converted.isna()
                )

                count = int(invalid.sum())

                if count > 0:
                    invalid_by_column[col] = count
                    invalid_count += count

    return invalid_count, invalid_by_column


def standardize_city_column(df):

    result = df.copy()

    city_col = None

    for col in result.columns:
        if str(col).lower() == "city":
            city_col = col
            break

    if city_col is None:
        return result, 0

    original = result[city_col].copy()

    def normalize(value):

        if pd.isna(value):
            return value

        text = str(value).strip().lower()

        if text in CITY_MAPPING:
            return CITY_MAPPING[text]

        return text.title()

    result[city_col] = result[city_col].apply(normalize)

    changes = int(
        (
            original.fillna("__NA__").astype(str)
            != result[city_col].fillna("__NA__").astype(str)
        ).sum()
    )

    return result, changes


def detect_iqr_outliers(df):

    total = 0
    details = {}

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (series < lower)
                | (series > upper)
            ).sum()
        )

        if count > 0:
            details[col] = {
                "count": count,
                "lower": float(lower),
                "upper": float(upper),
            }

            total += count

    return total, details


def detect_ml_anomalies(df, contamination_pct):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty or len(numeric) < 20:
        return 0, len(df), []

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric = numeric.fillna(
        numeric.median(numeric_only=True)
    )

    numeric = numeric.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return 0, len(df), []

    contamination = min(
        max(contamination_pct / 100, 0.01),
        0.49,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    prediction = model.fit_predict(numeric)

    anomalies = prediction == -1

    anomaly_count = int(anomalies.sum())

    return (
        anomaly_count,
        len(df) - anomaly_count,
        prediction,
    )


def build_quality_score(
    df,
    missing,
    duplicates,
    invalid,
):

    rows = max(len(df), 1)

    missing_rate = missing / (
        rows * max(len(df.columns), 1)
    )

    duplicate_rate = duplicates / rows

    invalid_rate = invalid / rows

    score = 100

    score -= missing_rate * 50
    score -= duplicate_rate * 30
    score -= invalid_rate * 20

    return max(0, min(100, score))


def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    # Remove duplicates
    before = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    rows_removed = before - len(cleaned)

    # Standardize city
    cleaned, city_changes = standardize_city_column(
        cleaned
    )

    # Numeric conversion and filling
    values_filled = 0

    numeric_cols = cleaned.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        missing_before = int(
            cleaned[col].isna().sum()
        )

        if missing_before > 0:

            median_value = cleaned[col].median()

            if pd.notna(median_value):

                cleaned[col] = cleaned[col].fillna(
                    median_value
                )

                values_filled += missing_before

    # Categorical filling
    categorical_cols = cleaned.select_dtypes(
        exclude=np.number
    ).columns

    for col in categorical_cols:

        missing_before = int(
            cleaned[col].isna().sum()
        )

        if missing_before == 0:
            continue

        mode = cleaned[col].mode(dropna=True)

        if not mode.empty:

            cleaned[col] = cleaned[col].fillna(
                mode.iloc[0]
            )

            values_filled += missing_before

    return (
        cleaned,
        original_rows,
        rows_removed,
        values_filled,
        city_changes,
    )


def find_sales_column(df):

    preferred = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "sale",
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    for col in df.columns:

        name = str(col).lower()

        if (
            "sales" in name
            or "revenue" in name
            or "amount" in name
        ):
            return col

    return None


def prepare_business_data(df):

    sales_col = find_sales_column(df)

    if sales_col is None:
        return (
            None,
            None,
            None,
        )

    business = df.copy()

    business[sales_col] = pd.to_numeric(
        business[sales_col],
        errors="coerce",
    )

    business = business[
        business[sales_col].notna()
    ].copy()

    business = business[
        business[sales_col] >= 0
    ].copy()

    chart_df = business.copy()

    # IQR upper bound for visual scaling
    sales = chart_df[sales_col].dropna()

    upper_bound = None

    if len(sales) >= 4:

        q1 = sales.quantile(0.25)
        q3 = sales.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            upper_bound = float(
                q3 + 1.5 * iqr
            )

            chart_df = chart_df[
                chart_df[sales_col]
                <= upper_bound
            ].copy()

    total_sales = float(
        business[sales_col].sum()
    )

    average_sale = float(
        business[sales_col].mean()
    )

    highest_sale = float(
        business[sales_col].max()
    )

    return (
        business,
        chart_df,
        {
            "sales_column": sales_col,
            "total_sales": total_sales,
            "average_sale": average_sale,
            "highest_sale": highest_sale,
            "business_upper_bound": upper_bound,
        },
    )


def create_powerbi_exports(
    cleaned_df,
    profile_df,
    business_df,
):

    exports = {}

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    exports[
        "DataGuard_Data_Profile.csv"
    ] = profile_df.to_csv(
        index=False
    ).encode("utf-8")

    if business_df is not None:

        sales_col = find_sales_column(
            business_df
        )

        if sales_col:

            sales_summary = pd.DataFrame(
                {
                    "Metric": [
                        "Total Sales",
                        "Average Sale",
                        "Highest Sale",
                        "Sales Records",
                    ],
                    "Value": [
                        business_df[sales_col].sum(),
                        business_df[sales_col].mean(),
                        business_df[sales_col].max(),
                        len(business_df),
                    ],
                }
            )

            exports[
                "DataGuard_Sales_Summary.csv"
            ] = sales_summary.to_csv(
                index=False
            ).encode("utf-8")

            if "City" in business_df.columns:

                city_summary = (
                    business_df
                    .groupby("City", dropna=False)[
                        sales_col
                    ]
                    .sum()
                    .reset_index()
                    .sort_values(
                        sales_col,
                        ascending=False,
                    )
                )

                exports[
                    "DataGuard_City_Summary.csv"
                ] = city_summary.to_csv(
                    index=False
                ).encode("utf-8")

            if "Category" in business_df.columns:

                category_summary = (
                    business_df
                    .groupby(
                        "Category",
                        dropna=False,
                    )[sales_col]
                    .sum()
                    .reset_index()
                    .sort_values(
                        sales_col,
                        ascending=False,
                    )
                )

                exports[
                    "DataGuard_Category_Summary.csv"
                ] = category_summary.to_csv(
                    index=False
                ).encode("utf-8")

    return exports


def create_zip(files):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        for name, data in files.items():
            z.writestr(name, data)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

def run_gemini(analysis):

    try:

        from google import genai

        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )

        if not api_key:
            return (
                "Gemini is not configured. "
                "Please add GEMINI_API_KEY to Streamlit secrets."
            )

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are the AI analyst inside DataGuard AI.

IMPORTANT RULE:
Use ONLY the measured values supplied below.

DO NOT:
- recalculate numbers
- estimate numbers
- invent numbers
- change numbers
- create alternative values
- infer missing numeric values

If a metric is "Not available", write "Not available".

DATASET
-------
Dataset Name: {analysis['file_name']}
Rows: {analysis['rows']}
Columns: {analysis['columns']}

QUALITY
-------
Quality Score: {analysis['quality_score']:.2f}
Missing Values: {analysis['missing_values']}
Duplicate Rows: {analysis['duplicates']}
Invalid Values: {analysis['invalid_values']}

OUTLIERS
--------
IQR Outliers: {analysis['iqr_outliers']}

ANOMALIES
---------
ML Anomalies: {analysis['ml_anomalies']}
Normal Records: {analysis['normal_records']}
Isolation Forest Sensitivity: {analysis['contamination_pct']}%

CLEANING
--------
Rows Removed: {analysis['rows_removed']}
Values Filled: {analysis['values_filled']}
City Standardization Changes: {analysis['city_standardized']}

BUSINESS
--------
Sales Column: {analysis['sales_column']}
Total Sales: {analysis['total_sales']}
Average Sale: {analysis['average_sale']}
Highest Sale: {analysis['highest_sale']}
Business Visualization Upper Bound:
{analysis['business_upper_bound']}

Write a concise professional report with exactly these sections:

### 1. Overall Assessment
### 2. Confirmed Data Quality Problems
### 3. Statistical Outlier Findings
### 4. ML Anomaly Findings
### 5. Possible Root Causes
### 6. Cleaning Results
### 7. Business Impact
### 8. Power BI Recommendations

Rules:
- Preserve all numeric values exactly.
- Do not recalculate the business upper bound.
- ML anomalies are screening signals, not confirmed errors.
- IQR outliers are statistical signals, not automatically errors.
- Do not recommend deleting anomalies automatically.
- Keep the tone professional and concise.
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
        )

        if hasattr(response, "output_text"):

            text = response.output_text

            if text:
                return text

        return (
            "Gemini completed the analysis but "
            "did not return readable text."
        )

    except Exception as e:

        return (
            f"Gemini analysis failed: {str(e)}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:8px 0 20px;">
            <div style="font-size:34px;">🛡️</div>
            <div style="font-size:22px;font-weight:800;color:white;">
                DataGuard AI
            </div>
            <div style="font-size:12px;color:#8995ad;">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "**WORKSPACE**"
    )

    pages = [
        "Dashboard",
        "Upload Data",
        "Data Profile",
        "Quality Checks",
        "Anomaly Detection",
        "Data Cleaning",
        "Analytics",
        "Power BI",
        "AI Analysis",
        "Reports",
    ]

    selected_page = st.radio(
        "Navigation",
        pages,
        index=pages.index(
            st.session_state.page
        ),
        label_visibility="collapsed",
    )

    st.session_state.page = selected_page

    st.divider()

    st.markdown(
        "**DETECTION SETTINGS**"
    )

    sensitivity = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
    )

    st.session_state.contamination_pct = sensitivity

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.divider()

    st.markdown(
        "**SYSTEM**"
    )

    st.markdown(
        """
        <div class="status">
            <div class="status-dot"></div>
            All systems operational
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP HEADER
# ============================================================

file_label = (
    st.session_state.file_name
    if st.session_state.file_name
    else "No dataset loaded"
)

st.markdown(
    f"""
    <div class="topbar">
        <div class="brand-title">
            Data Intelligence Workspace
        </div>
        <div class="brand-subtitle">
            AI-powered data quality, anomaly detection and business analytics.
        </div>
        <div style="
            margin-top:12px;
            font-size:13px;
            color:#c3cbda;
        ">
            📄 <strong>{file_label}</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = None

if st.session_state.page == "Upload Data":

    st.markdown(
        """
        <div class="upload-box">
            <div class="upload-icon">📂</div>
            <div style="font-size:20px;font-weight:700;color:white;">
                Upload your dataset
            </div>
            <div style="margin-top:8px;color:#8995ad;">
                CSV, XLSX or XLS files • Maximum 50 MB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Browse Files",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(
                ".csv"
            ):

                df = pd.read_csv(
                    uploaded_file
                )

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.df = df
            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.chart_df = None
            st.session_state.analysis = None
            st.session_state.gemini_analysis = None
            st.session_state.analysis_complete = False

            st.success(
                f"Successfully loaded {uploaded_file.name}"
            )

        except Exception as e:

            st.error(
                f"Could not read the file: {e}"
            )


# ============================================================
# AUTO LOAD / ANALYSIS
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if (
        not st.session_state.analysis_complete
        or st.session_state.analysis is None
    ):

        # -----------------------------
        # Profile
        # -----------------------------

        profile_df = profile_dataset(df)

        # -----------------------------
        # Quality
        # -----------------------------

        missing_values = int(
            df.isna().sum().sum()
        )

        duplicates = int(
            df.duplicated().sum()
        )

        invalid_values, invalid_details = (
            detect_invalid_values(df)
        )

        quality_score = build_quality_score(
            df,
            missing_values,
            duplicates,
            invalid_values,
        )

        # -----------------------------
        # IQR
        # -----------------------------

        iqr_outliers, iqr_details = (
            detect_iqr_outliers(df)
        )

        # -----------------------------
        # ML
        # -----------------------------

        (
            ml_anomalies,
            normal_records,
            ml_prediction,
        ) = detect_ml_anomalies(
            df,
            st.session_state.contamination_pct,
        )

        # -----------------------------
        # Cleaning
        # -----------------------------

        (
            cleaned_df,
            original_rows,
            rows_removed,
            values_filled,
            city_standardized,
        ) = clean_dataset(df)

        # -----------------------------
        # Business
        # -----------------------------

        (
            business_df,
            chart_df,
            business_metrics,
        ) = prepare_business_data(
            cleaned_df
        )

        sales_column = (
            business_metrics["sales_column"]
            if business_metrics
            else "Not available"
        )

        total_sales = (
            business_metrics["total_sales"]
            if business_metrics
            else None
        )

        average_sale = (
            business_metrics["average_sale"]
            if business_metrics
            else None
        )

        highest_sale = (
            business_metrics["highest_sale"]
            if business_metrics
            else None
        )

        business_upper_bound = (
            business_metrics[
                "business_upper_bound"
            ]
            if business_metrics
            else None
        )

        # -----------------------------
        # Analysis dictionary
        # -----------------------------

        analysis = {

            "file_name": st.session_state.file_name,

            "rows": len(df),

            "columns": len(df.columns),

            "missing_values": missing_values,

            "duplicates": duplicates,

            "invalid_values": invalid_values,

            "quality_score": quality_score,

            "iqr_outliers": iqr_outliers,

            "ml_anomalies": ml_anomalies,

            "normal_records": normal_records,

            "contamination_pct":
                st.session_state.contamination_pct,

            "rows_removed": rows_removed,

            "values_filled": values_filled,

            "city_standardized":
                city_standardized,

            "numeric_columns":
                len(
                    df.select_dtypes(
                        include=np.number
                    ).columns
                ),

            "categorical_columns":
                len(
                    df.select_dtypes(
                        exclude=np.number
                    ).columns
                ),

            "datetime_columns":
                len(
                    detect_datetime_columns(df)
                ),

            "identifier_columns":
                len(
                    detect_identifier_columns(df)
                ),

            "sales_column":
                sales_column,

            "total_sales":
                total_sales,

            "average_sale":
                average_sale,

            "highest_sale":
                highest_sale,

            "business_upper_bound":
                business_upper_bound,

            "profile_df":
                profile_df,

            "iqr_details":
                iqr_details,

            "invalid_details":
                invalid_details,
        }

        st.session_state.cleaned_df = (
            cleaned_df
        )

        st.session_state.business_df = (
            business_df
        )

        st.session_state.chart_df = (
            chart_df
        )

        st.session_state.analysis = (
            analysis
        )

        st.session_state.analysis_complete = True


# ============================================================
# WORKFLOW
# ============================================================

if st.session_state.analysis is not None:

    workflow = [
        ("01", "Upload"),
        ("02", "Profile"),
        ("03", "Quality"),
        ("04", "Anomalies"),
        ("05", "Clean"),
        ("06", "Analytics"),
        ("07", "Power BI"),
        ("08", "AI"),
    ]

    current = st.session_state.page

    page_map = {
        "Dashboard": 3,
        "Upload Data": 0,
        "Data Profile": 1,
        "Quality Checks": 2,
        "Anomaly Detection": 3,
        "Data Cleaning": 4,
        "Analytics": 5,
        "Power BI": 6,
        "AI Analysis": 7,
        "Reports": 7,
    }

    current_index = page_map.get(
        current,
        0,
    )

    workflow_html = '<div class="workflow">'

    for i, (number, label) in enumerate(
        workflow
    ):

        if i < current_index:
            cls = "workflow-step done"
            text = f"✓ {number} {label}"

        elif i == current_index:
            cls = "workflow-step active"
            text = f"{number} {label}"

        else:
            cls = "workflow-step"
            text = f"{number} {label}"

        workflow_html += (
            f'<div class="{cls}">{text}</div>'
        )

        if i < len(workflow) - 1:
            workflow_html += (
                '<div class="workflow-arrow">→</div>'
            )

    workflow_html += "</div>"

    st.markdown(
        workflow_html,
        unsafe_allow_html=True,
    )


# ============================================================
# NO DATA STATE
# ============================================================

if st.session_state.analysis is None:

    st.title("Welcome to DataGuard AI")

    st.write(
        "Upload a dataset to begin automated profiling, "
        "quality checks, anomaly detection and analytics."
    )

    if st.button(
        "Upload Dataset",
        type="primary",
    ):

        st.session_state.page = (
            "Upload Data"
        )

        st.rerun()


# ============================================================
# DATA AVAILABLE
# ============================================================

else:

    analysis = st.session_state.analysis

    cleaned_df = (
        st.session_state.cleaned_df
    )

    business_df = (
        st.session_state.business_df
    )

    chart_df = (
        st.session_state.chart_df
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    if st.session_state.page == "Dashboard":

        st.title("Data Quality Overview")

        st.caption(
            "Monitor, analyze, and improve the quality of your dataset."
        )

        cols = st.columns(6)

        metrics = [
            (
                "Data Quality",
                f"{analysis['quality_score']:.2f}%",
            ),
            (
                "Total Records",
                f"{analysis['rows']:,}",
            ),
            (
                "Columns",
                f"{analysis['columns']:,}",
            ),
            (
                "Missing Values",
                f"{analysis['missing_values']:,}",
            ),
            (
                "Duplicates",
                f"{analysis['duplicates']:,}",
            ),
            (
                "Anomalies",
                f"{analysis['ml_anomalies']:,}",
            ),
        ]

        for col, (label, value) in zip(
            cols,
            metrics,
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            {label}
                        </div>
                        <div class="metric-value">
                            {value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")

        left, right = st.columns(
            [1, 1.5]
        )

        with left:

            st.markdown(
                '<div class="score-box">',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="score-number">
                    {analysis['quality_score']:.1f}
                </div>
                <div class="score-label">
                    Overall Data Quality Score
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right:

            st.markdown(
                """
                <div class="card">
                    <div class="card-title">
                        Issues Detected
                    </div>
                    <div class="card-subtitle">
                        Measured signals from the uploaded dataset.
                    </div>
                """,
                unsafe_allow_html=True,
            )

            issues = [
                (
                    "Missing Values",
                    analysis["missing_values"],
                ),
                (
                    "Duplicate Records",
                    analysis["duplicates"],
                ),
                (
                    "Potential Outliers",
                    analysis["iqr_outliers"],
                ),
                (
                    "Invalid Values",
                    analysis["invalid_values"],
                ),
                (
                    "ML Anomalies",
                    analysis["ml_anomalies"],
                ),
            ]

            for name, value in issues:

                st.markdown(
                    f"""
                    <div class="issue">
                        <div class="issue-name">
                            {name}
                        </div>
                        <div class="issue-value">
                            {value:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.subheader(
            "Anomaly Detection"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Normal Records",
            f"{analysis['normal_records']:,}",
        )

        b.metric(
            "Anomalies",
            f"{analysis['ml_anomalies']:,}",
        )

        anomaly_rate = (
            analysis["ml_anomalies"]
            / max(analysis["rows"], 1)
            * 100
        )

        c.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.2f}%",
        )

        d.metric(
            "Algorithm",
            "Isolation Forest",
        )

        st.info(
            "ML anomalies are screening signals, not confirmed errors."
        )


    # ========================================================
    # UPLOAD PAGE
    # ========================================================

    elif st.session_state.page == "Upload Data":

        st.title("Upload Data")

        st.write(
            "Upload CSV or Excel data to begin the DataGuard workflow."
        )

        st.file_uploader(
            "Replace current dataset",
            type=[
                "csv",
                "xlsx",
                "xls",
            ],
            key="replacement_upload",
        )

        st.success(
            f"Current dataset: {st.session_state.file_name}"
        )


    # ========================================================
    # PROFILE
    # ========================================================

    elif st.session_state.page == "Data Profile":

        st.title("Data Profile")

        a, b, c, d = st.columns(4)

        a.metric(
            "Rows",
            f"{analysis['rows']:,}",
        )

        b.metric(
            "Columns",
            f"{analysis['columns']:,}",
        )

        c.metric(
            "Duplicate Rows",
            f"{analysis['duplicates']:,}",
        )

        d.metric(
            "Memory",
            f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        )

        st.subheader(
            "Column Profile"
        )

        profile_df = analysis[
            "profile_df"
        ].copy()

        search = st.text_input(
            "Search columns",
            placeholder="Search by column name...",
        )

        if search:

            profile_df = profile_df[
                profile_df["Column"]
                .astype(str)
                .str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # QUALITY
    # ========================================================

    elif st.session_state.page == "Quality Checks":

        st.title("Quality Checks")

        st.caption(
            "Automated checks across completeness, validity, consistency and uniqueness."
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Completeness",
            f"{max(0, 100 - analysis['missing_values'] / max(analysis['rows'] * analysis['columns'], 1) * 100):.2f}%",
        )

        q2.metric(
            "Duplicates",
            f"{analysis['duplicates']:,}",
        )

        q3.metric(
            "Invalid Values",
            f"{analysis['invalid_values']:,}",
        )

        q4.metric(
            "IQR Outliers",
            f"{analysis['iqr_outliers']:,}",
        )

        st.divider()

        quality_table = pd.DataFrame(
            {
                "Check": [
                    "Completeness",
                    "Duplicate Records",
                    "Invalid Values",
                    "Statistical Outliers",
                    "Data Types",
                ],
                "Result": [
                    analysis["missing_values"],
                    analysis["duplicates"],
                    analysis["invalid_values"],
                    analysis["iqr_outliers"],
                    "Detected",
                ],
            }
        )

        st.dataframe(
            quality_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif st.session_state.page == "Anomaly Detection":

        st.title("Anomaly Detection")

        st.caption(
            "Isolation Forest identifies unusual records across numeric features."
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Total Records",
            f"{analysis['rows']:,}",
        )

        b.metric(
            "Normal",
            f"{analysis['normal_records']:,}",
        )

        c.metric(
            "Anomalies",
            f"{analysis['ml_anomalies']:,}",
        )

        d.metric(
            "Sensitivity",
            f"{analysis['contamination_pct']}%",
        )

        st.divider()

        st.subheader(
            "Detection Settings"
        )

        st.slider(
            "Isolation Forest sensitivity",
            1,
            20,
            st.session_state.contamination_pct,
            key="anomaly_page_sensitivity",
        )

        st.info(
            "Isolation Forest anomalies are screening signals and should be reviewed before being treated as business errors."
        )

        numeric_cols = st.session_state.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        st.subheader(
            "Numeric Features"
        )

        st.write(
            ", ".join(numeric_cols)
            if numeric_cols
            else "No numeric features detected."
        )

        if numeric_cols:

            chart_col = st.selectbox(
                "Select feature",
                numeric_cols,
            )

            st.line_chart(
                st.session_state.df[
                    chart_col
                ].reset_index(drop=True),
                height=320,
            )


    # ========================================================
    # CLEANING
    # ========================================================

    elif st.session_state.page == "Data Cleaning":

        st.title("Data Cleaning")

        st.caption(
            "Automated cleaning based on measured data quality issues."
        )

        original_rows = analysis[
            "rows"
        ]

        cleaned_rows = len(
            cleaned_df
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Original Rows",
            f"{original_rows:,}",
        )

        c2.metric(
            "Cleaned Rows",
            f"{cleaned_rows:,}",
        )

        c3.metric(
            "Rows Removed",
            f"{analysis['rows_removed']:,}",
        )

        c4.metric(
            "Values Filled",
            f"{analysis['values_filled']:,}",
        )

        st.divider()

        st.subheader(
            "Cleaning Summary"
        )

        cleaning_summary = pd.DataFrame(
            {
                "Cleaning Action": [
                    "Duplicate Rows Removed",
                    "Missing Values Filled",
                    "City Values Standardized",
                ],
                "Records / Values": [
                    analysis["rows_removed"],
                    analysis["values_filled"],
                    analysis["city_standardized"],
                ],
                "Status": [
                    "Completed",
                    "Completed",
                    "Completed",
                ],
            }
        )

        st.dataframe(
            cleaning_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Cleaned Dataset Preview"
        )

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

        csv = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Cleaned Dataset",
            csv,
            file_name="DataGuard_Cleaned_Data.csv",
            mime="text/csv",
            type="primary",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif st.session_state.page == "Analytics":

        st.title("Business Analytics")

        st.caption(
            "Explore business metrics from the cleaned dataset."
        )

        if business_df is None:

            st.warning(
                "No recognizable Sales, Revenue or Amount column was found."
            )

        else:

            sales_col = analysis[
                "sales_column"
            ]

            a, b, c, d = st.columns(4)

            a.metric(
                "Total Sales",
                f"{analysis['total_sales']:,.2f}",
            )

            b.metric(
                "Average Sale",
                f"{analysis['average_sale']:,.2f}",
            )

            c.metric(
                "Highest Sale",
                f"{analysis['highest_sale']:,.2f}",
            )

            d.metric(
                "Sales Records",
                f"{len(business_df):,}",
            )

            st.divider()

            # Sales distribution
            st.subheader(
                "Sales Distribution"
            )

            st.bar_chart(
                chart_df[
                    sales_col
                ].value_counts(
                    bins=10
                ).sort_index(),
                height=350,
            )

            # Category
            if "Category" in business_df.columns:

                st.subheader(
                    "Sales by Category"
                )

                category_summary = (
                    business_df
                    .groupby(
                        "Category"
                    )[sales_col]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    category_summary,
                    height=350,
                )

            # City
            if "City" in business_df.columns:

                st.subheader(
                    "Sales by City"
                )

                city_summary = (
                    business_df
                    .groupby(
                        "City"
                    )[sales_col]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(15)
                )

                st.bar_chart(
                    city_summary,
                    height=350,
                )

            # Monthly
            date_cols = detect_datetime_columns(
                business_df
            )

            if date_cols:

                date_col = date_cols[0]

                temp = business_df.copy()

                temp["_date"] = safe_to_datetime(
                    temp[date_col]
                )

                monthly = (
                    temp.dropna(
                        subset=["_date"]
                    )
                    .set_index("_date")[
                        sales_col
                    ]
                    .resample("ME")
                    .sum()
                )

                if not monthly.empty:

                    st.subheader(
                        "Monthly Sales"
                    )

                    st.line_chart(
                        monthly,
                        height=350,
                    )


    # ========================================================
    # POWER BI
    # ========================================================

    elif st.session_state.page == "Power BI":

        st.title("Power BI")

        st.caption(
            "Prepare cleaned and summarized datasets for Power BI."
        )

        profile_df = analysis[
            "profile_df"
        ]

        exports = create_powerbi_exports(
            cleaned_df,
            profile_df,
            business_df,
        )

        st.subheader(
            "Power BI Export Package"
        )

        for filename, data in exports.items():

            st.download_button(
                f"Download {filename}",
                data,
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}",
            )

        zip_data = create_zip(
            exports
        )

        st.divider()

        st.download_button(
            "Download Complete Power BI Package",
            zip_data,
            file_name="DataGuard_PowerBI_Package.zip",
            mime="application/zip",
            type="primary",
        )

        st.info(
            "Recommended Power BI model: use cleaned data as the main fact table and the generated summary files for supporting visuals."
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif st.session_state.page == "AI Analysis":

        st.title("AI Analysis")

        st.caption(
            "Interpret measured DataGuard signals using Gemini."
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Quality Score",
            f"{analysis['quality_score']:.2f}%",
        )

        b.metric(
            "Records",
            f"{analysis['rows']:,}",
        )

        c.metric(
            "ML Anomalies",
            f"{analysis['ml_anomalies']:,}",
        )

        d.metric(
            "Missing Values",
            f"{analysis['missing_values']:,}",
        )

        st.divider()

        st.subheader(
            "DataGuard AI Report"
        )

        st.write(
            "Gemini interprets the measured DataGuard results. "
            "It does not replace the application's calculations."
        )

        if st.button(
            "✨ Generate Gemini Analysis",
            type="primary",
        ):

            with st.spinner(
                "Gemini is analyzing the DataGuard results..."
            ):

                result = run_gemini(
                    analysis
                )

                st.session_state.gemini_analysis = (
                    result
                )

        if st.session_state.gemini_analysis:

            st.subheader(
                "Gemini Findings"
            )

            result = (
                st.session_state.gemini_analysis
            )

            if isinstance(
                result,
                str,
            ):

                output_text = result

            elif hasattr(
                result,
                "output_text",
            ):

                output_text = (
                    result.output_text
                )

            else:

                output_text = str(result)

            st.markdown(
                output_text,
                unsafe_allow_html=False,
            )

        else:

            st.info(
                "Click Generate Gemini Analysis to generate the AI report."
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif st.session_state.page == "Reports":

        st.title("Reports")

        st.caption(
            "Executive summary of the DataGuard AI assessment."
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Overall Score",
            f"{analysis['quality_score']:.2f}%",
        )

        b.metric(
            "Records",
            f"{analysis['rows']:,}",
        )

        c.metric(
            "Issues Resolved",
            f"{analysis['rows_removed'] + analysis['values_filled']:,}",
        )

        d.metric(
            "Anomalies",
            f"{analysis['ml_anomalies']:,}",
        )

        st.divider()

        st.subheader(
            "Executive Summary"
        )

        st.write(
            f"""
            **{analysis['file_name']}** contains
            **{analysis['rows']:,} records** across
            **{analysis['columns']} columns**.

            The measured DataGuard quality score is
            **{analysis['quality_score']:.2f}%**.

            The dataset contains
            **{analysis['missing_values']:,} missing values**,
            **{analysis['duplicates']:,} duplicate rows** and
            **{analysis['invalid_values']:,} invalid values**.

            Isolation Forest identified
            **{analysis['ml_anomalies']:,} potential anomalies**
            at **{analysis['contamination_pct']}% sensitivity**.

            These anomaly results are screening signals and should
            be reviewed before being treated as confirmed business errors.
            """
        )

        st.subheader(
            "Cleaning Summary"
        )

        report_table = pd.DataFrame(
            {
                "Metric": [
                    "Rows Removed",
                    "Values Filled",
                    "City Standardization Changes",
                    "IQR Outliers",
                    "ML Anomalies",
                ],
                "Value": [
                    analysis["rows_removed"],
                    analysis["values_filled"],
                    analysis["city_standardized"],
                    analysis["iqr_outliers"],
                    analysis["ml_anomalies"],
                ],
            }
        )

        st.dataframe(
            report_table,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Recommendations"
        )

        recommendations = [
            "Review ML anomalies before treating them as confirmed errors.",
            "Investigate repeated duplicate records at the source.",
            "Monitor missing-value patterns during future ingestion.",
            "Use standardized city/category fields for Power BI dimensions.",
            "Keep statistical outliers for investigation rather than automatically deleting them.",
        ]

        for recommendation in recommendations:
            st.markdown(
                f"• {recommendation}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform<br>
        Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
