%%writefile /content/app.py

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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f8fafc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #111827;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .brand-box {
        padding: 4px 4px 20px 4px;
    }

    .brand-icon {
        font-size: 30px;
    }

    .brand-name {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        margin-top: 3px;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #6b7280;
        margin-top: 2px;
    }

    .sidebar-label {
        font-size: 10px;
        font-weight: 800;
        color: #9ca3af;
        letter-spacing: .08em;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .sidebar-note {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px;
        font-size: 11px;
        color: #6b7280;
        line-height: 1.5;
    }

    /* ---------- PAGE HEADER ---------- */

    .page-title {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
        letter-spacing: -0.8px;
    }

    .page-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,.02);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 18px;
        min-height: 145px;
        box-shadow: 0 1px 2px rgba(0,0,0,.02);
    }

    .metric-card-dark {
        background: #111827;
        border: 1px solid #111827;
        border-radius: 16px;
        padding: 18px;
        min-height: 145px;
        box-shadow: 0 4px 12px rgba(15,23,42,.10);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #6b7280;
        letter-spacing: .04em;
    }

    .metric-label-dark {
        font-size: 11px;
        font-weight: 700;
        color: #9ca3af;
        letter-spacing: .04em;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 800;
        color: #111827;
        margin-top: 10px;
    }

    .metric-value-dark {
        font-size: 31px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 10px;
    }

    .metric-description {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 5px;
    }

    .metric-good {
        font-size: 12px;
        color: #22c55e;
        margin-top: 5px;
        font-weight: 700;
    }

    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 17px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 3px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 16px;
    }

    /* ---------- STATUS ---------- */

    .status-complete {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        border-radius: 24px;
        padding: 8px 15px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
    }

    .status-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        border-radius: 12px;
        padding: 12px 15px;
        font-size: 13px;
    }

    .status-info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 12px;
        padding: 12px 15px;
        font-size: 13px;
    }

    /* ---------- DATASET BAR ---------- */

    .dataset-bar {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 15px 18px;
        margin-bottom: 20px;
    }

    .dataset-label {
        font-size: 10px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: .05em;
        font-weight: 700;
    }

    .dataset-name {
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
    }

    .dataset-time {
        color: #6b7280;
        font-size: 13px;
    }

    /* ---------- PIPELINE ---------- */

    .pipeline-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
        min-height: 75px;
    }

    .pipeline-step {
        font-size: 10px;
        color: #9ca3af;
        font-weight: 700;
    }

    .pipeline-name {
        color: #047857;
        font-size: 13px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        border-top: 1px solid #e5e7eb;
        margin-top: 35px;
        padding-top: 18px;
        text-align: center;
        color: #9ca3af;
        font-size: 11px;
    }

    /* ---------- STREAMLIT BUTTON ---------- */

    .stButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 14px;
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "business_df" not in st.session_state:
    st.session_state.business_df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "analysis_time" not in st.session_state:
    st.session_state.analysis_time = None

if "gemini_analysis" not in st.session_state:
    st.session_state.gemini_analysis = None


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
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_to_datetime(series):
    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except TypeError:
        try:
            return pd.to_datetime(
                series,
                errors="coerce",
            )
        except Exception:
            return pd.Series(pd.NaT, index=series.index)
    except Exception:
        return pd.Series(pd.NaT, index=series.index)


def detect_datetime_columns(df):
    datetime_columns = []

    tokens = [
        "date",
        "time",
        "datetime",
        "timestamp",
        "created",
        "updated",
        "month",
        "year",
    ]

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        name = str(column).lower()

        parsed = safe_to_datetime(df[column])
        ratio = parsed.notna().mean()

        if any(token in name for token in tokens):
            if ratio >= 0.50:
                datetime_columns.append(column)
        else:
            if ratio >= 0.95:
                datetime_columns.append(column)

    return list(dict.fromkeys(datetime_columns))


def detect_identifier_columns(df, datetime_columns=None):

    if datetime_columns is None:
        datetime_columns = []

    identifier_columns = []

    for column in df.columns:

        if column in datetime_columns:
            continue

        name = str(column).lower()

        identifier_tokens = [
            "_id",
            "id_",
            "identifier",
            "customerid",
            "customer_id",
            "orderid",
            "order_id",
            "productid",
            "product_id",
        ]

        if any(token in name for token in identifier_tokens):
            identifier_columns.append(column)
            continue

        if len(df) > 0:

            unique_ratio = (
                df[column].nunique(dropna=True) / len(df)
            )

            if (
                unique_ratio >= 0.98
                and (
                    pd.api.types.is_integer_dtype(df[column])
                    or pd.api.types.is_object_dtype(df[column])
                    or pd.api.types.is_string_dtype(df[column])
                )
            ):
                identifier_columns.append(column)

    return list(dict.fromkeys(identifier_columns))


def profile_data(df):

    rows = []

    for column in df.columns:

        rows.append(
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Non-Null Count": int(df[column].notna().sum()),
                "Missing": int(df[column].isna().sum()),
                "Unique Values": int(
                    df[column].nunique(dropna=True)
                ),
            }
        )

    return pd.DataFrame(rows)


def detect_invalid_values(df):

    invalid_details = []

    tokens = [
        "age",
        "quantity",
        "qty",
        "sales",
        "revenue",
        "amount",
        "price",
        "profit",
        "discount",
        "salary",
    ]

    invalid_count = 0

    for column in df.columns:

        name = str(column).lower()

        if not any(token in name for token in tokens):
            continue

        series = df[column]

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_numeric_mask = (
            series.notna()
            & numeric.isna()
        )

        count_non_numeric = int(
            non_numeric_mask.sum()
        )

        if count_non_numeric > 0:

            invalid_details.append(
                {
                    "Column": column,
                    "Issue": "Non-numeric value",
                    "Count": count_non_numeric,
                }
            )

            invalid_count += count_non_numeric

        if "age" in name:

            negative_mask = numeric < 0
            count_negative = int(
                negative_mask.fillna(False).sum()
            )

            if count_negative > 0:

                invalid_details.append(
                    {
                        "Column": column,
                        "Issue": "Negative age",
                        "Count": count_negative,
                    }
                )

                invalid_count += count_negative

        elif any(
            token in name
            for token in [
                "quantity",
                "qty",
                "sales",
                "revenue",
                "amount",
                "price",
            ]
        ):

            negative_mask = numeric < 0
            count_negative = int(
                negative_mask.fillna(False).sum()
            )

            if count_negative > 0:

                invalid_details.append(
                    {
                        "Column": column,
                        "Issue": "Negative numeric value",
                        "Count": count_negative,
                    }
                )

                invalid_count += count_negative

    return invalid_count, pd.DataFrame(invalid_details)


def standardize_city_column(df):

    cleaned = df.copy()

    changes = 0
    city_columns = []

    for column in cleaned.columns:

        if "city" not in str(column).lower():
            continue

        city_columns.append(column)

        original = cleaned[column].copy()

        normalized = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        standardized = normalized.map(CITY_MAPPING)

        standardized = standardized.fillna(
            normalized.str.title()
        )

        cleaned[column] = standardized

        changes += int(
            (
                original.astype("string")
                != cleaned[column].astype("string")
            ).fillna(False).sum()
        )

    return cleaned, changes


def detect_iqr_outliers(df):

    details = []
    total_outliers = 0

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        count = int(mask.fillna(False).sum())

        if count > 0:

            total_outliers += count

            details.append(
                {
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "Lower Bound": lower,
                    "Upper Bound": upper,
                    "Outlier Count": count,
                }
            )

    return total_outliers, pd.DataFrame(details)


def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:
        return pd.Series(False, index=df.index), 0

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_df = numeric_df.fillna(
        numeric_df.median(numeric_only=True)
    )

    numeric_df = numeric_df.fillna(0)

    if len(numeric_df) < 10:
        return (
            pd.Series(False, index=df.index),
            0,
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_pct / 100,
        random_state=42,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    anomaly_mask = predictions == -1

    return (
        pd.Series(
            anomaly_mask,
            index=df.index,
        ),
        int(anomaly_mask.sum()),
    )


def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
):

    if total_cells <= 0:
        return 100.0

    missing_penalty = (
        missing_count / total_cells
    ) * 100

    duplicate_penalty = (
        duplicate_count
        / max(1, total_cells)
    ) * 100

    invalid_penalty = (
        invalid_count
        / max(1, total_cells)
    ) * 100

    score = 100 - (
        missing_penalty * 0.50
        + duplicate_penalty * 0.30
        + invalid_penalty * 0.20
    )

    return round(
        max(0, min(100, score)),
        2,
    )


def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    rows_removed = (
        original_rows - len(cleaned)
    )

    cleaned, city_changes = (
        standardize_city_column(cleaned)
    )

    values_filled = 0

    numeric_columns = (
        cleaned
        .select_dtypes(include=np.number)
        .columns
    )

    for column in numeric_columns:

        try:

            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            ).astype("float64")

        except Exception:
            continue

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            median_value = (
                cleaned[column].median()
            )

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

                values_filled += (
                    missing_before
                )

    categorical_columns = (
        cleaned
        .select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
    )

    for column in categorical_columns:

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            mode_values = (
                cleaned[column]
                .mode(dropna=True)
            )

            if len(mode_values) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode_values.iloc[0])
                )

                values_filled += (
                    missing_before
                )

    return (
        cleaned,
        rows_removed,
        values_filled,
        city_changes,
    )


def find_sales_column(df):

    preferred_names = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    for name in preferred_names:

        for column in df.columns:

            if str(column).lower() == name:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    return column

    for column in df.columns:

        name = str(column).lower()

        if any(
            token in name
            for token in [
                "sales",
                "revenue",
                "amount",
            ]
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

    return None


def prepare_business_sales(df):

    sales_column = find_sales_column(df)

    if sales_column is None:
        return (
            pd.DataFrame(),
            None,
            None,
        )

    business_df = df.copy()

    business_df[sales_column] = pd.to_numeric(
        business_df[sales_column],
        errors="coerce",
    )

    business_df = business_df[
        business_df[sales_column].notna()
    ]

    business_df = business_df[
        business_df[sales_column] >= 0
    ]

    if len(business_df) == 0:
        return (
            pd.DataFrame(),
            sales_column,
            None,
        )

    q1 = business_df[
        sales_column
    ].quantile(0.25)

    q3 = business_df[
        sales_column
    ].quantile(0.75)

    iqr = q3 - q1

    upper_bound = (
        q3 + 1.5 * iqr
    )

    chart_df = business_df[
        business_df[sales_column]
        <= upper_bound
    ].copy()

    return (
        chart_df,
        sales_column,
        upper_bound,
    )


def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

    profile = profile_data(
        cleaned_df
    )

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    exports[
        "DataGuard_Data_Profile.csv"
    ] = profile.to_csv(
        index=False
    ).encode("utf-8")

    if (
        sales_column is not None
        and not business_df.empty
    ):

        sales_summary = (
            business_df
            .groupby(sales_column)
            .size()
            .reset_index(
                name="Record_Count"
            )
        )

        exports[
            "DataGuard_Sales_Summary.csv"
        ] = sales_summary.to_csv(
            index=False
        ).encode("utf-8")

    city_columns = [
        column
        for column in cleaned_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns:

        city_column = city_columns[0]

        city_summary = (
            cleaned_df
            .groupby(city_column)
            .size()
            .reset_index(
                name="Record_Count"
            )
        )

        exports[
            "DataGuard_City_Summary.csv"
        ] = city_summary.to_csv(
            index=False
        ).encode("utf-8")

    datetime_columns = (
        detect_datetime_columns(
            cleaned_df
        )
    )

    if (
        datetime_columns
        and sales_column is not None
    ):

        date_column = datetime_columns[0]

        temp = cleaned_df.copy()

        temp["_DG_DATE"] = (
            safe_to_datetime(
                temp[date_column]
            )
        )

        temp[sales_column] = pd.to_numeric(
            temp[sales_column],
            errors="coerce",
        )

        monthly = (
            temp.dropna(
                subset=[
                    "_DG_DATE",
                    sales_column,
                ]
            )
            .assign(
                Month=lambda x:
                x["_DG_DATE"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month")[
                sales_column
            ]
            .sum()
            .reset_index()
        )

        exports[
            "DataGuard_Monthly_Sales.csv"
        ] = monthly.to_csv(
            index=False
        ).encode("utf-8")

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for name, data in exports.items():

            zip_file.writestr(
                name,
                data,
            )

    buffer.seek(0)

    return buffer.getvalue()


def generate_root_cause_hypotheses(
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_count,
    ml_count,
):

    causes = []

    if missing_count > 0:
        causes.append(
            "Missing values may indicate incomplete data entry, optional fields, or upstream extraction gaps."
        )

    if duplicate_count > 0:
        causes.append(
            "Duplicate records may result from repeated ingestion, retry logic, or duplicate source records."
        )

    if invalid_count > 0:
        causes.append(
            "Invalid numeric values may indicate inconsistent source data or weak validation rules."
        )

    if iqr_count > 0:
        causes.append(
            "IQR outliers may represent legitimate extreme observations and should be validated against business context."
        )

    if ml_count > 0:
        causes.append(
            "Isolation Forest anomalies indicate unusual combinations of numeric values and should be investigated before being treated as errors."
        )

    return causes


# ============================================================
# GEMINI
# ============================================================

def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:
        return (
            "Gemini AI is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets "
            "or environment variables."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are analyzing a data quality report generated by DataGuard AI.

Analyze ONLY the supplied report.

Do NOT invent:
- columns
- errors
- causes
- business events
- fraud
- system failures
- ETL failures
- retry behavior
- manual entry problems
- facts not present in the report

Possible causes must be clearly labeled as hypotheses.

Confirmed quality problems must only use supplied metrics.

IQR outliers and Isolation Forest anomalies are NOT automatically data errors.

Cleaning has already been performed. Do not recommend repeating the same cleaning.

Business chart extreme sales exclusion is visualization-only.
Those records were NOT deleted from the analytical dataset.

Keep the analysis concise and portfolio-friendly.

Use exactly these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

DATA QUALITY REPORT:

{summary_text}
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "temperature": 0.1
            },
        )

        return response.output_text

    except Exception as e:

        return (
            "Gemini analysis could not be generated.\n\n"
            f"Reason: {str(e)}\n\n"
            "The DataGuard AI core pipeline is still fully functional."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-icon">🛡️</div>
            <div class="brand-name">DataGuard AI</div>
            <div class="brand-subtitle">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Data Quality",
            "Anomalies",
            "Cleaning",
            "Analytics",
            "Power BI",
            "AI Analysis",
            "Reports",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-label">CONFIGURATION</div>',
        unsafe_allow_html=True,
    )

    contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.divider()

    st.markdown(
        """
        <div class="sidebar-note">
            <b>ML anomalies are screening signals,
            not confirmed errors.</b>
            <br><br>
            DataGuard AI combines statistical
            validation with machine-learning
            anomaly detection.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            margin-top:18px;
            color:#9ca3af;
            font-size:10px;
            text-align:center;
        ">
            DataGuard AI • Portfolio Edition
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

if st.session_state.df is None:

    st.markdown(
        """
        <div style="margin-bottom:25px;">
            <div class="page-title">
                Data Quality Command Center
            </div>

            <div class="page-subtitle">
                Profile, detect, clean and monitor
                your data in one workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <div class="section-title">
                Upload your dataset
            </div>

            <div class="section-subtitle">
                Upload a CSV or Excel file to begin
                the DataGuard AI quality pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
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

            st.session_state.analysis_time = (
                datetime.now()
            )

            st.session_state.gemini_analysis = (
                None
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"Unable to read the file: {e}"
            )

    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

df = st.session_state.df

datetime_columns = detect_datetime_columns(
    df
)

identifier_columns = (
    detect_identifier_columns(
        df,
        datetime_columns,
    )
)

profile_df = profile_data(df)

missing_count = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

invalid_count, invalid_df = (
    detect_invalid_values(df)
)

iqr_count, iqr_df = (
    detect_iqr_outliers(df)
)

anomaly_mask, ml_anomaly_count = (
    detect_ml_anomalies(
        df,
        contamination_pct,
    )
)

normal_records = (
    len(df) - ml_anomaly_count
)

quality_score = build_quality_score(
    total_cells=df.shape[0] * df.shape[1],
    missing_count=missing_count,
    duplicate_count=duplicate_count,
    invalid_count=invalid_count,
)

(
    cleaned_df,
    rows_removed,
    values_filled,
    city_changes,
) = clean_dataset(df)

(
    business_df,
    sales_column,
    sales_upper_bound,
) = prepare_business_sales(
    cleaned_df
)

st.session_state.cleaned_df = (
    cleaned_df
)

st.session_state.business_df = (
    business_df
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    # HEADER
    header_col, status_col = st.columns(
        [7, 2]
    )

    with header_col:

        st.markdown(
            """
            <div>
                <div style="
                    font-size:30px;
                    font-weight:800;
                    color:#111827;
                    letter-spacing:-0.8px;
                ">
                    Data Quality Overview
                </div>

                <div style="
                    color:#6b7280;
                    font-size:14px;
                    margin-top:5px;
                ">
                    Monitor the health, reliability
                    and anomaly status of your dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with status_col:

        st.markdown(
            """
            <div style="
                background:#ecfdf5;
                border:1px solid #a7f3d0;
                color:#047857;
                border-radius:24px;
                padding:8px 15px;
                font-size:13px;
                font-weight:700;
                text-align:center;
                margin-top:5px;
            ">
                ● Analysis Complete
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ACTIVE DATASET
    analysis_time = (
        st.session_state.analysis_time
    )

    if analysis_time is not None:

        formatted_time = (
            analysis_time.strftime(
                "%d %b %Y, %I:%M %p"
            )
        )

    else:
        formatted_time = "Recently"

    st.markdown(
        f"""
        <div class="dataset-bar">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
            ">

                <div>
                    <div class="dataset-label">
                        Active Dataset
                    </div>

                    <div class="dataset-name">
                        📄 {st.session_state.file_name}
                    </div>
                </div>

                <div class="dataset-time">
                    Last analyzed:
                    <strong style="color:#111827;">
                        {formatted_time}
                    </strong>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI CARDS
    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div class="metric-label">
                        TOTAL RECORDS
                    </div>

                    <div style="font-size:21px;">
                        📊
                    </div>

                </div>

                <div class="metric-value">
                    {len(df):,}
                </div>

                <div class="metric-description">
                    Records analyzed
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div class="metric-label">
                        MISSING CELLS
                    </div>

                    <div style="font-size:21px;">
                        ⚠️
                    </div>

                </div>

                <div class="metric-value">
                    {missing_count:,}
                </div>

                <div class="metric-description">
                    Missing values detected
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div class="metric-label">
                        DUPLICATES
                    </div>

                    <div style="font-size:21px;">
                        🔁
                    </div>

                </div>

                <div class="metric-value">
                    {duplicate_count:,}
                </div>

                <div class="metric-description">
                    Duplicate records
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:

        st.markdown(
            f"""
            <div class="metric-card-dark">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div class="metric-label-dark">
                        QUALITY SCORE
                    </div>

                    <div style="font-size:21px;">
                        🛡️
                    </div>

                </div>

                <div class="metric-value-dark">
                    {quality_score}
                    <span style="
                        font-size:14px;
                        color:#9ca3af;
                    ">
                        /100
                    </span>
                </div>

                <div class="metric-good">
                    ● Excellent
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # QUALITY HEALTH
    health_left, health_right = st.columns(
        [1.15, 1]
    )

    with health_left:

        st.markdown(
            f"""
            <div class="card">

                <div class="section-title">
                    Dataset Health
                </div>

                <div class="section-subtitle">
                    Overall quality assessment
                </div>

                <div style="
                    height:13px;
                    background:#e5e7eb;
                    border-radius:20px;
                    overflow:hidden;
                ">

                    <div style="
                        width:{quality_score}%;
                        height:100%;
                        background:linear-gradient(
                            90deg,
                            #10b981,
                            #22c55e
                        );
                        border-radius:20px;
                    ">
                    </div>

                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-top:9px;
                    font-size:12px;
                    color:#6b7280;
                ">

                    <span>0</span>

                    <strong style="color:#111827;">
                        {quality_score}% healthy
                    </strong>

                    <span>100</span>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with health_right:

        total_issue_occurrences = (
            missing_count
            + duplicate_count
            + invalid_count
        )

        st.markdown(
            f"""
            <div class="card">

                <div class="section-title">
                    Issues Requiring Attention
                </div>

                <div class="section-subtitle">
                    Confirmed quality issue occurrences
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:10px;
                ">
                    <span style="font-size:13px;">
                        Missing values
                    </span>

                    <strong>
                        {missing_count:,}
                    </strong>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:10px;
                ">
                    <span style="font-size:13px;">
                        Duplicate records
                    </span>

                    <strong>
                        {duplicate_count:,}
                    </strong>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                ">
                    <span style="font-size:13px;">
                        Invalid values
                    </span>

                    <strong>
                        {invalid_count:,}
                    </strong>
                </div>

                <div style="
                    border-top:1px solid #e5e7eb;
                    margin-top:15px;
                    padding-top:13px;
                    display:flex;
                    justify-content:space-between;
                    font-size:13px;
                ">

                    <strong>
                        Total issue occurrences
                    </strong>

                    <strong>
                        {total_issue_occurrences:,}
                    </strong>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ANOMALY MONITORING
    st.markdown(
        """
        <div class="section-title">
            Anomaly Monitoring
        </div>

        <div class="section-subtitle">
            Statistical and machine-learning screening
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.markdown(
            f"""
            <div class="card">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    IQR OUTLIERS
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {iqr_count:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Statistically unusual observations
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:

        st.markdown(
            f"""
            <div class="card">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    ML ANOMALIES
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {ml_anomaly_count:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Isolation Forest screening signals
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:

        st.markdown(
            f"""
            <div class="card">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    NORMAL RECORDS
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {normal_records:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Not flagged by Isolation Forest
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="status-warning" style="margin-top:12px;">
            <b>Interpretation:</b>
            IQR outliers and ML anomalies are screening
            signals. They are not automatically confirmed
            data errors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # CLEANING SUMMARY
    st.markdown(
        """
        <div class="section-title">
            Cleaning Summary
        </div>

        <div class="section-subtitle">
            Automated transformations applied to the dataset
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    ROWS REMOVED
                </div>

                <div class="metric-value">
                    {rows_removed:,}
                </div>

                <div class="metric-description">
                    Duplicate rows removed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    VALUES FILLED
                </div>

                <div class="metric-value">
                    {values_filled:,}
                </div>

                <div class="metric-description">
                    Missing values imputed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">
                    CITIES STANDARDIZED
                </div>

                <div class="metric-value">
                    {city_changes:,}
                </div>

                <div class="metric-description">
                    City values normalized
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # DATASET PREVIEW
    st.markdown(
        """
        <div class="section-title">
            Dataset Preview
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_left, preview_right = st.columns(
        [2, 1]
    )

    with preview_left:

        st.markdown(
            f"""
            <div style="
                color:#6b7280;
                font-size:12px;
                margin-bottom:8px;
            ">
                Showing <strong>10</strong> of
                <strong>{len(df):,}</strong> records
            </div>
            """,
            unsafe_allow_html=True,
        )

    with preview_right:

        st.markdown(
            """
            <div style="
                color:#9ca3af;
                font-size:12px;
                text-align:right;
            ">
                Preview only — full dataset analyzed
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # PIPELINE STATUS
    st.markdown(
        """
        <div class="section-title">
            Pipeline Status
        </div>

        <div class="section-subtitle">
            DataGuard AI processing pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = [
        ("01", "Profile"),
        ("02", "Quality"),
        ("03", "Anomalies"),
        ("04", "Cleaning"),
        ("05", "Analytics"),
        ("06", "Exports"),
    ]

    cols = st.columns(6)

    for col, (step, name) in zip(
        cols,
        pipeline,
    ):

        with col:

            st.markdown(
                f"""
                <div class="pipeline-card">

                    <div class="pipeline-step">
                        STEP {step}
                    </div>

                    <div class="pipeline-name">
                        ✓ {name}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.markdown(
        """
        <div class="page-title">
            Data Quality
        </div>

        <div class="page-subtitle">
            Detailed profiling and validation results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

    q2.metric(
        "Duplicates",
        f"{duplicate_count:,}",
    )

    q3.metric(
        "Invalid Values",
        f"{invalid_count:,}",
    )

    q4.metric(
        "Quality Score",
        f"{quality_score}/100",
    )

    st.markdown("### Dataset Profile")

    st.dataframe(
        profile_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Missing Values")

    missing_by_column = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing_by_column.columns = [
        "Column",
        "Missing Count",
    ]

    missing_by_column = (
        missing_by_column[
            missing_by_column[
                "Missing Count"
            ] > 0
        ]
        .sort_values(
            "Missing Count",
            ascending=False,
        )
    )

    if len(missing_by_column) > 0:

        st.dataframe(
            missing_by_column,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No missing values detected."
        )

    st.markdown("### Invalid Values")

    if len(invalid_df) > 0:

        st.dataframe(
            invalid_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No invalid values detected."
        )

    st.markdown("### Detected Date/Time Columns")

    if datetime_columns:

        st.write(
            ", ".join(
                map(
                    str,
                    datetime_columns,
                )
            )
        )

    else:

        st.info(
            "No date/time columns detected."
        )

    st.markdown("### Detected Identifier Columns")

    if identifier_columns:

        st.write(
            ", ".join(
                map(
                    str,
                    identifier_columns,
                )
            )
        )

    else:

        st.info(
            "No identifier columns detected."
        )


# ============================================================
# ANOMALIES
# ============================================================

elif page == "Anomalies":

    st.markdown(
        """
        <div class="page-title">
            Anomaly Detection
        </div>

        <div class="page-subtitle">
            Statistical outlier detection and Isolation Forest
            screening.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "IQR Outliers",
        f"{iqr_count:,}",
    )

    a2.metric(
        "ML Anomalies",
        f"{ml_anomaly_count:,}",
    )

    a3.metric(
        "Normal Records",
        f"{normal_records:,}",
    )

    st.warning(
        "IQR outliers and Isolation Forest anomalies "
        "are screening signals, not automatically confirmed errors."
    )

    st.markdown("### IQR Outlier Analysis")

    if len(iqr_df) > 0:

        st.dataframe(
            iqr_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No IQR outliers detected."
        )

    st.markdown("### Isolation Forest")

    st.write(
        f"""
        Isolation Forest sensitivity:
        **{contamination_pct}%**
        """
    )

    anomaly_results = df.copy()

    anomaly_results[
        "ML_Anomaly"
    ] = np.where(
        anomaly_mask,
        "Anomaly",
        "Normal",
    )

    st.dataframe(
        anomaly_results[
            anomaly_results["ML_Anomaly"]
            == "Anomaly"
        ].head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Showing up to 100 flagged records."
    )


# ============================================================
# CLEANING
# ============================================================

elif page == "Cleaning":

    st.markdown(
        """
        <div class="page-title">
            Automated Cleaning
        </div>

        <div class="page-subtitle">
            Review the transformations applied by DataGuard AI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Rows",
        f"{len(df):,}",
    )

    c2.metric(
        "Cleaned Rows",
        f"{len(cleaned_df):,}",
    )

    c3.metric(
        "Rows Removed",
        f"{rows_removed:,}",
    )

    st.markdown("### Cleaning Actions")

    cleaning_actions = pd.DataFrame(
        {
            "Action": [
                "Duplicate removal",
                "Missing value imputation",
                "City standardization",
                "Numeric type normalization",
            ],
            "Result": [
                f"{rows_removed:,} rows removed",
                f"{values_filled:,} values filled",
                f"{city_changes:,} city values standardized",
                "Numeric columns normalized",
            ],
        }
    )

    st.dataframe(
        cleaning_actions,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Cleaned Dataset Preview")

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Download Cleaned Dataset")

    cleaned_csv = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Cleaned CSV",
        data=cleaned_csv,
        file_name="DataGuard_Cleaned_Data.csv",
        mime="text/csv",
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        """
        <div class="page-title">
            Business Analytics
        </div>

        <div class="page-subtitle">
            Explore business-level patterns from the cleaned dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if business_df.empty:

        st.info(
            "No Sales, Revenue or Amount column was detected, "
            "so sales-based business analytics are unavailable."
        )

    else:

        st.success(
            f"Detected sales column: **{sales_column}**"
        )

        if sales_upper_bound is not None:

            st.caption(
                f"""
                For visualization only, Sales values above
                the cleaned-data IQR upper bound
                ({sales_upper_bound:,.2f}) are excluded.
                These records are not deleted from the
                analytical dataset.
                """
            )

        # SALES DISTRIBUTION

        st.markdown("### Sales Distribution")

        sales_values = business_df[
            sales_column
        ].dropna()

        if len(sales_values) > 0:

            histogram_counts, histogram_edges = (
                np.histogram(
                    sales_values,
                    bins=10,
                )
            )

            histogram_df = pd.DataFrame(
                {
                    "Sales Range": [
                        f"{histogram_edges[i]:,.0f} - "
                        f"{histogram_edges[i+1]:,.0f}"
                        for i in range(
                            len(histogram_edges) - 1
                        )
                    ],
                    "Records": histogram_counts,
                }
            )

            st.bar_chart(
                histogram_df.set_index(
                    "Sales Range"
                )
            )

        # CITY

        city_columns = [
            column
            for column in business_df.columns
            if "city" in str(column).lower()
        ]

        if city_columns:

            city_column = city_columns[0]

            st.markdown("### Records by City")

            city_summary = (
                business_df[
                    city_column
                ]
                .value_counts()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                city_summary
            )

        # PRODUCT

        product_columns = [
            column
            for column in business_df.columns
            if "product" in str(column).lower()
        ]

        if product_columns:

            product_column = (
                product_columns[0]
            )

            st.markdown(
                "### Sales by Product"
            )

            product_sales = (
                business_df
                .groupby(product_column)[
                    sales_column
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                product_sales
            )

        # CATEGORY

        category_columns = [
            column
            for column in business_df.columns
            if "category" in str(column).lower()
        ]

        if category_columns:

            category_column = (
                category_columns[0]
            )

            st.markdown(
                "### Sales by Category"
            )

            category_sales = (
                business_df
                .groupby(category_column)[
                    sales_column
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                category_sales
            )

        # MONTHLY SALES

        if datetime_columns:

            date_column = datetime_columns[0]

            st.markdown(
                "### Monthly Sales Trend"
            )

            monthly_df = cleaned_df.copy()

            monthly_df["_DG_DATE"] = (
                safe_to_datetime(
                    monthly_df[
                        date_column
                    ]
                )
            )

            monthly_df[
                sales_column
            ] = pd.to_numeric(
                monthly_df[
                    sales_column
                ],
                errors="coerce",
            )

            monthly_df = (
                monthly_df
                .dropna(
                    subset=[
                        "_DG_DATE",
                        sales_column,
                    ]
                )
                .assign(
                    Month=lambda x:
                    x["_DG_DATE"]
                    .dt.to_period("M")
                    .astype(str)
                )
                .groupby("Month")[
                    sales_column
                ]
                .sum()
            )

            if len(monthly_df) > 0:

                st.line_chart(
                    monthly_df
                )


# ============================================================
# POWER BI
# ============================================================

elif page == "Power BI":

    st.markdown(
        """
        <div class="page-title">
            Power BI Export
        </div>

        <div class="page-subtitle">
            Generate Power BI-ready CSV files from the cleaned data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        """
        DataGuard AI currently uses a manual Power BI workflow.
        Azure API automation is not required.
        """
    )

    exports = create_powerbi_exports(
        cleaned_df,
        business_df,
        sales_column,
    )

    st.markdown("### Available Exports")

    for filename in exports:

        st.write(
            f"✓ {filename}"
        )

    zip_data = create_zip(
        exports
    )

    st.download_button(
        "⬇️ Download Power BI Export ZIP",
        data=zip_data,
        file_name="DataGuard_PowerBI_Exports.zip",
        mime="application/zip",
        type="primary",
    )

    st.markdown("### Manual Power BI Workflow")

    steps = [
        "Download the ZIP file.",
        "Extract the CSV files.",
        "Open Power BI Desktop.",
        "Select Get Data → Text/CSV.",
        "Import the cleaned dataset and supporting tables.",
        "Create relationships and dashboards in Power BI.",
    ]

    for index, step in enumerate(
        steps,
        start=1,
    ):

        st.write(
            f"**{index}.** {step}"
        )

    st.markdown("### Individual Files")

    for filename, data in exports.items():

        st.download_button(
            f"Download {filename}",
            data=data,
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.markdown(
        """
        <div class="page-title">
            AI Analysis
        </div>

        <div class="page-subtitle">
            Generate a portfolio-friendly interpretation
            of the DataGuard AI quality results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        """
        Gemini AI is optional. The core DataGuard AI
        data-quality and anomaly pipeline works independently.
        """
    )

    root_causes = generate_root_cause_hypotheses(
        missing_count,
        duplicate_count,
        invalid_count,
        iqr_count,
        ml_anomaly_count,
    )

    summary_text = f"""
Dataset:
{st.session_state.file_name}

Rows:
{len(df)}

Columns:
{df.shape[1]}

Missing cells:
{missing_count}

Duplicate records:
{duplicate_count}

Invalid values:
{invalid_count}

Quality score:
{quality_score}/100

IQR outlier observations:
{iqr_count}

Isolation Forest sensitivity:
{contamination_pct}%

Isolation Forest anomalies:
{ml_anomaly_count}

Normal records:
{normal_records}

Original rows:
{len(df)}

Cleaned rows:
{len(cleaned_df)}

Rows removed:
{rows_removed}

Values filled:
{values_filled}

Cities standardized:
{city_changes}

Sales column:
{sales_column}

Sales IQR visualization upper bound:
{sales_upper_bound}

Root-cause hypotheses:
{root_causes}
"""

    if st.button(
        "✨ Generate Gemini AI Analysis",
        type="primary",
    ):

        with st.spinner(
            "Gemini is analyzing the data-quality report..."
        ):

            result = get_gemini_analysis(
                summary_text
            )

        st.session_state.gemini_analysis = (
            result
        )

    if st.session_state.gemini_analysis:

        st.markdown(
            st.session_state.gemini_analysis
        )

    else:

        st.info(
            "Click **Generate Gemini AI Analysis** "
            "to request an AI-generated interpretation "
            "of the data-quality results."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        """
        <div class="page-title">
            Data Quality Report
        </div>

        <div class="page-subtitle">
            Consolidated DataGuard AI analysis report.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Dataset Summary")

    report_summary = pd.DataFrame(
        {
            "Metric": [
                "Dataset",
                "Rows",
                "Columns",
                "Missing Cells",
                "Duplicate Records",
                "Invalid Values",
                "Quality Score",
                "IQR Outliers",
                "ML Anomalies",
                "Normal Records",
                "Cleaned Rows",
                "Rows Removed",
                "Values Filled",
                "Cities Standardized",
            ],
            "Value": [
                st.session_state.file_name,
                f"{len(df):,}",
                f"{df.shape[1]:,}",
                f"{missing_count:,}",
                f"{duplicate_count:,}",
                f"{invalid_count:,}",
                f"{quality_score}/100",
                f"{iqr_count:,}",
                f"{ml_anomaly_count:,}",
                f"{normal_records:,}",
                f"{len(cleaned_df):,}",
                f"{rows_removed:,}",
                f"{values_filled:,}",
                f"{city_changes:,}",
            ],
        }
    )

    st.dataframe(
        report_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Root-Cause Hypotheses")

    root_causes = generate_root_cause_hypotheses(
        missing_count,
        duplicate_count,
        invalid_count,
        iqr_count,
        ml_anomaly_count,
    )

    if root_causes:

        for cause in root_causes:

            st.write(
                f"• {cause}"
            )

    else:

        st.success(
            "No quality issues requiring root-cause hypotheses were detected."
        )

    st.markdown("### Final Interpretation")

    if quality_score >= 95:

        st.success(
            f"""
            The dataset has an overall quality score of
            **{quality_score}/100**. Confirmed issues are
            relatively limited compared with the total dataset.
            Statistical and ML anomaly results should be reviewed
            using business context before treating them as errors.
            """
        )

    elif quality_score >= 85:

        st.warning(
            f"""
            The dataset has a quality score of
            **{quality_score}/100**. Several data-quality
            issues should be reviewed before downstream analysis.
            """
        )

    else:

        st.error(
            f"""
            The dataset has a quality score of
            **{quality_score}/100**. Significant data-quality
            remediation is recommended before business reporting.
            """
        )

    st.markdown("### Download Report Data")

    report_csv = report_summary.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Quality Report CSV",
        data=report_csv,
        file_name="DataGuard_Quality_Report.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI &nbsp;•&nbsp;
        AI-Powered Data Quality & Anomaly Detection
        <br>
        Built with Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
