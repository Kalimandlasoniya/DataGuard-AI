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

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #f1f5f9;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .sidebar-brand {
        padding: 10px 5px 22px 5px;
    }

    .sidebar-logo {
        font-size: 30px;
        font-weight: 800;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 800;
        color: #ffffff !important;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #94a3b8 !important;
        margin-top: 3px;
    }

    .sidebar-divider {
        height: 1px;
        background: #334155;
        margin: 14px 0 18px 0;
    }


    /* ========================================================
       PAGE HEADERS
       ======================================================== */

    .page-title {
        font-size: 34px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 15px;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.10);
    }

    .kpi-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #1d4ed8;
    }

    .kpi-caption {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 5px;
    }


    /* ========================================================
       STATUS BADGES
       ======================================================== */

    .status-good {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #dcfce7;
        color: #15803d;
        font-size: 12px;
        font-weight: 700;
    }

    .status-warning {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #fef3c7;
        color: #b45309;
        font-size: 12px;
        font-weight: 700;
    }

    .status-danger {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #fee2e2;
        color: #dc2626;
        font-size: 12px;
        font-weight: 700;
    }


    /* ========================================================
       PANELS
       ======================================================== */

    .dashboard-panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-top: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .panel-title {
        font-size: 17px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .panel-description {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 14px;
    }


    /* ========================================================
       UPLOAD
       ======================================================== */

    .upload-banner {
        background: #ffffff;
        border: 1px dashed #93c5fd;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 10px;
    }


    /* ========================================================
       INFO BOXES
       ======================================================== */

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 13px 15px;
        color: #1e40af;
        font-size: 13px;
        margin: 10px 0;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 13px 15px;
        color: #92400e;
        font-size: 13px;
        margin: 10px 0;
    }

    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 13px 15px;
        color: #166534;
        font-size: 13px;
        margin: 10px 0;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 650;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricValue"] {
        color: #1d4ed8;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #93c5fd;
        border-radius: 12px;
        padding: 10px;
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "df": None,
    "cleaned_df": None,
    "business_df": None,
    "file_name": None,
    "analysis_complete": False,
    "gemini_analysis": None,
    "contamination_pct": 5,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CITY MAPPING
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
# DATETIME DETECTION
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

            return pd.Series(
                pd.NaT,
                index=series.index,
            )

    except Exception:

        return pd.Series(
            pd.NaT,
            index=series.index,
        )


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

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            datetime_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            continue

        column_name = str(column).lower()

        parsed = safe_to_datetime(df[column])

        valid_ratio = parsed.notna().mean()

        if any(
            token in column_name
            for token in tokens
        ):

            if valid_ratio >= 0.50:
                datetime_columns.append(column)

        else:

            if valid_ratio >= 0.95:
                datetime_columns.append(column)

    return list(
        dict.fromkeys(datetime_columns)
    )


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier_columns(
    df,
    datetime_columns=None,
):

    if datetime_columns is None:
        datetime_columns = []

    identifier_columns = []

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

    for column in df.columns:

        if column in datetime_columns:
            continue

        name = str(column).lower()

        if any(
            token in name
            for token in identifier_tokens
        ):

            identifier_columns.append(column)
            continue

        if len(df) > 0:

            unique_ratio = (
                df[column].nunique(
                    dropna=True
                )
                / len(df)
            )

            if (
                unique_ratio >= 0.98
                and (
                    pd.api.types.is_integer_dtype(
                        df[column]
                    )
                    or pd.api.types.is_object_dtype(
                        df[column]
                    )
                    or pd.api.types.is_string_dtype(
                        df[column]
                    )
                )
            ):

                identifier_columns.append(column)

    return list(
        dict.fromkeys(identifier_columns)
    )


# ============================================================
# DATA PROFILING
# ============================================================

def profile_dataset(df):

    return pd.DataFrame(
        {
            "Column": df.columns,

            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],

            "Non-Null Count": [
                int(
                    df[column].notna().sum()
                )
                for column in df.columns
            ],

            "Missing": [
                int(
                    df[column].isna().sum()
                )
                for column in df.columns
            ],

            "Unique Values": [
                int(
                    df[column].nunique(
                        dropna=True
                    )
                )
                for column in df.columns
            ],
        }
    )


# ============================================================
# INVALID VALUES
# ============================================================

def detect_invalid_values(df):

    invalid_details = []

    numeric_tokens = [
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

        if not any(
            token in name
            for token in numeric_tokens
        ):
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        original_non_null = df[column].notna()

        conversion_failures = (
            numeric_values.isna()
            & original_non_null
        )

        negative_values = pd.Series(
            False,
            index=df.index,
        )

        if any(
            token in name
            for token in [
                "age",
                "quantity",
                "qty",
                "sales",
                "revenue",
                "amount",
                "price",
            ]
        ):

            negative_values = (
                numeric_values < 0
            )

        count = int(
            conversion_failures.sum()
            + negative_values.sum()
        )

        if count > 0:

            invalid_count += count

            invalid_details.append(
                {
                    "Column": column,
                    "Invalid Values": count,
                }
            )

    return (
        invalid_count,
        pd.DataFrame(
            invalid_details
        ),
    )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

def standardize_city_column(df):

    result = df.copy()

    total_changes = 0

    city_columns = [
        column
        for column in result.columns
        if "city" in str(column).lower()
    ]

    for column in city_columns:

        original = result[column].copy()

        cleaned = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        standardized = cleaned.map(
            CITY_MAPPING
        )

        standardized = standardized.fillna(
            cleaned.str.title()
        )

        changes = (
            original.astype("string").fillna("")
            != standardized.fillna("")
        )

        changed_count = int(
            changes.sum()
        )

        total_changes += changed_count

        result[column] = standardized

    return (
        result,
        total_changes,
    )


# ============================================================
# IQR OUTLIERS
# ============================================================

def detect_iqr_outliers(df):

    details = []

    total_outliers = 0

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

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

        count = int(
            mask.sum()
        )

        if count > 0:

            total_outliers += count

            details.append(
                {
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "Lower Bound": lower,
                    "Upper Bound": upper,
                    "Outliers": count,
                }
            )

    return (
        total_outliers,
        pd.DataFrame(details),
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:

        return (
            pd.Series(
                False,
                index=df.index,
            ),
            0,
        )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_df = numeric_df.fillna(
        numeric_df.median(
            numeric_only=True
        )
    )

    numeric_df = numeric_df.fillna(0)

    if len(numeric_df) < 10:

        return (
            pd.Series(
                False,
                index=df.index,
            ),
            0,
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=(
            contamination_pct / 100
        ),
        random_state=42,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    anomaly_mask = (
        predictions == -1
    )

    return (
        pd.Series(
            anomaly_mask,
            index=df.index,
        ),
        int(
            anomaly_mask.sum()
        ),
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
):

    if total_cells <= 0:
        return 100.0

    missing_penalty = (
        missing_count
        / total_cells
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
        max(
            0,
            min(
                100,
                score,
            ),
        ),
        2,
    )


# ============================================================
# CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    rows_removed = (
        original_rows
        - len(cleaned)
    )

    cleaned, city_changes = (
        standardize_city_column(
            cleaned
        )
    )

    values_filled = 0

    numeric_columns = (
        cleaned
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numeric_columns:

        try:

            cleaned[column] = (
                pd.to_numeric(
                    cleaned[column],
                    errors="coerce",
                )
                .astype("float64")
            )

        except Exception:
            continue

        missing_before = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing_before > 0:

            median_value = (
                cleaned[column]
                .median()
            )

            if pd.notna(
                median_value
            ):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        median_value
                    )
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
            cleaned[column]
            .isna()
            .sum()
        )

        if missing_before > 0:

            mode_values = (
                cleaned[column]
                .mode(
                    dropna=True
                )
            )

            if len(mode_values) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        mode_values.iloc[0]
                    )
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


# ============================================================
# SALES DETECTION
# ============================================================

def find_sales_column(df):

    preferred_names = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in preferred_names:

        if name in lower_map:

            column = lower_map[name]

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


# ============================================================
# BUSINESS DATA
# ============================================================

def prepare_business_data(df):

    sales_column = (
        find_sales_column(df)
    )

    if sales_column is None:
        return None, None, None

    business_df = df.copy()

    business_df[sales_column] = (
        pd.to_numeric(
            business_df[sales_column],
            errors="coerce",
        )
    )

    business_df = business_df[
        business_df[sales_column]
        .notna()
    ]

    business_df = business_df[
        business_df[sales_column] >= 0
    ]

    if len(business_df) == 0:
        return (
            None,
            sales_column,
            None,
        )

    q1 = (
        business_df[sales_column]
        .quantile(0.25)
    )

    q3 = (
        business_df[sales_column]
        .quantile(0.75)
    )

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


# ============================================================
# POWER BI EXPORTS
# ============================================================

def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

    profile = profile_dataset(
        cleaned_df
    )

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    exports[
        "DataGuard_Data_Profile.csv"
    ] = (
        profile
        .to_csv(index=False)
        .encode("utf-8")
    )

    if (
        business_df is not None
        and sales_column is not None
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
        ] = (
            sales_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    city_columns = [
        column
        for column in cleaned_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns:

        city_column = (
            city_columns[0]
        )

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
        ] = (
            city_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    datetime_columns = (
        detect_datetime_columns(
            cleaned_df
        )
    )

    if (
        datetime_columns
        and sales_column is not None
    ):

        date_column = (
            datetime_columns[0]
        )

        monthly_df = (
            cleaned_df.copy()
        )

        monthly_df[date_column] = (
            safe_to_datetime(
                monthly_df[date_column]
            )
        )

        monthly_df[sales_column] = (
            pd.to_numeric(
                monthly_df[sales_column],
                errors="coerce",
            )
        )

        monthly_df = (
            monthly_df.dropna(
                subset=[
                    date_column,
                    sales_column,
                ]
            )
        )

        monthly_summary = (
            monthly_df
            .assign(
                Month=(
                    monthly_df[
                        date_column
                    ]
                    .dt
                    .to_period("M")
                    .astype(str)
                )
            )
            .groupby("Month")[
                sales_column
            ]
            .sum()
            .reset_index()
        )

        exports[
            "DataGuard_Monthly_Sales.csv"
        ] = (
            monthly_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for filename, data in (
            exports.items()
        ):

            zip_file.writestr(
                filename,
                data,
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    "",
)

if not GEMINI_API_KEY:

    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY",
        "",
    )


def get_gemini_analysis(
    summary_text
):

    if not GEMINI_API_KEY:

        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit "
            "secrets to enable AI analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior data quality analyst.

Analyze ONLY the supplied DataGuard AI report.

Do not invent:
- facts
- columns
- errors
- causes
- business events
- fraud
- system failures
- ETL failures
- retry behavior
- manual-entry behavior

Possible causes must be clearly labeled as hypotheses.

IQR outliers and Isolation Forest anomalies
are NOT automatically confirmed data errors.

The cleaning process has already been performed.
Do not recommend repeating the same cleaning steps.

Business chart filtering is visualization-only.

Use the exact supplied metrics.

Create a concise analysis with:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

REPORT:

{summary_text}
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "temperature": 0.1,
            },
        )

        return response.output_text

    except Exception as error:

        return (
            "Gemini analysis could not be generated.\n\n"
            f"Reason: {error}\n\n"
            "The DataGuard AI pipeline continues "
            "to work without Gemini."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                🛡️
            </div>

            <div class="sidebar-title">
                DataGuard AI
            </div>

            <div class="sidebar-subtitle">
                AI Data Quality Platform
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Navigation"
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
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Anomaly Detection"
    )

    contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
        step=1,
    )

    st.session_state.contamination_pct = (
        contamination_pct
    )

    st.caption(
        "Higher sensitivity flags more records "
        "as unusual. ML anomalies are screening "
        "signals, not confirmed errors."
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Pipeline"
    )

    st.markdown(
        """
        **01** Upload  
        **02** Profile  
        **03** Quality Checks  
        **04** Anomaly Detection  
        **05** Cleaning  
        **06** Analytics  
        **07** Power BI  
        **08** AI Analysis
        """
    )

    st.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "DataGuard AI • Portfolio Edition"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="page-title">'
    'DataGuard AI'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    'AI-powered data quality, anomaly detection '
    'and analytics platform'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    """
    <div class="upload-banner">

        <div style="
            font-size:18px;
            font-weight:750;
            color:#111827;
        ">
            Upload your dataset
        </div>

        <div style="
            color:#6b7280;
            font-size:13px;
            margin-top:5px;
            margin-bottom:8px;
        ">
            Start a new data-quality analysis by uploading
            a CSV or Excel dataset.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a data file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    label_visibility="collapsed",
)


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

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

            st.session_state.analysis_complete = (
                False
            )

            st.session_state.gemini_analysis = None

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )

            st.stop()


# ============================================================
# ANALYSIS
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if not st.session_state.analysis_complete:

        with st.spinner(
            "Analyzing your dataset..."
        ):

            rows = len(df)

            columns = len(df.columns)

            total_cells = (
                rows * columns
            )

            missing_count = int(
                df.isna().sum().sum()
            )

            duplicate_count = int(
                df.duplicated().sum()
            )

            (
                invalid_count,
                invalid_details,
            ) = detect_invalid_values(
                df
            )

            quality_score = (
                build_quality_score(
                    total_cells,
                    missing_count,
                    duplicate_count,
                    invalid_count,
                )
            )

            datetime_columns = (
                detect_datetime_columns(
                    df
                )
            )

            identifier_columns = (
                detect_identifier_columns(
                    df,
                    datetime_columns,
                )
            )

            numeric_count = len(
                df.select_dtypes(
                    include=np.number
                ).columns
            )

            categorical_count = len(
                df.select_dtypes(
                    include=[
                        "object",
                        "string",
                        "category",
                    ]
                ).columns
            )

            (
                iqr_outlier_count,
                iqr_details,
            ) = detect_iqr_outliers(
                df
            )

            (
                anomaly_mask,
                ml_anomaly_count,
            ) = detect_ml_anomalies(
                df,
                contamination_pct,
            )

            normal_count = (
                rows
                - ml_anomaly_count
            )

            (
                cleaned_df,
                rows_removed,
                values_filled,
                city_changes,
            ) = clean_dataset(
                df
            )

            (
                business_df,
                sales_column,
                upper_bound,
            ) = prepare_business_data(
                cleaned_df
            )

            st.session_state.cleaned_df = (
                cleaned_df
            )

            st.session_state.business_df = (
                business_df
            )

            st.session_state.analysis = {

                "rows": rows,

                "columns": columns,

                "total_cells": total_cells,

                "missing_count": missing_count,

                "duplicate_count": duplicate_count,

                "invalid_count": invalid_count,

                "quality_score": quality_score,

                "datetime_columns": (
                    datetime_columns
                ),

                "identifier_columns": (
                    identifier_columns
                ),

                "numeric_count": (
                    numeric_count
                ),

                "categorical_count": (
                    categorical_count
                ),

                "iqr_outlier_count": (
                    iqr_outlier_count
                ),

                "iqr_details": (
                    iqr_details
                ),

                "anomaly_mask": (
                    anomaly_mask
                ),

                "ml_anomaly_count": (
                    ml_anomaly_count
                ),

                "normal_count": (
                    normal_count
                ),

                "invalid_details": (
                    invalid_details
                ),

                "rows_removed": (
                    rows_removed
                ),

                "values_filled": (
                    values_filled
                ),

                "city_changes": (
                    city_changes
                ),

                "sales_column": (
                    sales_column
                ),

                "upper_bound": (
                    upper_bound
                ),
            }

            st.session_state.analysis_complete = (
                True
            )


# ============================================================
# APPLICATION
# ============================================================

if (
    st.session_state.df is not None
    and st.session_state.analysis_complete
):

    df = st.session_state.df

    cleaned_df = (
        st.session_state.cleaned_df
    )

    business_df = (
        st.session_state.business_df
    )

    analysis = (
        st.session_state.analysis
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.markdown(
            '<div class="section-title">'
            'Dashboard Overview'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>{st.session_state.file_name}</b>
                &nbsp; • &nbsp;
                {analysis["rows"]:,} records
                &nbsp; • &nbsp;
                {analysis["columns"]} columns
                &nbsp; • &nbsp;
                Full dataset analyzed
            </div>
            """,
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = (
            st.columns(4)
        )

        score = (
            analysis["quality_score"]
        )

        if score >= 99:
            score_status = "Excellent"
        elif score >= 95:
            score_status = "Good"
        elif score >= 85:
            score_status = "Needs Attention"
        else:
            score_status = "Critical"

        with k1:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        DATA QUALITY SCORE
                    </div>

                    <div class="kpi-value">
                        {score}/100
                    </div>

                    <div class="kpi-caption">
                        {score_status}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        MISSING CELLS
                    </div>

                    <div class="kpi-value">
                        {analysis["missing_count"]:,}
                    </div>

                    <div class="kpi-caption">
                        Across all columns
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        DUPLICATES
                    </div>

                    <div class="kpi-value">
                        {analysis["duplicate_count"]:,}
                    </div>

                    <div class="kpi-caption">
                        Duplicate records detected
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        ML ANOMALIES
                    </div>

                    <div class="kpi-value">
                        {analysis["ml_anomaly_count"]:,}
                    </div>

                    <div class="kpi-caption">
                        {contamination_pct}% sensitivity
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Data Profile'
            '</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4, p5 = (
            st.columns(5)
        )

        with p1:
            st.metric(
                "Rows",
                f'{analysis["rows"]:,}',
            )

        with p2:
            st.metric(
                "Columns",
                analysis["columns"],
            )

        with p3:
            st.metric(
                "Numeric",
                analysis["numeric_count"],
            )

        with p4:
            st.metric(
                "Categorical",
                analysis["categorical_count"],
            )

        with p5:
            st.metric(
                "IQR Outliers",
                f'{analysis["iqr_outlier_count"]:,}',
            )


        # ----------------------------------------------------
        # QUALITY MONITORING
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Quality Monitoring'
            '</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(
            [1.3, 1]
        )

        with left:

            quality_data = pd.DataFrame(
                {
                    "Metric": [
                        "Missing",
                        "Duplicates",
                        "Invalid Values",
                        "IQR Outliers",
                        "ML Anomalies",
                    ],
                    "Count": [
                        analysis[
                            "missing_count"
                        ],
                        analysis[
                            "duplicate_count"
                        ],
                        analysis[
                            "invalid_count"
                        ],
                        analysis[
                            "iqr_outlier_count"
                        ],
                        analysis[
                            "ml_anomaly_count"
                        ],
                    ],
                }
            )

            st.bar_chart(
                quality_data.set_index(
                    "Metric"
                )
            )

        with right:

            st.markdown(
                """
                <div class="dashboard-panel">

                    <div class="panel-title">
                        Dataset Status
                    </div>
                """,
                unsafe_allow_html=True,
            )

            if score >= 95:

                st.markdown(
                    '<span class="status-good">'
                    'GOOD DATA QUALITY'
                    '</span>',
                    unsafe_allow_html=True,
                )

            elif score >= 85:

                st.markdown(
                    '<span class="status-warning">'
                    'NEEDS ATTENTION'
                    '</span>',
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    '<span class="status-danger">'
                    'CRITICAL'
                    '</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <p style="
                    color:#4b5563;
                    font-size:13px;
                    margin-top:12px;
                ">
                    DataGuard AI identified
                    <b>{analysis["missing_count"]:,}</b>
                    missing cells,
                    <b>{analysis["duplicate_count"]:,}</b>
                    duplicate records and
                    <b>{analysis["invalid_count"]:,}</b>
                    invalid values.
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


        # ----------------------------------------------------
        # BUSINESS PREVIEW
        # ----------------------------------------------------

        if business_df is not None:

            st.markdown(
                '<div class="section-title">'
                'Business Analytics Preview'
                '</div>',
                unsafe_allow_html=True,
            )

            sales_column = (
                analysis["sales_column"]
            )

            preview1, preview2 = (
                st.columns(2)
            )

            with preview1:

                st.markdown(
                    f"""
                    <div class="dashboard-panel">

                        <div class="panel-title">
                            Sales Distribution
                        </div>

                        <div class="panel-description">
                            Distribution of
                            {sales_column}
                            values used for visualization.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                chart_values = (
                    business_df[
                        sales_column
                    ]
                )

                hist, bins = np.histogram(
                    chart_values,
                    bins=10,
                )

                hist_df = pd.DataFrame(
                    {
                        "Sales Range": [
                            f"{bins[i]:,.0f} - "
                            f"{bins[i+1]:,.0f}"
                            for i in range(
                                len(bins) - 1
                            )
                        ],
                        "Records": hist,
                    }
                )

                st.bar_chart(
                    hist_df.set_index(
                        "Sales Range"
                    )
                )

            with preview2:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_column = (
                        city_columns[0]
                    )

                    city_counts = (
                        cleaned_df[
                            city_column
                        ]
                        .value_counts()
                        .head(10)
                    )

                    st.markdown(
                        """
                        <div class="dashboard-panel">

                            <div class="panel-title">
                                Records by City
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        city_counts
                    )


        # ----------------------------------------------------
        # DATA PREVIEW
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Data Preview'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Showing 10 of "
            f"{len(df):,} records"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=350,
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    elif page == "Data Quality":

        st.markdown(
            '<div class="section-title">'
            'Data Quality'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Detailed profiling and rule-based quality validation.'
            '</div>',
            unsafe_allow_html=True,
        )

        q1, q2, q3, q4 = (
            st.columns(4)
        )

        q1.metric(
            "Quality Score",
            f'{analysis["quality_score"]}/100',
        )

        q2.metric(
            "Missing Cells",
            f'{analysis["missing_count"]:,}',
        )

        q3.metric(
            "Duplicates",
            f'{analysis["duplicate_count"]:,}',
        )

        q4.metric(
            "Invalid Values",
            f'{analysis["invalid_count"]:,}',
        )

        st.markdown(
            '<div class="section-title">'
            'Data Profile'
            '</div>',
            unsafe_allow_html=True,
        )

        profile = profile_dataset(
            df
        )

        st.dataframe(
            profile,
            use_container_width=True,
            height=450,
        )

        st.markdown(
            '<div class="section-title">'
            'Detected Data Types'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        NUMERIC COLUMNS
                    </div>

                    <div class="kpi-value">
                        {analysis["numeric_count"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        CATEGORICAL COLUMNS
                    </div>

                    <div class="kpi-value">
                        {analysis["categorical_count"]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        DATETIME COLUMNS
                    </div>

                    <div class="kpi-value">
                        {len(analysis["datetime_columns"])}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander(
            "View detected Date/Time columns"
        ):

            if analysis[
                "datetime_columns"
            ]:

                st.write(
                    analysis[
                        "datetime_columns"
                    ]
                )

            else:

                st.info(
                    "No Date/Time columns detected."
                )

        with st.expander(
            "View detected Identifier columns"
        ):

            if analysis[
                "identifier_columns"
            ]:

                st.write(
                    analysis[
                        "identifier_columns"
                    ]
                )

            else:

                st.info(
                    "No Identifier columns detected."
                )

        st.markdown(
            '<div class="section-title">'
            'Invalid Values'
            '</div>',
            unsafe_allow_html=True,
        )

        if analysis[
            "invalid_details"
        ].empty:

            st.success(
                "No invalid values were detected."
            )

        else:

            st.dataframe(
                analysis[
                    "invalid_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-title">'
            'City Standardization'
            '</div>',
            unsafe_allow_html=True,
        )

        st.info(
            f'{analysis["city_changes"]:,} '
            "city values can be standardized "
            "during cleaning."
        )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        st.markdown(
            '<div class="section-title">'
            'Anomaly Detection'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Statistical outlier detection and '
            'machine-learning anomaly screening.'
            '</div>',
            unsafe_allow_html=True,
        )

        a1, a2, a3 = (
            st.columns(3)
        )

        a1.metric(
            "IQR Outlier Observations",
            f'{analysis["iqr_outlier_count"]:,}',
        )

        a2.metric(
            "ML Anomalies",
            f'{analysis["ml_anomaly_count"]:,}',
        )

        a3.metric(
            "Normal Records",
            f'{analysis["normal_count"]:,}',
        )

        st.markdown(
            """
            <div class="warning-box">

                <b>Important:</b>
                IQR outliers and ML anomalies are not
                automatically data errors.

                They identify statistically unusual
                observations that may require business validation.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            'IQR Statistical Outliers'
            '</div>',
            unsafe_allow_html=True,
        )

        if analysis[
            "iqr_details"
        ].empty:

            st.success(
                "No IQR outliers detected."
            )

        else:

            st.dataframe(
                analysis[
                    "iqr_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            '<div class="section-title">'
            'Isolation Forest'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="dashboard-panel">

                <div class="panel-title">
                    ML anomaly monitoring
                </div>

                <div class="panel-description">
                    Current sensitivity:
                    <b>{contamination_pct}%</b>
                </div>

                <p style="
                    font-size:14px;
                    color:#4b5563;
                ">
                    DataGuard AI detected
                    <b>{analysis["ml_anomaly_count"]:,}</b>
                    potentially unusual records and
                    <b>{analysis["normal_count"]:,}</b>
                    normal records.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        anomaly_display = (
            df.copy()
        )

        anomaly_display[
            "ML_Anomaly"
        ] = np.where(
            analysis[
                "anomaly_mask"
            ],
            "Anomaly",
            "Normal",
        )

        with st.expander(
            "View ML anomaly records"
        ):

            anomaly_records = (
                anomaly_display[
                    anomaly_display[
                        "ML_Anomaly"
                    ]
                    == "Anomaly"
                ]
            )

            st.caption(
                f"Showing up to 500 anomaly "
                f"records from "
                f"{len(anomaly_records):,} detected."
            )

            st.dataframe(
                anomaly_records.head(
                    500
                ),
                use_container_width=True,
                height=450,
            )


    # ========================================================
    # CLEANING
    # ========================================================

    elif page == "Cleaning":

        st.markdown(
            '<div class="section-title">'
            'Automated Cleaning'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Rule-based cleaning applied to create '
            'a Power BI-ready dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Original Rows",
            f'{len(df):,}',
        )

        c2.metric(
            "Cleaned Rows",
            f'{len(cleaned_df):,}',
        )

        c3.metric(
            "Rows Removed",
            f'{analysis["rows_removed"]:,}',
        )

        c4.metric(
            "Values Filled",
            f'{analysis["values_filled"]:,}',
        )

        st.markdown(
            """
            <div class="success-box">

                <b>Cleaning actions:</b>
                duplicate removal, city standardization,
                numeric missing-value imputation using
                median values, and categorical missing-value
                imputation using mode values.

            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis[
            "city_changes"
        ] > 0:

            st.info(
                f'{analysis["city_changes"]:,} '
                "city values were standardized."
            )

        st.markdown(
            '<div class="section-title">'
            'Cleaned Dataset Preview'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Showing 10 of "
            f"{len(cleaned_df):,} "
            "cleaned records"
        )

        st.dataframe(
            cleaned_df.head(10),
            use_container_width=True,
            height=350,
        )

        csv_data = (
            cleaned_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name="DataGuard_Cleaned_Data.csv",
            mime="text/csv",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "Analytics":

        st.markdown(
            '<div class="section-title">'
            'Business Analytics'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Business-focused views generated from '
            'the cleaned dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        if business_df is None:

            st.warning(
                "No Sales, Revenue, or Amount column "
                "was detected. Business sales charts "
                "are therefore unavailable."
            )

        else:

            sales_column = (
                analysis["sales_column"]
            )

            upper_bound = (
                analysis["upper_bound"]
            )

            st.markdown(
                f"""
                <div class="info-box">

                    Business charts use the cleaned dataset.

                    For visualization only, Sales values above
                    the cleaned-data IQR upper bound
                    <b>{upper_bound:,.2f}</b>
                    are excluded.

                    These records are
                    <b>not deleted</b>
                    from the analytical dataset.

                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-title">'
                'Sales Distribution'
                '</div>',
                unsafe_allow_html=True,
            )

            hist, bins = np.histogram(
                business_df[
                    sales_column
                ],
                bins=10,
            )

            hist_df = pd.DataFrame(
                {
                    "Sales Range": [
                        f"{bins[i]:,.0f} - "
                        f"{bins[i+1]:,.0f}"
                        for i in range(
                            len(bins) - 1
                        )
                    ],
                    "Records": hist,
                }
            )

            st.bar_chart(
                hist_df.set_index(
                    "Sales Range"
                )
            )

            left, right = (
                st.columns(2)
            )

            with left:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_column = (
                        city_columns[0]
                    )

                    city_counts = (
                        cleaned_df[
                            city_column
                        ]
                        .value_counts()
                        .head(10)
                    )

                    st.markdown(
                        '<div class="section-title">'
                        'Records by City'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        city_counts
                    )

            with right:

                product_columns = [
                    column
                    for column in cleaned_df.columns
                    if "product"
                    in str(column).lower()
                ]

                if product_columns:

                    product_column = (
                        product_columns[0]
                    )

                    product_sales = (
                        business_df
                        .groupby(
                            product_column
                        )[sales_column]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                    )

                    st.markdown(
                        '<div class="section-title">'
                        'Sales by Product'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        product_sales
                    )

            category_columns = [
                column
                for column in cleaned_df.columns
                if "category"
                in str(column).lower()
            ]

            if category_columns:

                category_column = (
                    category_columns[0]
                )

                category_sales = (
                    business_df
                    .groupby(
                        category_column
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    '<div class="section-title">'
                    'Sales by Category'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.bar_chart(
                    category_sales
                )

            datetime_columns = (
                analysis[
                    "datetime_columns"
                ]
            )

            if datetime_columns:

                date_column = (
                    datetime_columns[0]
                )

                monthly_df = (
                    cleaned_df.copy()
                )

                monthly_df[
                    date_column
                ] = safe_to_datetime(
                    monthly_df[
                        date_column
                    ]
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
                    monthly_df.dropna(
                        subset=[
                            date_column,
                            sales_column,
                        ]
                    )
                )

                monthly_sales = (
                    monthly_df
                    .assign(
                        Month=(
                            monthly_df[
                                date_column
                            ]
                            .dt
                            .to_period("M")
                            .astype(str)
                        )
                    )
                    .groupby(
                        "Month"
                    )[sales_column]
                    .sum()
                )

                st.markdown(
                    '<div class="section-title">'
                    'Monthly Sales Trend'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.line_chart(
                    monthly_sales
                )


    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        st.markdown(
            '<div class="section-title">'
            'Power BI Export Center'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Download cleaned and supporting CSV datasets '
            'ready for Power BI Desktop.'
            '</div>',
            unsafe_allow_html=True,
        )

        exports = (
            create_powerbi_exports(
                cleaned_df,
                business_df,
                analysis[
                    "sales_column"
                ],
            )
        )

        st.markdown(
            """
            <div class="info-box">

                <b>Manual Power BI workflow</b><br><br>

                1. Download the ZIP file.<br>
                2. Extract the CSV files.<br>
                3. Open Power BI Desktop.<br>
                4. Select Get Data → Text/CSV.<br>
                5. Import the cleaned dataset and supporting tables.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            'Available Exports'
            '</div>',
            unsafe_allow_html=True,
        )

        for filename, data in (
            exports.items()
        ):

            left, right = (
                st.columns(
                    [4, 1]
                )
            )

            with left:

                st.markdown(
                    f"""
                    <div class="dashboard-panel">

                        <div class="panel-title">
                            {filename}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:

                st.download_button(
                    "Download",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=(
                        f"download_{filename}"
                    ),
                )

        zip_data = create_zip(
            exports
        )

        st.markdown(
            '<div class="section-title">'
            'Complete Export Package'
            '</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            "📦 Download Power BI ZIP",
            data=zip_data,
            file_name=(
                "DataGuard_PowerBI_Exports.zip"
            ),
            mime="application/zip",
            type="primary",
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif page == "AI Analysis":

        st.markdown(
            '<div class="section-title">'
            'Gemini AI Analysis'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'AI-generated interpretation based only on '
            'the DataGuard quality report.'
            '</div>',
            unsafe_allow_html=True,
        )

        summary_text = f"""
DataGuard AI Data Quality Report

Dataset:
{st.session_state.file_name}

Rows:
{analysis["rows"]}

Columns:
{analysis["columns"]}

Total cells:
{analysis["total_cells"]}

Missing cells:
{analysis["missing_count"]}

Duplicate records:
{analysis["duplicate_count"]}

Invalid values:
{analysis["invalid_count"]}

Quality score:
{analysis["quality_score"]}/100

Numeric columns:
{analysis["numeric_count"]}

Categorical columns:
{analysis["categorical_count"]}

Date/Time columns:
{analysis["datetime_columns"]}

Identifier columns:
{analysis["identifier_columns"]}

IQR outlier observations:
{analysis["iqr_outlier_count"]}

Isolation Forest sensitivity:
{contamination_pct}%

Isolation Forest anomalies:
{analysis["ml_anomaly_count"]}

Normal records:
{analysis["normal_count"]}

Original rows:
{analysis["rows"]}

Cleaned rows:
{len(cleaned_df)}

Rows removed:
{analysis["rows_removed"]}

Values filled:
{analysis["values_filled"]}

City values standardized:
{analysis["city_changes"]}

Sales column:
{analysis["sales_column"]}

Business chart IQR upper bound:
{analysis["upper_bound"]}

IQR outliers and ML anomalies are screening
signals, not automatically confirmed errors.

Business visualization filtering does not delete
records from the analytical dataset.
"""

        if not GEMINI_API_KEY:

            st.warning(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to Streamlit secrets "
                "to enable AI analysis."
            )

        if st.button(
            "✨ Generate Gemini AI Analysis",
            type="primary",
        ):

            with st.spinner(
                "Gemini is analyzing the report..."
            ):

                result = (
                    get_gemini_analysis(
                        summary_text
                    )
                )

            st.session_state.gemini_analysis = (
                result
            )

        if (
            st.session_state.gemini_analysis
        ):

            st.markdown(
                """
                <div class="dashboard-panel">
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        else:

            st.info(
                "Click Generate Gemini AI Analysis "
                "to request an AI-generated interpretation."
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.markdown(
            '<div class="section-title">'
            'Data Quality Report'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-description">'
            'Portfolio-ready summary of the DataGuard AI analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

        report_text = f"""
DATAGUARD AI — DATA QUALITY REPORT

Dataset
-------
File: {st.session_state.file_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET OVERVIEW
----------------
Rows: {analysis["rows"]:,}
Columns: {analysis["columns"]}
Total Cells: {analysis["total_cells"]:,}

DATA QUALITY
------------
Quality Score: {analysis["quality_score"]}/100
Missing Cells: {analysis["missing_count"]:,}
Duplicate Records: {analysis["duplicate_count"]:,}
Invalid Values: {analysis["invalid_count"]:,}

DATA PROFILE
------------
Numeric Columns: {analysis["numeric_count"]}
Categorical Columns: {analysis["categorical_count"]}
Date/Time Columns: {len(analysis["datetime_columns"])}
Identifier Columns: {len(analysis["identifier_columns"])}

ANOMALY DETECTION
-----------------
IQR Outlier Observations: {analysis["iqr_outlier_count"]:,}
Isolation Forest Sensitivity: {contamination_pct}%
ML Anomalies: {analysis["ml_anomaly_count"]:,}
Normal Records: {analysis["normal_count"]:,}

CLEANING RESULTS
----------------
Original Rows: {analysis["rows"]:,}
Cleaned Rows: {len(cleaned_df):,}
Rows Removed: {analysis["rows_removed"]:,}
Values Filled: {analysis["values_filled"]:,}
City Values Standardized: {analysis["city_changes"]:,}

BUSINESS ANALYTICS
------------------
Sales Column: {analysis["sales_column"]}
Business Visualization IQR Upper Bound: {analysis["upper_bound"]}

IMPORTANT INTERPRETATION
------------------------
IQR outliers are statistically unusual observations.

Isolation Forest anomalies are screening signals.

Neither should automatically be treated as confirmed errors.

Extreme sales values excluded from business charts
are excluded only for visualization and are not deleted
from the analytical dataset.
"""

        st.code(
            report_text,
            language="text",
        )

        st.download_button(
            "📄 Download Report",
            data=report_text.encode(
                "utf-8"
            ),
            file_name=(
                "DataGuard_Data_Quality_Report.txt"
            ),
            mime="text/plain",
        )

        if (
            st.session_state.gemini_analysis
        ):

            st.markdown(
                '<div class="section-title">'
                'Gemini Analysis'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )


# ============================================================
# NO DATA STATE
# ============================================================

else:

    st.markdown(
        """
        <div class="dashboard-panel"
             style="
                text-align:center;
                padding:55px 20px;
             ">

            <div style="
                font-size:48px;
            ">
                🛡️
            </div>

            <div style="
                font-size:24px;
                font-weight:800;
                color:#111827;
                margin-top:10px;
            ">
                Welcome to DataGuard AI
            </div>

            <div style="
                color:#6b7280;
                font-size:14px;
                max-width:600px;
                margin:10px auto;
            ">
                Upload a CSV or Excel dataset to automatically
                profile data quality, detect anomalies, clean
                data and prepare business analytics.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🛡️ DataGuard AI
        &nbsp; • &nbsp;
        AI Data Quality & Anomaly Detection Platform

        <br>

        Built with Python, Pandas, Scikit-learn and Streamlit

    </div>
    """,
    unsafe_allow_html=True,
)

