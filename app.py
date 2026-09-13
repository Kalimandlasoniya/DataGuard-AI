import io
import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")


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
        background: #f4f7fb;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #101828;
    }

    p {
        color: #475467;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0b1220;
    }

    section[data-testid="stSidebar"] > div {
        background: #0b1220;
    }

    section[data-testid="stSidebar"] * {
        color: #e6edf7;
    }

    .sidebar-logo {
        font-size: 25px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        line-height: 1.5;
        color: #98a2b3;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #667085;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .pipeline-item {
        background: #111b2e;
        border: 1px solid #1d2939;
        border-radius: 9px;
        padding: 9px 12px;
        margin-bottom: 7px;
        font-size: 13px;
    }

    .pipeline-number {
        color: #60a5fa;
        font-weight: 800;
    }

    .technology-box {
        background: #111b2e;
        border: 1px solid #1d2939;
        border-radius: 10px;
        padding: 12px;
        font-size: 12px;
        line-height: 1.6;
        color: #98a2b3;
    }

    /* ---------- HERO ---------- */

    .hero {
        background:
            radial-gradient(
                circle at top right,
                rgba(96,165,250,0.30),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 55%,
                #1d4ed8 100%
            );

        border-radius: 22px;
        padding: 42px 45px;
        margin-bottom: 25px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 850;
        line-height: 1.1;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        color: #dbeafe;
        font-size: 17px;
        line-height: 1.7;
        max-width: 850px;
        margin-bottom: 22px;
    }

    .hero-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);
        color: #dbeafe;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.7px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.05);
        height: 100%;
    }

    .feature-card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 16px;
        padding: 24px;
        min-height: 190px;
        box-shadow: 0 4px 14px rgba(16, 24, 40, 0.05);
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #101828;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #667085;
        font-size: 14px;
        line-height: 1.65;
    }

    .section-title {
        color: #101828;
        font-size: 22px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* ---------- KPI ---------- */

    .kpi-card {
        background: white;
        border: 1px solid #e4e7ec;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.05);
        min-height: 115px;
    }

    .kpi-label {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }

    .kpi-value {
        color: #101828;
        font-size: 28px;
        font-weight: 850;
        margin-top: 8px;
    }

    /* ---------- STATUS ---------- */

    .status-card {
        border-radius: 14px;
        padding: 18px;
        background: white;
        border: 1px solid #e4e7ec;
        box-shadow: 0 3px 12px rgba(16, 24, 40, 0.04);
    }

    .status-title {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
    }

    .status-value {
        color: #101828;
        font-size: 22px;
        font-weight: 800;
        margin-top: 6px;
    }

    /* ---------- MINI CARDS ---------- */

    .mini-card {
        background: #f8fafc;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 16px;
        min-height: 125px;
    }

    .mini-title {
        color: #101828;
        font-weight: 800;
        font-size: 15px;
        margin-bottom: 7px;
    }

    .mini-text {
        color: #667085;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #b2ddff;
        border-radius: 16px;
        padding: 10px;
    }

    /* ---------- BUTTON ---------- */

    .stDownloadButton button {
        background: #175cd3;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
    }

    /* ---------- TABS ---------- */

    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        padding: 30px 10px 10px;
        color: #98a2b3;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_to_datetime(series):
    """
    Safely convert a Series to datetime without allowing
    malformed values to crash the application.
    """
    try:
        try:
            return pd.to_datetime(
                series,
                errors="coerce",
                format="mixed"
            )
        except Exception:
            return pd.to_datetime(
                series,
                errors="coerce"
            )
    except Exception:
        return pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]"
        )


def detect_datetime_columns(df):
    """
    Detect real datetime columns.

    Important:
    Numeric IDs such as Customer_ID are NOT treated as dates.
    """

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
        "day",
    )

    for column in df.columns:

        series = df[column]

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Never interpret numeric columns as dates
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Only examine object/string columns
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        converted = safe_to_datetime(series)

        if len(series) == 0:
            continue

        valid_ratio = converted.notna().mean()

        if valid_ratio < 0.85:
            continue

        name = str(column).lower()

        sample = series.dropna().astype(str).head(50)

        if len(sample) == 0:
            continue

        looks_date_like = (
            sample.str.contains(
                r"[-/:]",
                regex=True
            ).mean() >= 0.50
        )

        if (
            any(keyword in name for keyword in date_keywords)
            or looks_date_like
        ):
            datetime_columns.append(column)

    return datetime_columns


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()


def detect_invalid_values(df):
    """
    Detect common invalid values in columns such as:
    Age, Quantity, Sales, Discount, etc.
    """

    invalid_count = 0
    invalid_details = []

    for column in df.columns:

        name = str(column).lower()

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        mask = pd.Series(False, index=df.index)

        if "age" in name:
            mask = (series < 0) | (series > 120)

        elif "quantity" in name:
            mask = series < 0

        elif "discount" in name:
            mask = (series < 0) | (series > 1)

        elif "price" in name or "sales" in name or "amount" in name:
            mask = series < 0

        count = int(mask.sum())

        if count > 0:
            invalid_count += count

            invalid_details.append(
                {
                    "Column": column,
                    "Invalid Values": count,
                }
            )

    return invalid_count, pd.DataFrame(invalid_details)


CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru",
}


def detect_city_issues(df):
    """
    Detect non-standard city labels.
    """

    city_columns = [
        col
        for col in df.columns
        if "city" in str(col).lower()
    ]

    if not city_columns:
        return 0, pd.DataFrame()

    column = city_columns[0]

    series = df[column].astype("string")

    issue_mask = series.isin(
        [
            "Bangalore",
            "bangalore",
            "BLR",
            "BENGALURU",
        ]
    )

    count = int(issue_mask.sum())

    details = pd.DataFrame(
        {
            "Original Value": series[issue_mask]
            .value_counts()
            .index,
            "Standard Value": [
                CITY_MAPPING.get(
                    value,
                    value
                )
                for value in series[issue_mask]
                .value_counts()
                .index
            ],
            "Rows": series[issue_mask]
            .value_counts()
            .values,
        }
    )

    return count, details


def iqr_outliers(df):
    """
    Detect IQR-based statistical outliers.
    """

    results = []

    for column in numeric_columns(df):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (series < lower)
            | (series > upper)
        )

        count = int(mask.sum())

        results.append(
            {
                "Column": column,
                "Outliers": count,
                "Lower Bound": lower,
                "Upper Bound": upper,
            }
        )

    return pd.DataFrame(results)


def ml_anomalies(df, contamination=0.05):
    """
    Isolation Forest anomaly detection.

    IDs are excluded because identifiers should not normally
    determine whether a record is anomalous.
    """

    nums = df.select_dtypes(
        include=np.number
    ).copy()

    if nums.empty:
        return pd.Series(
            False,
            index=df.index
        )

    # Remove likely ID columns
    remove_columns = []

    for column in nums.columns:

        name = str(column).lower()

        if (
            name.endswith("id")
            or name.startswith("id_")
            or name == "id"
            or "customer_id" in name
            or "order_id" in name
            or "product_id" in name
        ):
            remove_columns.append(column)

    nums = nums.drop(
        columns=remove_columns,
        errors="ignore"
    )

    if nums.empty:
        return pd.Series(
            False,
            index=df.index
        )

    nums = nums.replace(
        [np.inf, -np.inf],
        np.nan
    )

    nums = nums.fillna(
        nums.median(numeric_only=True)
    )

    nums = nums.fillna(0)

    if len(nums) < 10:
        return pd.Series(
            False,
            index=df.index
        )

    try:

        model = IsolationForest(
            contamination=float(contamination),
            random_state=42,
            n_estimators=200,
            n_jobs=-1,
        )

        predictions = model.fit_predict(nums)

        return pd.Series(
            predictions == -1,
            index=df.index
        )

    except Exception:
        return pd.Series(
            False,
            index=df.index
        )


def quality_score(
    df,
    invalid_count,
    city_rows,
    outlier_rows
):
    """
    Quality score based on confirmed/statistical issues.

    ML anomaly flags are intentionally NOT included because
    an ML flag is a screening signal, not proof of bad data.
    """

    total_cells = max(
        df.shape[0] * df.shape[1],
        1
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    total_rows = max(
        len(df),
        1
    )

    missing_rate = missing_cells / total_cells
    duplicate_rate = duplicates / total_rows
    invalid_rate = invalid_count / total_cells
    city_rate = city_rows / total_rows
    outlier_rate = (
        outlier_rows / total_rows
        if total_rows
        else 0
    )

    score = 100

    score -= min(
        missing_rate * 25,
        25
    )

    score -= min(
        duplicate_rate * 20,
        20
    )

    score -= min(
        invalid_rate * 25,
        25
    )

    score -= min(
        city_rate * 15,
        15
    )

    score -= min(
        outlier_rate * 15,
        15
    )

    score = max(
        0,
        min(
            100,
            score
        )
    )

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 60:
        status = "Needs Review"
    else:
        status = "Critical"

    return round(score, 1), status


def clean_data(df):
    """
    Automated cleaning:
    - Standardize known city labels
    - Replace invalid numeric values with NaN
    - Fill numeric missing values with median
    - Fill categorical missing values with mode
    - Fill datetime missing values with mode
    - Remove duplicate rows
    """

    cleaned = df.copy()

    # --------------------------------
    # City standardization
    # --------------------------------

    for column in cleaned.columns:

        if "city" in str(column).lower():

            cleaned[column] = (
                cleaned[column]
                .replace(CITY_MAPPING)
            )

    # --------------------------------
    # Numeric columns
    # --------------------------------

    for column in cleaned.columns:

        if pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):

            # Convert to float first.
            # Prevents:
            # TypeError: Invalid value '23.5'
            # for dtype Int64

            cleaned[column] = (
                pd.to_numeric(
                    cleaned[column],
                    errors="coerce"
                )
                .astype("float64")
            )

            name = str(column).lower()

            invalid_mask = pd.Series(
                False,
                index=cleaned.index
            )

            if "age" in name:

                invalid_mask = (
                    (cleaned[column] < 0)
                    | (cleaned[column] > 120)
                )

            elif "quantity" in name:

                invalid_mask = (
                    cleaned[column] < 0
                )

            elif "discount" in name:

                invalid_mask = (
                    (cleaned[column] < 0)
                    | (cleaned[column] > 1)
                )

            elif (
                "price" in name
                or "sales" in name
                or "amount" in name
            ):

                invalid_mask = (
                    cleaned[column] < 0
                )

            cleaned.loc[
                invalid_mask,
                column
            ] = np.nan

            median_value = cleaned[column].median()

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

    # --------------------------------
    # Categorical columns
    # --------------------------------

    for column in categorical_columns(
        cleaned
    ):

        mode = cleaned[column].mode(
            dropna=True
        )

        if not mode.empty:

            cleaned[column] = (
                cleaned[column]
                .fillna(mode.iloc[0])
            )
        else:

            cleaned[column] = (
                cleaned[column]
                .fillna("Unknown")
            )

    # --------------------------------
    # Datetime columns
    # --------------------------------

    datetime_cols = detect_datetime_columns(
        cleaned
    )

    for column in datetime_cols:

        converted = safe_to_datetime(
            cleaned[column]
        )

        if converted.notna().any():

            mode = converted.mode(
                dropna=True
            )

            if not mode.empty:

                converted = converted.fillna(
                    mode.iloc[0]
                )

        cleaned[column] = converted

    # --------------------------------
    # Remove duplicates
    # --------------------------------

    cleaned = cleaned.drop_duplicates()

    return cleaned


def create_data_dictionary(df):
    rows = []

    for column in df.columns:

        rows.append(
            {
                "Column": column,
                "Data Type": str(
                    df[column].dtype
                ),
                "Non-Null Count": int(
                    df[column].notna().sum()
                ),
                "Missing Count": int(
                    df[column].isna().sum()
                ),
                "Unique Values": int(
                    df[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def create_quality_summary(
    df,
    invalid_count,
    duplicate_count,
    city_rows,
    outlier_rows,
    ml_count,
    score,
    status,
):
    return pd.DataFrame(
        [
            {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing Cells": int(
                    df.isna().sum().sum()
                ),
                "Duplicate Rows": duplicate_count,
                "Invalid Values": invalid_count,
                "City Inconsistencies": city_rows,
                "IQR Outlier Records": outlier_rows,
                "ML Anomaly Flags": ml_count,
                "Quality Score": score,
                "Quality Status": status,
            }
        ]
    )


def create_numeric_summary(df):

    nums = df.select_dtypes(
        include=np.number
    )

    if nums.empty:
        return pd.DataFrame()

    return nums.describe().T.reset_index().rename(
        columns={
            "index": "Column"
        }
    )


def create_missing_summary(df):

    result = (
        df.isna()
        .sum()
        .reset_index()
    )

    result.columns = [
        "Column",
        "Missing Count"
    ]

    result["Missing %"] = (
        result["Missing Count"]
        / max(len(df), 1)
        * 100
    )

    return result.sort_values(
        "Missing Count",
        ascending=False
    )


def dataframe_to_csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")


def create_zip(files):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for filename, data in files.items():

            z.writestr(
                filename,
                data
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-logo">
            🛡️ DataGuard AI
        </div>

        <div class="sidebar-subtitle">
            Data quality monitoring<br>
            & anomaly detection
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Upload</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-section">ML Settings</div>',
        unsafe_allow_html=True,
    )

    contamination_percent = st.slider(
        "Anomaly sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "Expected percentage of records "
            "flagged for ML review."
        ),
    )

    st.caption(
        f"Current setting: {contamination_percent}%"
    )

    st.caption(
        "This is a screening sensitivity setting, "
        "not proof that flagged records are incorrect."
    )

    st.markdown(
        '<div class="sidebar-section">Pipeline</div>',
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        ("01", "Upload"),
        ("02", "Profile"),
        ("03", "Detect issues"),
        ("04", "Analyze anomalies"),
        ("05", "Clean data"),
        ("06", "Export for Power BI"),
    ]

    for number, step in pipeline_steps:

        st.markdown(
            f"""
            <div class="pipeline-item">
                <span class="pipeline-number">● {number}</span>
                &nbsp; {step}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section">Technology</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="technology-box">
            Python • Pandas • Scikit-learn • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LANDING HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🛡️ DataGuard AI
        </div>

        <div class="hero-subtitle">
            Turn raw data into trusted insights with
            AI-powered data quality monitoring,
            anomaly detection and intelligent data cleaning.
        </div>

        <div class="hero-badge">
            DATA QUALITY • ANOMALY DETECTION • AI ANALYSIS • POWER BI
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NO FILE UPLOADED
# ============================================================

if uploaded_file is None:

    st.markdown(
        '<div class="section-title">🚀 Start Analyzing Your Data</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a CSV or Excel file from the sidebar to "
        "automatically profile, validate, detect anomalies, "
        "clean your data and prepare Power BI-ready reports."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🔍
                </div>

                <div class="feature-title">
                    Detect Data Issues
                </div>

                <div class="feature-text">
                    Find missing values, duplicates,
                    invalid values, inconsistent categories
                    and statistical outliers.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🚨
                </div>

                <div class="feature-title">
                    Detect Anomalies
                </div>

                <div class="feature-text">
                    Use Isolation Forest to identify unusual
                    records that deserve further investigation.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    📦
                </div>

                <div class="feature-title">
                    Clean & Export
                </div>

                <div class="feature-text">
                    Automatically clean the dataset and create
                    Power BI-ready CSV reports.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card" style="margin-top:25px;">

            <h3>
                🚀 From Raw Data to Analysis-Ready Data
            </h3>

            <p style="line-height:1.7;">
                DataGuard AI provides an end-to-end data quality
                workflow:
                <strong>
                    profiling → validation → anomaly detection →
                    cleaning → business insights → Power BI export.
                </strong>
            </p>

            <div style="
                display:grid;
                grid-template-columns:repeat(4,1fr);
                gap:12px;
                margin-top:18px;
            ">

                <div class="mini-card">

                    <div class="mini-title">
                        🔎 Profile
                    </div>

                    <div class="mini-text">
                        Understand your dataset structure
                        and quality.
                    </div>

                </div>

                <div class="mini-card">

                    <div class="mini-title">
                        🚨 Detect
                    </div>

                    <div class="mini-text">
                        Find quality issues and
                        unusual records.
                    </div>

                </div>

                <div class="mini-card">

                    <div class="mini-title">
                        🧹 Clean
                    </div>

                    <div class="mini-text">
                        Prepare reliable and
                        analysis-ready data.
                    </div>

                </div>

                <div class="mini-card">

                    <div class="mini-title">
                        📊 Export
                    </div>

                    <div class="mini-text">
                        Create Power BI-ready files
                        and reports.
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer">
            🛡️ DataGuard AI • Built with Python,
            Pandas, Scikit-learn & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

    elif file_name.endswith(".xlsx"):

        df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

    elif file_name.endswith(".xls"):

        df = pd.read_excel(
            uploaded_file,
            engine="xlrd"
        )

    else:

        st.error(
            "Unsupported file format."
        )

        st.stop()

except Exception as e:

    st.error(
        f"Unable to read the uploaded file: {e}"
    )

    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

rows = len(df)
columns = len(df.columns)

missing_cells = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

numeric_cols = numeric_columns(df)
categorical_cols = categorical_columns(df)

datetime_cols = detect_datetime_columns(df)

invalid_count, invalid_details = (
    detect_invalid_values(df)
)

city_rows, city_details = (
    detect_city_issues(df)
)

outlier_df = iqr_outliers(df)

outlier_rows = int(
    outlier_df["Outliers"].sum()
) if not outlier_df.empty else 0

anomaly_flags = ml_anomalies(
    df,
    contamination=contamination_percent / 100
)

ml_count = int(
    anomaly_flags.sum()
)

score, quality_status = quality_score(
    df,
    invalid_count,
    city_rows,
    outlier_rows,
)


# ============================================================
# HERO AFTER UPLOAD
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <div class="hero-title">
            🛡️ DataGuard AI
        </div>

        <div class="hero-subtitle">
            Dataset:
            <strong>{uploaded_file.name}</strong>
            <br>
            Automated data profiling, quality validation,
            anomaly detection and cleaning.
        </div>

        <div class="hero-badge">
            DATASET LOADED • ANALYSIS READY
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI ROW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

kpis = [
    ("Rows", f"{rows:,}"),
    ("Columns", f"{columns:,}"),
    ("Missing", f"{missing_cells:,}"),
    ("Duplicates", f"{duplicate_count:,}"),
    ("Quality", f"{score}/100"),
    ("ML Flags", f"{ml_count:,}"),
]

for col, (label, value) in zip(
    [k1, k2, k3, k4, k5, k6],
    kpis
):

    with col:

        st.markdown(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    {label}
                </div>

                <div class="kpi-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# STATUS ROW
# ============================================================

st.markdown(
    '<div class="section-title">🩺 Quality Status</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

status_values = [
    (
        "Overall Quality",
        quality_status,
    ),
    (
        "Invalid Values",
        f"{invalid_count:,}",
    ),
    (
        "Statistical Outliers",
        f"{outlier_rows:,}",
    ),
    (
        "ML Review Flags",
        f"{ml_count:,}",
    ),
]

for col, (label, value) in zip(
    [s1, s2, s3, s4],
    status_values
):

    with col:

        st.markdown(
            f"""
            <div class="status-card">

                <div class="status-title">
                    {label}
                </div>

                <div class="status-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "📊 Overview",
        "🔍 Data Quality",
        "🚨 Anomalies",
        "📈 Insights",
        "🧹 Cleaning",
        "🤖 AI Analysis",
        "📦 Power BI",
        "📄 Report",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">👀 Data Preview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=420,
    )

    st.markdown(
        '<div class="section-title">📐 Dataset Structure</div>',
        unsafe_allow_html=True,
    )

    structure = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[col].dtype)
                for col in df.columns
            ],
            "Missing": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique": [
                int(df[col].nunique())
                for col in df.columns
            ],
        }
    )

    st.dataframe(
        structure,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATA QUALITY TAB
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">🔍 Data Quality Analysis</div>',
        unsafe_allow_html=True,
    )

    q1, q2 = st.columns(2)

    with q1:

        st.markdown(
            "### Missing Values"
        )

        missing_summary = create_missing_summary(
            df
        )

        st.dataframe(
            missing_summary,
            use_container_width=True,
            hide_index=True,
        )

    with q2:

        st.markdown(
            "### Duplicate Records"
        )

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count:,} duplicate rows detected."
            )

            st.dataframe(
                df[
                    df.duplicated(
                        keep=False
                    )
                ].head(100),
                use_container_width=True,
            )

        else:

            st.success(
                "No duplicate rows detected."
            )

    st.markdown(
        "### ⚠️ Invalid Values"
    )

    if invalid_details.empty:

        st.success(
            "No common invalid numeric values detected."
        )

    else:

        st.dataframe(
            invalid_details,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### 🏙️ Category Consistency"
    )

    if city_details.empty:

        st.info(
            "No City column with known inconsistent labels was detected."
        )

    else:

        st.dataframe(
            city_details,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### 📊 IQR Outliers"
    )

    if outlier_df.empty:

        st.info(
            "No numeric columns were suitable for IQR analysis."
        )

    else:

        st.dataframe(
            outlier_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ANOMALIES TAB
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">🚨 Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"""
        Isolation Forest is currently configured to flag approximately
        {contamination_percent}% of records for review.
        These are screening flags, not confirmed data errors.
        """
    )

    anomaly_df = df.copy()

    anomaly_df["ML_Anomaly"] = anomaly_flags.map(
        {
            True: "Review",
            False: "Normal",
        }
    )

    flagged = anomaly_df[
        anomaly_df["ML_Anomaly"] == "Review"
    ]

    a1, a2 = st.columns(2)

    with a1:

        st.metric(
            "Records flagged",
            f"{len(flagged):,}"
        )

    with a2:

        st.metric(
            "Flag rate",
            f"{(len(flagged) / max(len(df), 1)) * 100:.1f}%"
        )

    st.markdown(
        "### 🚨 Flagged Records"
    )

    if flagged.empty:

        st.success(
            "No records were flagged."
        )

    else:

        st.dataframe(
            flagged.head(200),
            use_container_width=True,
            height=450,
        )

    st.caption(
        "Isolation Forest identifies records that look unusual "
        "relative to the numeric patterns in the dataset. "
        "A flag does not automatically mean the record is wrong."
    )


# ============================================================
# INSIGHTS TAB
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">📈 Business Insights</div>',
        unsafe_allow_html=True,
    )

    nums = df.select_dtypes(
        include=np.number
    )

    # --------------------------------
    # Numeric distribution
    # --------------------------------

    if not nums.empty:

        st.markdown(
            "### 📊 Numeric Distribution"
        )

        numeric_choice = st.selectbox(
            "Select numeric column",
            nums.columns.tolist(),
            key="numeric_distribution",
        )

        values = pd.to_numeric(
            df[numeric_choice],
            errors="coerce"
        ).dropna()

        if not values.empty:

            counts, bins = np.histogram(
                values,
                bins=20
            )

            labels = [
                f"{bins[i]:.2f}"
                for i in range(len(counts))
            ]

            chart_df = pd.DataFrame(
                {
                    "Range": labels,
                    "Records": counts,
                }
            )

            chart_df = chart_df.set_index(
                "Range"
            )

            st.bar_chart(
                chart_df,
                use_container_width=True,
            )

    # --------------------------------
    # Sales by category
    # --------------------------------

    sales_column = next(
        (
            col
            for col in df.columns
            if str(col).lower() == "sales"
            or "sales" in str(col).lower()
        ),
        None,
    )

    category_column = next(
        (
            col
            for col in df.columns
            if str(col).lower() == "category"
        ),
        None,
    )

    if (
        sales_column is not None
        and category_column is not None
    ):

        st.markdown(
            "### 💰 Sales by Category"
        )

        sales_category = (
            df.groupby(
                category_column,
                dropna=False
            )[sales_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            sales_category,
            use_container_width=True,
        )

    # --------------------------------
    # Date trend
    # --------------------------------

    if datetime_cols:

        st.markdown(
            "### 📅 Time Trend"
        )

        selected_date = st.selectbox(
            "Select date column",
            datetime_cols,
            key="trend_date_column",
        )

        trend_df = df.copy()

        trend_df["_Date"] = safe_to_datetime(
            trend_df[selected_date]
        )

        if sales_column is not None:

            trend_df[sales_column] = pd.to_numeric(
                trend_df[sales_column],
                errors="coerce"
            )

            valid = trend_df.dropna(
                subset=[
                    "_Date",
                    sales_column
                ]
            ).copy()

            if not valid.empty:

                valid["Date"] = (
                    valid["_Date"].dt.date
                )

                trend = (
                    valid.groupby("Date")[
                        sales_column
                    ]
                    .sum()
                )

                st.line_chart(
                    trend,
                    use_container_width=True,
                )

        else:

            st.info(
                "A Sales column was not detected, "
                "so a sales trend cannot be calculated."
            )

    else:

        st.info(
            "No reliable date/time column was detected."
        )


# ============================================================
# CLEANING TAB
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">🧹 Automated Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The cleaning pipeline standardizes known city labels,
        handles invalid numeric values, fills missing values,
        handles datetime values and removes duplicate rows.
        """
    )

    cleaned_df = clean_data(df)

    before_missing = int(
        df.isna().sum().sum()
    )

    after_missing = int(
        cleaned_df.isna().sum().sum()
    )

    before_rows = len(df)
    after_rows = len(cleaned_df)

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (
            "Rows Before",
            f"{before_rows:,}"
        ),
        (
            "Rows After",
            f"{after_rows:,}"
        ),
        (
            "Missing Before",
            f"{before_missing:,}"
        ),
        (
            "Missing After",
            f"{after_missing:,}"
        ),
    ]

    for col, (label, value) in zip(
        [c1, c2, c3, c4],
        metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {value}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "### 🧹 Cleaned Data Preview"
    )

    st.dataframe(
        cleaned_df.head(100),
        use_container_width=True,
        height=420,
    )

    cleaned_csv = dataframe_to_csv_bytes(
        cleaned_df
    )

    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
    )


# ============================================================
# AI ANALYSIS TAB
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">🤖 AI Data Quality Analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        Gemini can generate an executive-style interpretation
        of the detected data-quality and anomaly metrics.
        """
    )

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Your API key is used only for this analysis session.",
    )

    if st.button(
        "🤖 Generate AI Analysis",
        type="primary",
    ):

        if not api_key:

            st.warning(
                "Enter a Gemini API key to use AI analysis."
            )

        else:

            try:

                import google.generativeai as genai

                genai.configure(
                    api_key=api_key
                )

                model = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )

                prompt = f"""
You are a professional data quality analyst.

Analyze ONLY the supplied metrics.

Do not invent relationships, causes,
business facts or conclusions.

Clearly distinguish:
1. Confirmed data-quality issues
2. Statistical outliers
3. ML anomaly screening flags
4. Possible causes, which must be labeled as hypotheses

Dataset:
Rows: {rows}
Columns: {columns}
Missing cells: {missing_cells}
Duplicate rows: {duplicate_count}
Invalid values: {invalid_count}
City inconsistencies: {city_rows}
IQR outlier records: {outlier_rows}
ML anomaly flags: {ml_count}
ML sensitivity: {contamination_percent}%
Quality score: {score}/100
Quality status: {quality_status}

Numeric columns:
{numeric_cols}

Categorical columns:
{categorical_cols}

Datetime columns:
{datetime_cols}

Provide:

1. Executive Summary
2. Confirmed Data Quality Issues
3. Statistical Observations
4. ML Anomaly Interpretation
5. Root-Cause Hypotheses
6. Recommended Actions
7. Potential Business Impact
8. Priority Level

Do not call the dataset unusable simply because
the ML model flagged records.
"""

                with st.spinner(
                    "Generating AI analysis..."
                ):

                    response = model.generate_content(
                        prompt
                    )

                st.markdown(
                    "### 🤖 AI Analysis"
                )

                st.markdown(
                    response.text
                )

            except Exception as e:

                st.error(
                    f"AI analysis failed: {e}"
                )

                st.info(
                    "The rest of DataGuard AI works "
                    "without the Gemini feature."
                )


# ============================================================
# POWER BI TAB
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">📦 Power BI Export</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        DataGuard AI creates clean CSV files that can be
        imported directly into Power BI.
        """
    )

    cleaned_df = clean_data(df)

    dictionary_df = create_data_dictionary(
        cleaned_df
    )

    quality_df = create_quality_summary(
        df,
        invalid_count,
        duplicate_count,
        city_rows,
        outlier_rows,
        ml_count,
        score,
        quality_status,
    )

    numeric_summary_df = create_numeric_summary(
        cleaned_df
    )

    missing_summary_df = create_missing_summary(
        df
    )

    export_files = {
        "cleaned_data.csv":
            dataframe_to_csv_bytes(
                cleaned_df
            ),

        "data_dictionary.csv":
            dataframe_to_csv_bytes(
                dictionary_df
            ),

        "quality_summary.csv":
            dataframe_to_csv_bytes(
                quality_df
            ),

        "numeric_summary.csv":
            dataframe_to_csv_bytes(
                numeric_summary_df
            ),

        "missing_summary.csv":
            dataframe_to_csv_bytes(
                missing_summary_df
            ),
    }

    st.markdown(
        "### 📁 Generated Files"
    )

    for filename in export_files:

        st.write(
            f"✅ `{filename}`"
        )

    zip_data = create_zip(
        export_files
    )

    st.download_button(
        label="📦 Download Power BI Package",
        data=zip_data,
        file_name="DataGuard_AI_PowerBI_Package.zip",
        mime="application/zip",
        type="primary",
    )

    st.markdown(
        "### 📊 Power BI Workflow"
    )

    st.markdown(
        """
        **1. Download the package**

        ↓

        **2. Extract the ZIP**

        ↓

        **3. Open Power BI Desktop**

        ↓

        **4. Get Data → Text/CSV**

        ↓

        **5. Select `cleaned_data.csv`**

        ↓

        **6. Build your Power BI dashboard**

        ↓

        **7. Use `quality_summary.csv` and
        `data_dictionary.csv` for supporting analysis**
        """
    )


# ============================================================
# REPORT TAB
# ============================================================

with tabs[7]:

    st.markdown(
        '<div class="section-title">📄 Final Data Quality Report</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        ### 🛡️ DataGuard AI Report

        **Dataset:** {uploaded_file.name}

        **Rows:** {rows:,}

        **Columns:** {columns:,}

        **Missing Cells:** {missing_cells:,}

        **Duplicate Rows:** {duplicate_count:,}

        **Invalid Values:** {invalid_count:,}

        **City Inconsistencies:** {city_rows:,}

        **IQR Outlier Records:** {outlier_rows:,}

        **ML Anomaly Flags:** {ml_count:,}

        **Quality Score:** {score}/100

        **Quality Status:** {quality_status}

        ---

        ### 🔍 Detected Data Quality Issues

        - Missing values: **{missing_cells:,}**
        - Duplicate rows: **{duplicate_count:,}**
        - Invalid values: **{invalid_count:,}**
        - City inconsistencies: **{city_rows:,}**
        - Statistical outlier records: **{outlier_rows:,}**

        ### 🚨 ML Anomaly Screening

        Isolation Forest flagged:

        **{ml_count:,} records**

        at the selected sensitivity of:

        **{contamination_percent}%**

        These records should be investigated rather than
        automatically classified as incorrect.

        ### 🧹 Cleaning

        The automated cleaning pipeline can:

        - Standardize known city labels
        - Handle invalid numeric values
        - Fill missing numeric values
        - Fill categorical missing values
        - Handle detected datetime columns
        - Remove duplicate records

        ### 📊 Power BI

        The application generates:

        - Cleaned dataset
        - Data dictionary
        - Quality summary
        - Numeric summary
        - Missing-value summary

        """

    )

    cleaned_report = clean_data(df)

    report_text = f"""
DataGuard AI - Data Quality Report

Dataset: {uploaded_file.name}

Rows: {rows}
Columns: {columns}
Missing Cells: {missing_cells}
Duplicate Rows: {duplicate_count}
Invalid Values: {invalid_count}
City Inconsistencies: {city_rows}
IQR Outlier Records: {outlier_rows}
ML Anomaly Flags: {ml_count}
ML Sensitivity: {contamination_percent}%
Quality Score: {score}/100
Quality Status: {quality_status}

Cleaned Rows: {len(cleaned_report)}
Cleaned Missing Cells: {int(cleaned_report.isna().sum().sum())}
"""

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="DataGuard_AI_Report.txt",
        mime="text/plain",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI • Built with Python,
        Pandas, Scikit-learn & Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
