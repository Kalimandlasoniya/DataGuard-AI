```python
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
        background: #f6f8fb;
        color: #111827 !important;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Force main application text to dark */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        color: #111827 !important;
    }

    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] div {
        color: #111827;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #e5e7eb;
    }

    section[data-testid="stSidebar"]
    [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] * {
        color: #9ca3af !important;
    }

    .sidebar-brand {
        padding: 10px 5px 22px 5px;
    }

    .sidebar-logo {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff !important;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 800;
        margin-top: 3px;
        color: #ffffff !important;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #9ca3af !important;
        margin-top: 3px;
    }

    .sidebar-divider {
        height: 1px;
        background: #374151;
        margin: 12px 0 18px 0;
    }

    /* Sidebar navigation radio */
    section[data-testid="stSidebar"]
    [data-testid="stRadio"] label {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stRadio"] label span {
        color: #e5e7eb !important;
    }

    /* Sidebar slider */
    section[data-testid="stSidebar"]
    [data-testid="stWidgetLabel"] * {
        color: #e5e7eb !important;
    }

    /* ========================================================
       HEADERS
       ======================================================== */

    .page-title {
        font-size: 34px;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 2px;
    }

    .page-subtitle {
        font-size: 15px;
        color: #6b7280 !important;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #111827 !important;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #6b7280 !important;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        color: #111827 !important;
    }

    .kpi-card * {
        color: #111827;
    }

    .kpi-label {
        font-size: 13px;
        color: #6b7280 !important;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #111827 !important;
    }

    .kpi-caption {
        font-size: 12px;
        color: #9ca3af !important;
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
        color: #166534 !important;
        font-size: 12px;
        font-weight: 700;
    }

    .status-warning {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #fef3c7;
        color: #92400e !important;
        font-size: 12px;
        font-weight: 700;
    }

    .status-danger {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #fee2e2;
        color: #991b1b !important;
        font-size: 12px;
        font-weight: 700;
    }

    .status-info {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        background: #dbeafe;
        color: #1e40af !important;
        font-size: 12px;
        font-weight: 700;
    }

    /* ========================================================
       PANELS
       ======================================================== */

    .dashboard-panel {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        margin-top: 12px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        color: #111827 !important;
    }

    .dashboard-panel * {
        color: #111827;
    }

    .panel-title {
        font-size: 17px;
        font-weight: 750;
        color: #111827 !important;
        margin-bottom: 4px;
    }

    .panel-description {
        color: #6b7280 !important;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ========================================================
       UPLOAD
       ======================================================== */

    .upload-banner {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        color: #111827 !important;
    }

    /* ========================================================
       INFO / WARNING / SUCCESS
       ======================================================== */

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 13px 15px;
        color: #1e3a8a !important;
        font-size: 13px;
        margin: 10px 0;
    }

    .info-box * {
        color: #1e3a8a !important;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 13px 15px;
        color: #92400e !important;
        font-size: 13px;
        margin: 10px 0;
    }

    .warning-box * {
        color: #92400e !important;
    }

    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 13px 15px;
        color: #166534 !important;
        font-size: 13px;
        margin: 10px 0;
    }

    .success-box * {
        color: #166534 !important;
    }

    /* ========================================================
       STREAMLIT WIDGET TEXT
       ======================================================== */

    /* Captions */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #6b7280 !important;
    }

    /* Widget labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] * {
        color: #111827 !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] * {
        color: #111827 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
    }

    /* Expander */
    [data-testid="stExpander"],
    [data-testid="stExpander"] * {
        color: #111827 !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }

    .stButton > button:hover {
        color: #111827 !important;
        border-color: #9ca3af !important;
    }

    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 650;
        color: #111827 !important;
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }

    .stDownloadButton > button:hover {
        color: #111827 !important;
        border-color: #9ca3af !important;
    }

    /* ========================================================
       METRIC
       ======================================================== */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
        color: #111827 !important;
    }

    [data-testid="stMetric"] * {
        color: #111827 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ========================================================
       SELECT / INPUTS
       ======================================================== */

    [data-baseweb="select"] {
        background: #ffffff !important;
    }

    [data-baseweb="select"] * {
        color: #111827 !important;
    }

    input,
    textarea {
        color: #111827 !important;
        background: #ffffff !important;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #9ca3af !important;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    .footer * {
        color: #9ca3af !important;
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
    "cleaning_done": False,
    "contamination_pct": 5,
}

for key, value in defaults.items():
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

        parsed = safe_to_datetime(
            df[column]
        )

        valid_ratio = parsed.notna().mean()

        if any(
            token in column_name
            for token in tokens
        ):
            if valid_ratio >= 0.50:
                datetime_columns.append(
                    column
                )
        else:
            if valid_ratio >= 0.95:
                datetime_columns.append(
                    column
                )

    return list(
        dict.fromkeys(datetime_columns)
    )


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
            identifier_columns.append(
                column
            )
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
                identifier_columns.append(
                    column
                )

    return list(
        dict.fromkeys(identifier_columns)
    )


def profile_dataset(df):

    profile = pd.DataFrame({
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
    })

    return profile


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

        original_non_null = (
            df[column].notna()
        )

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

            invalid_details.append({
                "Column": column,
                "Invalid Values": count,
            })

    return (
        invalid_count,
        pd.DataFrame(invalid_details),
    )


def standardize_city_column(df):

    result = df.copy()

    total_changes = 0
    change_details = []

    city_columns = [
        column
        for column in result.columns
        if "city"
        in str(column).lower()
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

        valid_original = (
            original.astype("string")
        )

        changes = (
            valid_original.fillna("")
            != standardized.fillna("")
        )

        changed_count = int(
            changes.sum()
        )

        if changed_count > 0:

            total_changes += changed_count

            change_details.append({
                "Column": column,
                "Changed Values": changed_count,
            })

        result[column] = standardized

    return (
        result,
        total_changes,
    )


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

        lower = q1 - (
            1.5 * iqr
        )

        upper = q3 + (
            1.5 * iqr
        )

        mask = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        count = int(
            mask.sum()
        )

        if count > 0:

            total_outliers += count

            details.append({
                "Column": column,
                "Q1": q1,
                "Q3": q3,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outliers": count,
            })

    return (
        total_outliers,
        pd.DataFrame(details),
    )


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
            min(100, score),
        ),
        2,
    )


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
                ).astype("float64")
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
                .mode(dropna=True)
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

            if pd.api.types.is
```
