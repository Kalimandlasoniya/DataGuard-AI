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
# PREMIUM SAAS CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #f5f7fb;
        color: #111827;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0b1120;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] > div {
        background: #0b1120;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .sidebar-brand {
        padding: 8px 4px 18px 4px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 11px;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );
        font-size: 22px;
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.25);
    }

    .brand-name {
        font-size: 20px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.4px;
    }

    .brand-caption {
        color: #94a3b8 !important;
        font-size: 11px;
        margin-top: 1px;
    }

    .sidebar-section {
        color: #64748b !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 18px;
        margin-bottom: 7px;
    }

    .pipeline-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px;
        margin-top: 10px;
    }

    .pipeline-step {
        font-size: 11px;
        color: #94a3b8 !important;
        padding: 4px 0;
    }

    .pipeline-step span {
        color: #64748b !important;
        margin-right: 7px;
        font-weight: 700;
    }

    .sidebar-footer {
        color: #64748b !important;
        font-size: 10px;
        line-height: 1.5;
        padding-top: 12px;
    }

    /* ========================================================
       NAVIGATION RADIO
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] label {
        background: transparent;
        border-radius: 9px;
        padding: 7px 9px;
        margin: 2px 0;
        transition: 0.15s ease;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] label:hover {
        background: #172033;
    }

    /* ========================================================
       TOP HEADER
       ======================================================== */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
    }

    .eyebrow {
        color: #64748b;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .main-title {
        color: #0f172a;
        font-size: 34px;
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: -1.2px;
        margin: 0;
    }

    .main-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-top: 7px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 750;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
    }

    /* ========================================================
       HERO / UPLOAD
       ======================================================== */

    .upload-hero {
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 55%,
                #312e81 100%
            );
        border-radius: 20px;
        padding: 27px 30px;
        margin-bottom: 23px;
        color: white;
        box-shadow:
            0 15px 40px rgba(15, 23, 42, 0.13);
        position: relative;
        overflow: hidden;
    }

    .upload-hero:after {
        content: "";
        position: absolute;
        width: 230px;
        height: 230px;
        right: -90px;
        top: -100px;
        border-radius: 50%;
        background: rgba(96, 165, 250, 0.10);
    }

    .hero-label {
        color: #93c5fd;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }

    .hero-description {
        color: #cbd5e1;
        font-size: 13px;
        max-width: 700px;
        line-height: 1.6;
        margin-top: 5px;
    }

    .hero-meta {
        display: flex;
        gap: 8px;
        margin-top: 15px;
        flex-wrap: wrap;
    }

    .hero-tag {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 6px 9px;
        color: #dbeafe;
        font-size: 10px;
    }

    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow:
            0 4px 15px rgba(15, 23, 42, 0.035);
        margin-bottom: 16px;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
    }

    .card-title {
        color: #111827;
        font-size: 15px;
        font-weight: 800;
    }

    .card-subtitle {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 3px;
    }

    .section-heading {
        color: #0f172a;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-top: 25px;
        margin-bottom: 4px;
    }

    .section-subheading {
        color: #64748b;
        font-size: 12px;
        margin-bottom: 15px;
    }

    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 17px;
        min-height: 120px;
        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.035);
    }

    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }

    .kpi-icon {
        width: 30px;
        height: 30px;
        border-radius: 9px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: #f1f5f9;
        font-size: 14px;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 27px;
        font-weight: 850;
        margin-top: 10px;
        letter-spacing: -0.7px;
    }

    .kpi-caption {
        color: #94a3b8;
        font-size: 10px;
        margin-top: 3px;
    }

    /* ========================================================
       SCORE
       ======================================================== */

    .score-card {
        background: linear-gradient(
            135deg,
            #ffffff,
            #f8fafc
        );
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 22px;
        min-height: 220px;
        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.04);
    }

    .score-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .score-number {
        color: #0f172a;
        font-size: 54px;
        font-weight: 900;
        letter-spacing: -2px;
        margin-top: 12px;
    }

    .score-denom {
        color: #94a3b8;
        font-size: 17px;
        font-weight: 600;
    }

    .score-status {
        display: inline-block;
        margin-top: 8px;
        border-radius: 999px;
        padding: 6px 11px;
        font-size: 11px;
        font-weight: 800;
    }

    .score-good {
        background: #dcfce7;
        color: #166534;
    }

    .score-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .score-danger {
        background: #fee2e2;
        color: #991b1b;
    }

    .progress-track {
        height: 8px;
        background: #e2e8f0;
        border-radius: 999px;
        margin-top: 18px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );
    }

    /* ========================================================
       INFO / ALERTS
       ======================================================== */

    .notice {
        border-radius: 11px;
        padding: 12px 14px;
        font-size: 12px;
        line-height: 1.55;
        margin: 10px 0;
    }

    .notice-blue {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
    }

    .notice-green {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
    }

    .notice-yellow {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
    }

    .notice-red {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
    }

    /* ========================================================
       FEATURE CARDS
       ======================================================== */

    .feature-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 20px;
        min-height: 150px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.03);
    }

    .feature-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: #eff6ff;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 18px;
        margin-bottom: 13px;
    }

    .feature-title {
        color: #111827;
        font-size: 14px;
        font-weight: 800;
    }

    .feature-text {
        color: #64748b;
        font-size: 11px;
        line-height: 1.55;
        margin-top: 5px;
    }

    /* ========================================================
       FILE INFORMATION
       ======================================================== */

    .dataset-strip {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 13px;
        padding: 12px 15px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .dataset-name {
        color: #111827;
        font-size: 13px;
        font-weight: 800;
    }

    .dataset-meta {
        color: #64748b;
        font-size: 11px;
        margin-top: 3px;
    }

    /* ========================================================
       TABLES
       ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 700;
        border: 1px solid #dbe1ea;
        min-height: 38px;
    }

    .stButton > button[kind="primary"] {
        background: #2563eb;
        border-color: #2563eb;
    }

    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        font-size: 12px;
        font-weight: 700;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 10px;
        margin-top: 55px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    /* ========================================================
       EMPTY STATE
       ======================================================== */

    .empty-hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 38px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(15,23,42,0.035);
        margin-bottom: 25px;
    }

    .empty-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 15px auto;
        border-radius: 18px;
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(
            135deg,
            #eff6ff,
            #eef2ff
        );
        font-size: 30px;
    }

    .empty-title {
        color: #0f172a;
        font-size: 25px;
        font-weight: 850;
    }

    .empty-description {
        color: #64748b;
        font-size: 13px;
        max-width: 600px;
        margin: 7px auto 0 auto;
        line-height: 1.6;
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
    "analysis": None,
    "gemini_analysis": None,
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

    return list(dict.fromkeys(identifier_columns))


def profile_dataset(df):

    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non-Null Count": [
                int(df[column].notna().sum())
                for column in df.columns
            ],
            "Missing": [
                int(df[column].isna().sum())
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
        pd.DataFrame(invalid_details),
    )


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
            original.astype("string")
            .fillna("")
            != standardized.fillna("")
        )

        total_changes += int(
            changes.sum()
        )

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

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        count = int(mask.sum())

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


def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        ).copy()
    )

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
        cleaned.select_dtypes(
            include=np.number
        ).columns
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
        cleaned.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        ).columns
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

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

    return None


def prepare_business_data(df):

    sales_column = find_sales_column(df)

    if sales_column is None:
        return None, None, None

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
        return None, sales_column, None

    q1 = business_df[
        sales_column
    ].quantile(0.25)

    q3 = business_df[
        sales_column
    ].quantile(0.75)

    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr

    chart_df = business_df[
        business_df[sales_column] <= upper_bound
    ].copy()

    return (
        chart_df,
        sales_column,
        upper_bound,
    )


# ============================================================
# IMPROVED POWER BI EXPORTS
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

        # ----------------------------------------------------
        # SALES SUMMARY
        # ----------------------------------------------------

        sales_summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Sales",
                    "Average Sales",
                    "Minimum Sales",
                    "Maximum Sales",
                    "Record Count",
                ],
                "Value": [
                    business_df[
                        sales_column
                    ].sum(),
                    business_df[
                        sales_column
                    ].mean(),
                    business_df[
                        sales_column
                    ].min(),
                    business_df[
                        sales_column
                    ].max(),
                    len(business_df),
                ],
            }
        )

        exports[
            "DataGuard_Sales_Summary.csv"
        ] = (
            sales_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

        # ----------------------------------------------------
        # CITY SUMMARY
        # ----------------------------------------------------

        city_columns = [
            column
            for column in cleaned_df.columns
            if "city" in str(column).lower()
        ]

        if city_columns:

            city_column = city_columns[0]

            city_summary = (
                business_df
                .groupby(city_column)[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count",
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False,
                )
            )

            exports[
                "DataGuard_City_Summary.csv"
            ] = (
                city_summary
                .to_csv(index=False)
                .encode("utf-8")
            )

        # ----------------------------------------------------
        # CATEGORY SUMMARY
        # ----------------------------------------------------

        category_columns = [
            column
            for column in cleaned_df.columns
            if "category"
            in str(column).lower()
        ]

        if category_columns:

            category_column = category_columns[0]

            category_summary = (
                business_df
                .groupby(category_column)[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count",
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False,
                )
            )

            exports[
                "DataGuard_Category_Summary.csv"
            ] = (
                category_summary
                .to_csv(index=False)
                .encode("utf-8")
            )

        # ----------------------------------------------------
        # PRODUCT SUMMARY
        # ----------------------------------------------------

        product_columns = [
            column
            for column in cleaned_df.columns
            if "product"
            in str(column).lower()
        ]

        if product_columns:

            product_column = product_columns[0]

            product_summary = (
                business_df
                .groupby(product_column)[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count",
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False,
                )
            )

            exports[
                "DataGuard_Product_Summary.csv"
            ] = (
                product_summary
                .head(1000)
                .to_csv(index=False)
                .encode("utf-8")
            )

        # ----------------------------------------------------
        # MONTHLY SUMMARY
        # ----------------------------------------------------

        datetime_columns = (
            detect_datetime_columns(
                cleaned_df
            )
        )

        if datetime_columns:

            date_column = datetime_columns[0]

            monthly_df = cleaned_df.copy()

            monthly_df[
                date_column
            ] = safe_to_datetime(
                monthly_df[date_column]
            )

            monthly_df[
                sales_column
            ] = pd.to_numeric(
                monthly_df[sales_column],
                errors="coerce",
            )

            monthly_df = monthly_df.dropna(
                subset=[
                    date_column,
                    sales_column,
                ]
            )

            monthly_summary = (
                monthly_df
                .assign(
                    Month=monthly_df[
                        date_column
                    ]
                    .dt.to_period("M")
                    .astype(str)
                )
                .groupby("Month")[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count",
                )
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

        for filename, data in exports.items():

            zip_file.writestr(
                filename,
                data,
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

try:
    GEMINI_API_KEY = st.secrets.get(
        "GEMINI_API_KEY",
        "",
    )
except Exception:
    GEMINI_API_KEY = ""

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY",
        "",
    )


def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:

        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets "
            "to enable AI analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior data quality analyst.

Analyze ONLY the supplied DataGuard AI report.

Do not invent facts, columns, errors, causes,
business events, fraud, system failures,
ETL failures, retry behavior, or manual-entry behavior.

Possible causes must be clearly labeled as hypotheses.

IQR outliers and Isolation Forest anomalies are NOT
automatically confirmed data errors.

The cleaning process has already been performed.
Do not recommend repeating the same cleaning steps.

Extreme sales values excluded from business charts
were excluded ONLY for visualization and were NOT
deleted from the analytical dataset.

Use the exact supplied metrics.

Create a concise portfolio-friendly analysis with:

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
            "The core DataGuard AI pipeline continues "
            "to work without Gemini."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-row">
                <div class="brand-icon">🛡️</div>
                <div>
                    <div class="brand-name">
                        DataGuard AI
                    </div>
                    <div class="brand-caption">
                        AI Data Quality Platform
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Workspace</div>',
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
        '<div class="sidebar-section">Detection Settings</div>',
        unsafe_allow_html=True,
    )

    new_contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
        step=1,
    )

    if (
        new_contamination
        != st.session_state.contamination_pct
    ):

        st.session_state.contamination_pct = (
            new_contamination
        )

        if st.session_state.df is not None:

            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

        st.rerun()

    st.caption(
        "Higher sensitivity flags more records as unusual. "
        "ML anomalies are screening signals."
    )

    st.markdown(
        '<div class="sidebar-section">Pipeline</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-step">
                <span>01</span> Upload
            </div>
            <div class="pipeline-step">
                <span>02</span> Profile
            </div>
            <div class="pipeline-step">
                <span>03</span> Quality Checks
            </div>
            <div class="pipeline-step">
                <span>04</span> Anomaly Detection
            </div>
            <div class="pipeline-step">
                <span>05</span> Cleaning
            </div>
            <div class="pipeline-step">
                <span>06</span> Analytics
            </div>
            <div class="pipeline-step">
                <span>07</span> Power BI
            </div>
            <div class="pipeline-step">
                <span>08</span> AI Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-footer">
            DataGuard AI<br>
            Portfolio Edition<br><br>
            Python • Pandas • Scikit-learn<br>
            Streamlit • Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="eyebrow">
                Data Intelligence Workspace
            </div>
            <div class="main-title">
                DataGuard AI
            </div>
            <div class="main-subtitle">
                AI-powered data quality, anomaly detection
                and business analytics
            </div>
        </div>

        <div class="status-pill">
            <span class="status-dot"></span>
            Data Quality Engine
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD HERO
# ============================================================

st.markdown(
    """
    <div class="upload-hero">
        <div class="hero-label">
            START A NEW ANALYSIS
        </div>

        <div class="hero-title">
            Protect the quality of your data
        </div>

        <div class="hero-description">
            Upload a CSV or Excel dataset and DataGuard AI
            will profile your data, detect quality issues,
            identify statistical and ML anomalies, clean
            the dataset and prepare business-ready exports.
        </div>

        <div class="hero-meta">
            <div class="hero-tag">CSV</div>
            <div class="hero-tag">XLSX</div>
            <div class="hero-tag">Data Profiling</div>
            <div class="hero-tag">IQR Detection</div>
            <div class="hero-tag">Isolation Forest</div>
            <div class="hero-tag">Power BI Ready</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload dataset",
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
            st.session_state.analysis = None
            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )

            st.stop()


# ============================================================
# ANALYSIS ENGINE
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if not st.session_state.analysis_complete:

        with st.spinner(
            "Running DataGuard AI quality engine..."
        ):

            rows = len(df)
            columns = len(df.columns)

            total_cells = (
                rows * columns
            )

            missing_count = int(
                df.isna()
                .sum()
                .sum()
            )

            duplicate_count = int(
                df.duplicated()
                .sum()
            )

            invalid_count, invalid_details = (
                detect_invalid_values(df)
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
                detect_datetime_columns(df)
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

            iqr_outlier_count, iqr_details = (
                detect_iqr_outliers(df)
            )

            anomaly_mask, ml_anomaly_count = (
                detect_ml_anomalies(
                    df,
                    st.session_state.contamination_pct,
                )
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
            ) = clean_dataset(df)

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
                "datetime_columns": datetime_columns,
                "identifier_columns": identifier_columns,
                "numeric_count": numeric_count,
                "categorical_count": categorical_count,
                "iqr_outlier_count": iqr_outlier_count,
                "iqr_details": iqr_details,
                "anomaly_mask": anomaly_mask,
                "ml_anomaly_count": ml_anomaly_count,
                "normal_count": normal_count,
                "invalid_details": invalid_details,
                "rows_removed": rows_removed,
                "values_filled": values_filled,
                "city_changes": city_changes,
                "sales_column": sales_column,
                "upper_bound": upper_bound,
            }

            st.session_state.analysis_complete = True

            st.rerun()


# ============================================================
# DISPLAY APP
# ============================================================

if (
    st.session_state.df is not None
    and st.session_state.analysis_complete
):

    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df
    business_df = st.session_state.business_df
    analysis = st.session_state.analysis

    contamination_pct = (
        st.session_state.contamination_pct
    )

    # ========================================================
    # DATASET STRIP
    # ========================================================

    st.markdown(
        f"""
        <div class="dataset-strip">
            <div>
                <div class="dataset-name">
                    {st.session_state.file_name}
                </div>
                <div class="dataset-meta">
                    {analysis["rows"]:,} records
                    &nbsp; • &nbsp;
                    {analysis["columns"]} columns
                    &nbsp; • &nbsp;
                    Full dataset analyzed
                </div>
            </div>

            <div class="status-pill">
                <span class="status-dot"></span>
                Analysis Complete
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.markdown(
            """
            <div class="section-heading">
                Dashboard
            </div>

            <div class="section-subheading">
                Monitor dataset health, quality signals
                and business readiness.
            </div>
            """,
            unsafe_allow_html=True,
        )

        score = analysis["quality_score"]

        if score >= 99:
            status = "Excellent"
            score_class = "score-good"
        elif score >= 95:
            status = "Good"
            score_class = "score-good"
        elif score >= 85:
            status = "Needs Attention"
            score_class = "score-warning"
        else:
            status = "Critical"
            score_class = "score-danger"

        left, right = st.columns(
            [1.05, 2.2]
        )

        with left:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        DATA QUALITY SCORE
                    </div>

                    <div class="score-number">
                        {score}
                        <span class="score-denom">
                            /100
                        </span>
                    </div>

                    <span class="score-status {score_class}">
                        {status}
                    </span>

                    <div class="progress-track">
                        <div
                            class="progress-fill"
                            style="width:{score}%"
                        ></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:

            k1, k2 = st.columns(2)

            with k1:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-top">
                            <div class="kpi-label">
                                Missing Cells
                            </div>
                            <div class="kpi-icon">
                                ◇
                            </div>
                        </div>

                        <div class="kpi-value">
                            {analysis["missing_count"]:,}
                        </div>

                        <div class="kpi-caption">
                            Missing observations
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with k2:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-top">
                            <div class="kpi-label">
                                Duplicates
                            </div>
                            <div class="kpi-icon">
                                ⧉
                            </div>
                        </div>

                        <div class="kpi-value">
                            {analysis["duplicate_count"]:,}
                        </div>

                        <div class="kpi-caption">
                            Duplicate records
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            k3, k4 = st.columns(2)

            with k3:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-top">
                            <div class="kpi-label">
                                IQR Outliers
                            </div>
                            <div class="kpi-icon">
                                ◈
                            </div>
                        </div>

                        <div class="kpi-value">
                            {analysis["iqr_outlier_count"]:,}
                        </div>

                        <div class="kpi-caption">
                            Statistical signals
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with k4:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-top">
                            <div class="kpi-label">
                                ML Anomalies
                            </div>
                            <div class="kpi-icon">
                                AI
                            </div>
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

        st.markdown(
            """
            <div class="section-heading">
                Dataset Profile
            </div>

            <div class="section-subheading">
                Structural overview of the uploaded dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4, p5 = st.columns(5)

        p1.metric(
            "Rows",
            f'{analysis["rows"]:,}',
        )

        p2.metric(
            "Columns",
            analysis["columns"],
        )

        p3.metric(
            "Numeric",
            analysis["numeric_count"],
        )

        p4.metric(
            "Categorical",
            analysis["categorical_count"],
        )

        p5.metric(
            "Date / Time",
            len(
                analysis[
                    "datetime_columns"
                ]
            ),
        )

        st.markdown(
            """
            <div class="section-heading">
                Quality Monitoring
            </div>

            <div class="section-subheading">
                Signals detected across the complete dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        quality_df = pd.DataFrame(
            {
                "Metric": [
                    "Missing",
                    "Duplicates",
                    "Invalid",
                    "IQR Outliers",
                    "ML Anomalies",
                ],
                "Count": [
                    analysis["missing_count"],
                    analysis["duplicate_count"],
                    analysis["invalid_count"],
                    analysis["iqr_outlier_count"],
                    analysis["ml_anomaly_count"],
                ],
            }
        )

        st.bar_chart(
            quality_df.set_index(
                "Metric"
            ),
            height=300,
        )

        if business_df is not None:

            st.markdown(
                """
                <div class="section-heading">
                    Business Snapshot
                </div>

                <div class="section-subheading">
                    Quick view of the commercial signals
                    detected in your dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

            sales_column = analysis[
                "sales_column"
            ]

            b1, b2, b3 = st.columns(3)

            with b1:

                st.metric(
                    "Total Sales",
                    f'{business_df[sales_column].sum():,.0f}',
                )

            with b2:

                st.metric(
                    "Average Sale",
                    f'{business_df[sales_column].mean():,.2f}',
                )

            with b3:

                st.metric(
                    "Sales Records",
                    f'{len(business_df):,}',
                )

        st.markdown(
            """
            <div class="section-heading">
                Data Preview
            </div>

            <div class="section-subheading">
                First 10 records from the original dataset.
            </div>
            """,
            unsafe_allow_html=True,
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
            """
            <div class="section-heading">
                Data Quality
            </div>

            <div class="section-subheading">
                Detailed profiling and rule-based validation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Quality Score",
            f'{analysis["quality_score"]}/100',
        )

        q2.metric(
            "Missing",
            f'{analysis["missing_count"]:,}',
        )

        q3.metric(
            "Duplicates",
            f'{analysis["duplicate_count"]:,}',
        )

        q4.metric(
            "Invalid",
            f'{analysis["invalid_count"]:,}',
        )

        st.markdown(
            """
            <div class="section-heading">
                Column Profile
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            profile_dataset(df),
            use_container_width=True,
            height=450,
        )

        st.markdown(
            """
            <div class="section-heading">
                Detected Data Types
            </div>
            """,
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3)

        a.metric(
            "Numeric Columns",
            analysis["numeric_count"],
        )

        b.metric(
            "Categorical Columns",
            analysis["categorical_count"],
        )

        c.metric(
            "Date / Time Columns",
            len(
                analysis[
                    "datetime_columns"
                ]
            ),
        )

        with st.expander(
            "View Date / Time Columns"
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
            "View Identifier Columns"
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
                    "No identifier columns detected."
                )

        st.markdown(
            """
            <div class="section-heading">
                Invalid Values
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis[
            "invalid_details"
        ].empty:

            st.markdown(
                """
                <div class="notice notice-green">
                    No invalid values were detected by the
                    configured validation rules.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.dataframe(
                analysis[
                    "invalid_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="section-heading">
                City Standardization
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="notice notice-blue">
                <b>{analysis["city_changes"]:,}</b>
                city values can be standardized during
                the cleaning stage.
            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        st.markdown(
            """
            <div class="section-heading">
                Anomaly Detection
            </div>

            <div class="section-subheading">
                Statistical outlier detection and machine-learning
                anomaly screening.
            </div>
            """,
            unsafe_allow_html=True,
        )

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "IQR Outliers",
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
            <div class="notice notice-yellow">
                <b>Interpretation:</b>
                IQR outliers and Isolation Forest anomalies
                are statistically unusual observations.
                They are not automatically confirmed data errors.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-heading">
                Statistical Outliers
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis[
            "iqr_details"
        ].empty:

            st.markdown(
                """
                <div class="notice notice-green">
                    No IQR outliers detected.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.dataframe(
                analysis[
                    "iqr_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            """
            <div class="section-heading">
                Isolation Forest
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="notice notice-blue">
                Current sensitivity:
                <b>{contamination_pct}%</b>.
                Adjust this value from the sidebar to rerun
                the anomaly model.
            </div>
            """,
            unsafe_allow_html=True,
        )

        anomaly_display = df.copy()

        anomaly_display[
            "ML_Anomaly"
        ] = np.where(
            analysis[
                "anomaly_mask"
            ],
            "Anomaly",
            "Normal",
        )

        anomaly_records = (
            anomaly_display[
                anomaly_display[
                    "ML_Anomaly"
                ]
                == "Anomaly"
            ]
        )

        with st.expander(
            "View detected anomaly records"
        ):

            st.caption(
                f"Showing up to 500 of "
                f"{len(anomaly_records):,} "
                f"detected anomalies."
            )

            st.dataframe(
                anomaly_records.head(500),
                use_container_width=True,
                height=450,
            )


    # ========================================================
    # CLEANING
    # ========================================================

    elif page == "Cleaning":

        st.markdown(
            """
            <div class="section-heading">
                Automated Cleaning
            </div>

            <div class="section-subheading">
                Transform the uploaded dataset into a cleaner,
                analysis-ready version.
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

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
            <div class="notice notice-green">
                <b>Cleaning pipeline completed.</b>
                Duplicate records were removed, city values
                standardized, numeric missing values filled
                using median values, and categorical missing
                values filled using mode values.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if analysis[
            "city_changes"
        ] > 0:

            st.markdown(
                f"""
                <div class="notice notice-blue">
                    {analysis["city_changes"]:,}
                    city values were standardized.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="section-heading">
                Cleaned Dataset
            </div>

            <div class="section-subheading">
                Preview of the Power BI-ready dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            cleaned_df.head(10),
            use_container_width=True,
            height=350,
        )

        st.download_button(
            "Download Cleaned CSV",
            data=(
                cleaned_df
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name="DataGuard_Cleaned_Data.csv",
            mime="text/csv",
            type="primary",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "Analytics":

        st.markdown(
            """
            <div class="section-heading">
                Business Analytics
            </div>

            <div class="section-subheading">
                Explore business trends generated from
                the cleaned dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if business_df is None:

            st.markdown(
                """
                <div class="notice notice-yellow">
                    No Sales, Revenue or Amount column was detected.
                    Business sales analytics are therefore unavailable.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            sales_column = analysis[
                "sales_column"
            ]

            upper_bound = analysis[
                "upper_bound"
            ]

            st.markdown(
                f"""
                <div class="notice notice-blue">
                    Business charts use the cleaned dataset.
                    Values above the IQR upper bound
                    <b>{upper_bound:,.2f}</b> are excluded
                    only for visualization. The records remain
                    in the analytical dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # -----------------------------------------------
            # SALES SUMMARY
            # -----------------------------------------------

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                "Total Sales",
                f'{business_df[sales_column].sum():,.0f}',
            )

            s2.metric(
                "Average Sale",
                f'{business_df[sales_column].mean():,.2f}',
            )

            s3.metric(
                "Highest Sale",
                f'{business_df[sales_column].max():,.2f}',
            )

            s4.metric(
                "Records",
                f'{len(business_df):,}',
            )

            # -----------------------------------------------
            # SALES DISTRIBUTION
            # -----------------------------------------------

            st.markdown(
                """
                <div class="section-heading">
                    Sales Distribution
                </div>
                """,
                unsafe_allow_html=True,
            )

            values = business_df[
                sales_column
            ]

            if len(values) > 1:

                hist, bins = np.histogram(
                    values,
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
                    ),
                    height=320,
                )

            # -----------------------------------------------
            # CITY / PRODUCT
            # -----------------------------------------------

            left, right = st.columns(2)

            with left:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_column = city_columns[0]

                    city_sales = (
                        business_df
                        .groupby(city_column)[
                            sales_column
                        ]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                    )

                    st.markdown(
                        """
                        <div class="card">
                            <div class="card-title">
                                Sales by City
                            </div>
                            <div class="card-subtitle">
                                Top locations by total sales
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        city_sales,
                        height=300,
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
                        .groupby(product_column)[
                            sales_column
                        ]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                    )

                    st.markdown(
                        """
                        <div class="card">
                            <div class="card-title">
                                Sales by Product
                            </div>
                            <div class="card-subtitle">
                                Top products by total sales
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        product_sales,
                        height=300,
                    )

            # -----------------------------------------------
            # CATEGORY
            # -----------------------------------------------

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
                    .groupby(category_column)[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    """
                    <div class="section-heading">
                        Sales by Category
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.bar_chart(
                    category_sales,
                    height=300,
                )

            # -----------------------------------------------
            # MONTHLY TREND
            # -----------------------------------------------

            datetime_columns = (
                analysis[
                    "datetime_columns"
                ]
            )

            if datetime_columns:

                date_column = (
                    datetime_columns[0]
                )

                monthly_df = cleaned_df.copy()

                monthly_df[
                    date_column
                ] = safe_to_datetime(
                    monthly_df[date_column]
                )

                monthly_df[
                    sales_column
                ] = pd.to_numeric(
                    monthly_df[sales_column],
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
                        Month=monthly_df[
                            date_column
                        ]
                        .dt.to_period("M")
                        .astype(str)
                    )
                    .groupby(
                        "Month"
                    )[sales_column]
                    .sum()
                )

                if len(monthly_sales) > 0:

                    st.markdown(
                        """
                        <div class="section-heading">
                            Monthly Sales Trend
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.line_chart(
                        monthly_sales,
                        height=320,
                    )


    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        st.markdown(
            """
            <div class="section-heading">
                Power BI Export Center
            </div>

            <div class="section-subheading">
                Download cleaned data and supporting business
                tables prepared for Power BI Desktop.
            </div>
            """,
            unsafe_allow_html=True,
        )

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis[
                "sales_column"
            ],
        )

        st.markdown(
            """
            <div class="notice notice-blue">
                <b>Recommended workflow:</b>
                Download the complete package, extract the CSV
                files, then import them into Power BI Desktop
                using Get Data → Text/CSV.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-heading">
                Available Datasets
            </div>
            """,
            unsafe_allow_html=True,
        )

        for filename, data in exports.items():

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-title">
                            {filename}
                        </div>
                        <div class="card-subtitle">
                            Power BI-ready CSV dataset
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:

                st.download_button(
                    "Download",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                )

        st.markdown(
            """
            <div class="section-heading">
                Complete Export Package
            </div>
            """,
            unsafe_allow_html=True,
        )

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "Download Power BI Package",
            data=zip_data,
            file_name="DataGuard_PowerBI_Exports.zip",
            mime="application/zip",
            type="primary",
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif page == "AI Analysis":

        st.markdown(
            """
            <div class="section-heading">
                AI Analysis
            </div>

            <div class="section-subheading">
                Gemini interpretation based strictly on
                DataGuard's measured quality signals.
            </div>
            """,
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

IQR outliers and ML anomalies are screening signals,
not automatically confirmed errors.

Business visualization filtering does not delete records.
"""

        if not GEMINI_API_KEY:

            st.markdown(
                """
                <div class="notice notice-yellow">
                    Gemini API key is not configured.
                    Add <b>GEMINI_API_KEY</b> to Streamlit
                    secrets to enable AI analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "Generate AI Analysis",
            type="primary",
        ):

            with st.spinner(
                "Gemini is analyzing the DataGuard report..."
            ):

                result = get_gemini_analysis(
                    summary_text
                )

            st.session_state.gemini_analysis = (
                result
            )

        if st.session_state.gemini_analysis:

            st.markdown(
                """
                <div class="card">
                    <div class="card-title">
                        AI Assessment
                    </div>
                    <div class="card-subtitle">
                        Generated from the measured DataGuard report
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )

        else:

            st.markdown(
                """
                <div class="empty-hero">
                    <div class="empty-icon">
                        AI
                    </div>

                    <div class="empty-title">
                        Turn quality signals into insights
                    </div>

                    <div class="empty-description">
                        Generate a concise portfolio-friendly
                        interpretation of the detected quality
                        problems, outliers, anomalies, cleaning
                        results and business impact.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.markdown(
            """
            <div class="section-heading">
                Data Quality Report
            </div>

            <div class="section-subheading">
                Portfolio-ready summary of the complete
                DataGuard AI analysis.
            </div>
            """,
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
            "Download Data Quality Report",
            data=report_text.encode(
                "utf-8"
            ),
            file_name="DataGuard_Data_Quality_Report.txt",
            mime="text/plain",
            type="primary",
        )

        if st.session_state.gemini_analysis:

            st.markdown(
                """
                <div class="section-heading">
                    Gemini Analysis
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        """
        <div class="empty-hero">
            <div class="empty-icon">
                🛡️
            </div>

            <div class="empty-title">
                Analyze your data with confidence
            </div>

            <div class="empty-description">
                Upload a CSV or Excel dataset to profile
                its structure, identify quality problems,
                detect unusual records, clean the data
                and prepare it for business analytics.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-heading">
            What DataGuard AI checks
        </div>

        <div class="section-subheading">
            One workflow from raw dataset to analysis-ready data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">
                    🔍
                </div>

                <div class="feature-title">
                    Profile Data
                </div>

                <div class="feature-text">
                    Understand columns, types, missing
                    values, unique values and identifiers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">
                    ◈
                </div>

                <div class="feature-title">
                    Detect Anomalies
                </div>

                <div class="feature-text">
                    Combine IQR statistical detection
                    with Isolation Forest ML screening.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">
                    🧹
                </div>

                <div class="feature-title">
                    Clean Data
                </div>

                <div class="feature-text">
                    Remove duplicates, standardize cities
                    and handle missing values.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">
                    📊
                </div>

                <div class="feature-title">
                    Analyze & Export
                </div>

                <div class="feature-text">
                    Explore business trends and export
                    Power BI-ready datasets.
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
        <br><br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
