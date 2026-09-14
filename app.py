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

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f6f8fc;
        color: #0f172a;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] > div {
        background: #0b1220;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .sidebar-brand {
        padding: 8px 4px 22px 4px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        width: 43px;
        height: 43px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );
        font-size: 21px;
        box-shadow:
            0 8px 25px rgba(37, 99, 235, 0.25);
    }

    .brand-name {
        color: #ffffff;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.4px;
    }

    .brand-caption {
        color: #94a3b8 !important;
        font-size: 10px;
        margin-top: 2px;
    }

    .sidebar-section {
        color: #64748b !important;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-top: 19px;
        margin-bottom: 7px;
    }

    .pipeline-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px;
        margin-top: 8px;
    }

    .pipeline-step {
        color: #94a3b8 !important;
        font-size: 10px;
        padding: 4px 0;
    }

    .pipeline-step span {
        color: #475569 !important;
        font-weight: 800;
        margin-right: 7px;
    }

    .sidebar-note {
        color: #64748b !important;
        font-size: 10px;
        line-height: 1.5;
    }

    .sidebar-footer {
        color: #475569 !important;
        font-size: 9px;
        line-height: 1.5;
        margin-top: 20px;
    }

    /* ======================================================
       SIDEBAR RADIO
       ====================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] label {
        background: transparent;
        border-radius: 9px;
        padding: 7px 9px;
        margin: 2px 0;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stRadio"] label:hover {
        background: #172033;
    }

    /* ======================================================
       PAGE HEADER
       ====================================================== */

    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .page-title {
        color: #0f172a;
        font-size: 32px;
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: -1.2px;
        margin: 0;
    }

    .page-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-top: 7px;
        line-height: 1.5;
    }

    .engine-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 10px;
        font-weight: 800;
    }

    .engine-dot {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
    }

    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 52%,
                #312e81 100%
            );
        border-radius: 20px;
        padding: 30px;
        color: white;
        position: relative;
        overflow: hidden;
        margin-bottom: 24px;
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.14);
    }

    .hero:before {
        content: "";
        position: absolute;
        width: 320px;
        height: 320px;
        right: -130px;
        top: -170px;
        border-radius: 50%;
        background: rgba(96, 165, 250, 0.09);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: 100px;
        bottom: -190px;
        border-radius: 50%;
        background: rgba(124, 58, 237, 0.10);
    }

    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 760px;
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
        font-size: 28px;
        font-weight: 850;
        letter-spacing: -0.8px;
        margin-top: 7px;
    }

    .hero-text {
        color: #cbd5e1;
        font-size: 13px;
        line-height: 1.65;
        margin-top: 7px;
        max-width: 680px;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin-top: 17px;
    }

    .hero-tag {
        color: #dbeafe;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 8px;
        padding: 6px 9px;
        font-size: 9px;
        font-weight: 700;
    }

    /* ======================================================
       UPLOAD AREA
       ====================================================== */

    .upload-card {
        background: #ffffff;
        border: 1px dashed #94a3b8;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 24px;
    }

    .upload-icon {
        width: 48px;
        height: 48px;
        margin: 0 auto 10px auto;
        border-radius: 13px;
        background: #eff6ff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 21px;
    }

    .upload-title {
        color: #0f172a;
        font-size: 16px;
        font-weight: 800;
    }

    .upload-text {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
    }

    .upload-meta {
        color: #94a3b8;
        font-size: 9px;
        margin-top: 9px;
    }

    /* ======================================================
       SECTION HEADERS
       ====================================================== */

    .section-title {
        color: #0f172a;
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-top: 25px;
        margin-bottom: 3px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 11px;
        margin-bottom: 15px;
    }

    /* ======================================================
       CARDS
       ====================================================== */

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 19px;
        box-shadow:
            0 4px 16px rgba(15, 23, 42, 0.035);
        margin-bottom: 15px;
    }

    .card-title {
        color: #111827;
        font-size: 14px;
        font-weight: 800;
    }

    .card-subtitle {
        color: #94a3b8;
        font-size: 10px;
        margin-top: 3px;
    }

    /* ======================================================
       KPI
       ====================================================== */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        min-height: 112px;
        box-shadow:
            0 4px 14px rgba(15, 23, 42, 0.035);
    }

    .kpi-label {
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 26px;
        font-weight: 850;
        letter-spacing: -0.7px;
        margin-top: 9px;
    }

    .kpi-caption {
        color: #94a3b8;
        font-size: 9px;
        margin-top: 2px;
    }

    /* ======================================================
       SCORE
       ====================================================== */

    .score-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 22px;
        min-height: 210px;
        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.04);
    }

    .score-label {
        color: #64748b;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .score-number {
        color: #0f172a;
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -2px;
        margin-top: 9px;
    }

    .score-number span {
        color: #94a3b8;
        font-size: 16px;
        font-weight: 600;
    }

    .score-badge {
        display: inline-block;
        margin-top: 7px;
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 9px;
        font-weight: 800;
    }

    .badge-good {
        background: #dcfce7;
        color: #166534;
    }

    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }

    .score-track {
        width: 100%;
        height: 7px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-top: 17px;
    }

    .score-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );
    }

    /* ======================================================
       FEATURE CARDS
       ====================================================== */

    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 19px;
        min-height: 145px;
        box-shadow:
            0 4px 15px rgba(15, 23, 42, 0.03);
    }

    .feature-icon {
        width: 36px;
        height: 36px;
        background: #eff6ff;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        margin-bottom: 12px;
    }

    .feature-title {
        color: #111827;
        font-size: 13px;
        font-weight: 800;
    }

    .feature-text {
        color: #64748b;
        font-size: 10px;
        line-height: 1.55;
        margin-top: 5px;
    }

    /* ======================================================
       DATASET STRIP
       ====================================================== */

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
        font-size: 12px;
        font-weight: 800;
    }

    .dataset-meta {
        color: #64748b;
        font-size: 10px;
        margin-top: 3px;
    }

    /* ======================================================
       NOTICES
       ====================================================== */

    .notice {
        border-radius: 10px;
        padding: 11px 13px;
        font-size: 11px;
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

    /* ======================================================
       TABLE
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 700;
        min-height: 37px;
    }

    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 9px;
        margin-top: 50px;
        padding-top: 18px;
        border-top: 1px solid #e5e7eb;
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

    result = []

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

            result.append(column)
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            continue

        name = str(column).lower()

        parsed = safe_to_datetime(
            df[column]
        )

        ratio = parsed.notna().mean()

        if any(
            token in name
            for token in tokens
        ):

            if ratio >= 0.50:
                result.append(column)

        elif ratio >= 0.95:

            result.append(column)

    return list(dict.fromkeys(result))


def detect_identifier_columns(
    df,
    datetime_columns=None,
):

    if datetime_columns is None:
        datetime_columns = []

    result = []

    tokens = [
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
            for token in tokens
        ):

            result.append(column)
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

                result.append(column)

    return list(dict.fromkeys(result))


def profile_dataset(df):

    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[c].dtype)
                for c in df.columns
            ],
            "Non-Null": [
                int(df[c].notna().sum())
                for c in df.columns
            ],
            "Missing": [
                int(df[c].isna().sum())
                for c in df.columns
            ],
            "Unique": [
                int(
                    df[c].nunique(
                        dropna=True
                    )
                )
                for c in df.columns
            ],
        }
    )


def detect_invalid_values(df):

    details = []
    total_invalid = 0

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

            total_invalid += count

            details.append(
                {
                    "Column": column,
                    "Invalid Values": count,
                }
            )

    return (
        total_invalid,
        pd.DataFrame(details),
    )


def standardize_city_column(df):

    result = df.copy()

    total_changes = 0

    city_columns = [
        c
        for c in result.columns
        if "city" in str(c).lower()
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
            original
            .astype("string")
            .fillna("")
            != standardized
            .fillna("")
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
    total = 0

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

            total += count

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
        total,
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

    mask = (
        predictions == -1
    )

    return (
        pd.Series(
            mask,
            index=df.index,
        ),
        int(mask.sum()),
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
        * 100
    )

    duplicate_penalty = (
        duplicate_count
        / max(1, total_cells)
        * 100
    )

    invalid_penalty = (
        invalid_count
        / max(1, total_cells)
        * 100
    )

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

            modes = (
                cleaned[column]
                .mode(dropna=True)
            )

            if len(modes) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        modes.iloc[0]
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

    preferred = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in preferred:

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

    sales_column = find_sales_column(
        df
    )

    if sales_column is None:

        return (
            None,
            None,
            None,
        )

    business_df = df.copy()

    business_df[
        sales_column
    ] = pd.to_numeric(
        business_df[
            sales_column
        ],
        errors="coerce",
    )

    business_df = business_df[
        business_df[
            sales_column
        ].notna()
    ]

    business_df = business_df[
        business_df[
            sales_column
        ] >= 0
    ]

    if len(business_df) == 0:

        return (
            None,
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
        business_df[
            sales_column
        ] <= upper_bound
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

        city_columns = [
            c
            for c in cleaned_df.columns
            if "city" in str(c).lower()
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

        category_columns = [
            c
            for c in cleaned_df.columns
            if "category"
            in str(c).lower()
        ]

        if category_columns:

            category_column = (
                category_columns[0]
            )

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

        product_columns = [
            c
            for c in cleaned_df.columns
            if "product"
            in str(c).lower()
        ]

        if product_columns:

            product_column = (
                product_columns[0]
            )

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

        datetime_columns = (
            detect_datetime_columns(
                cleaned_df
            )
        )

        if datetime_columns:

            date_column = (
                datetime_columns[0]
            )

            monthly_df = cleaned_df.copy()

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

            monthly_summary = (
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


def get_gemini_analysis(
    summary_text
):

    if not GEMINI_API_KEY:

        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior data quality analyst.

Analyze ONLY the supplied DataGuard AI report.

Do not invent facts.

Do not claim a possible root cause is confirmed.

IQR outliers and Isolation Forest anomalies
are screening signals, not confirmed errors.

The cleaning process has already been performed.

Extreme sales values excluded from business charts
were excluded ONLY for visualization and were NOT
deleted from the analytical dataset.

Create a concise professional analysis with:

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
            f"Reason: {error}"
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

    contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
        step=1,
    )

    if (
        contamination
        != st.session_state.contamination_pct
    ):

        st.session_state.contamination_pct = (
            contamination
        )

        if st.session_state.df is not None:

            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

        st.rerun()

    st.markdown(
        """
        <div class="sidebar-note">
            Higher sensitivity flags more records as unusual.
            ML anomalies are screening signals.
        </div>
        """,
        unsafe_allow_html=True,
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
    <div class="page-header">

        <div>
            <div class="eyebrow">
                Data Intelligence Workspace
            </div>

            <div class="page-title">
                DataGuard AI
            </div>

            <div class="page-subtitle">
                AI-powered data quality, anomaly detection
                and business analytics
            </div>
        </div>

        <div class="engine-pill">
            <span class="engine-dot"></span>
            Data Quality Engine
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LANDING / UPLOAD
# ============================================================

if st.session_state.df is None:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-content">

                <div class="hero-label">
                    DATA QUALITY PLATFORM
                </div>

                <div class="hero-title">
                    Protect the quality of your data
                </div>

                <div class="hero-text">
                    Upload your dataset and let DataGuard AI
                    profile, validate, detect anomalies, clean
                    and prepare your data for business analytics.
                </div>

                <div class="hero-tags">
                    <div class="hero-tag">CSV</div>
                    <div class="hero-tag">XLSX</div>
                    <div class="hero-tag">Data Profiling</div>
                    <div class="hero-tag">IQR Detection</div>
                    <div class="hero-tag">Isolation Forest</div>
                    <div class="hero-tag">Power BI Ready</div>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-card">

            <div class="upload-icon">
                ↑
            </div>

            <div class="upload-title">
                Upload your dataset
            </div>

            <div class="upload-text">
                Drag and drop your CSV or Excel file,
                or browse your computer.
            </div>

            <div class="upload-meta">
                CSV • XLSX • XLS &nbsp; • &nbsp;
                Maximum 200 MB
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a dataset",
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
            st.session_state.analysis = None
            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )

    st.markdown(
        """
        <div class="section-title">
            What DataGuard AI checks
        </div>

        <div class="section-subtitle">
            One workflow from raw dataset to analysis-ready data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4)

    with f1:

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
                    Understand columns, data types,
                    missing values, unique values
                    and identifiers.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with f2:

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

    with f3:

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
                    Remove duplicates, standardize
                    city values and handle missing data.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with f4:

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
                    Explore business trends and create
                    Power BI-ready datasets.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ANALYSIS
# ============================================================

if (
    st.session_state.df is not None
    and not st.session_state.analysis_complete
):

    df = st.session_state.df

    with st.spinner(
        "Running DataGuard AI quality engine..."
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

        iqr_count, iqr_details = (
            detect_iqr_outliers(df)
        )

        anomaly_mask, ml_count = (
            detect_ml_anomalies(
                df,
                st.session_state.contamination_pct,
            )
        )

        normal_count = (
            rows - ml_count
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
            "iqr_outlier_count": iqr_count,
            "iqr_details": iqr_details,
            "anomaly_mask": anomaly_mask,
            "ml_anomaly_count": ml_count,
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
# APP PAGES
# ============================================================

if (
    st.session_state.df is not None
    and st.session_state.analysis_complete
):

    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df
    business_df = st.session_state.business_df
    analysis = st.session_state.analysis

    sensitivity = (
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
                    {analysis["rows"]:,} rows
                    &nbsp; • &nbsp;
                    {analysis["columns"]} columns
                    &nbsp; • &nbsp;
                    Analysis complete
                </div>
            </div>

            <div class="engine-pill">
                <span class="engine-dot"></span>
                Ready
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
            <div class="section-title">
                Dashboard
            </div>

            <div class="section-subtitle">
                Monitor dataset health and quality signals.
            </div>
            """,
            unsafe_allow_html=True,
        )

        score = analysis[
            "quality_score"
        ]

        if score >= 95:

            status = "Good"
            badge_class = "badge-good"

        elif score >= 85:

            status = "Needs Attention"
            badge_class = "badge-warning"

        else:

            status = "Critical"
            badge_class = "badge-danger"

        left, right = st.columns(
            [1, 2.1]
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
                        <span>/100</span>
                    </div>

                    <div class="score-badge {badge_class}">
                        {status}
                    </div>

                    <div class="score-track">
                        <div
                            class="score-fill"
                            style="width:{score}%"
                        ></div>
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:

            r1, r2 = st.columns(2)

            with r1:

                st.markdown(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Missing Cells
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

            with r2:

                st.markdown(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            Duplicate Records
                        </div>

                        <div class="kpi-value">
                            {analysis["duplicate_count"]:,}
                        </div>

                        <div class="kpi-caption">
                            Duplicate rows
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            r3, r4 = st.columns(2)

            with r3:

                st.markdown(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            IQR Outliers
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

            with r4:

                st.markdown(
                    f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            ML Anomalies
                        </div>

                        <div class="kpi-value">
                            {analysis["ml_anomaly_count"]:,}
                        </div>

                        <div class="kpi-caption">
                            {sensitivity}% sensitivity
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="section-title">
                Dataset Profile
            </div>

            <div class="section-subtitle">
                Structural overview of the uploaded data.
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
            <div class="section-title">
                Quality Monitoring
            </div>

            <div class="section-subtitle">
                Detected quality and anomaly signals.
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

            sales_column = analysis[
                "sales_column"
            ]

            st.markdown(
                """
                <div class="section-title">
                    Business Snapshot
                </div>

                <div class="section-subtitle">
                    Commercial metrics detected in the dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2, b3 = st.columns(3)

            b1.metric(
                "Total Sales",
                f'{business_df[sales_column].sum():,.0f}',
            )

            b2.metric(
                "Average Sale",
                f'{business_df[sales_column].mean():,.2f}',
            )

            b3.metric(
                "Sales Records",
                f'{len(business_df):,}',
            )

        st.markdown(
            """
            <div class="section-title">
                Data Preview
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
            <div class="section-title">
                Data Quality
            </div>

            <div class="section-subtitle">
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
            <div class="section-title">
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
            <div class="section-title">
                Detected Data Types
            </div>
            """,
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Numeric Columns",
            analysis["numeric_count"],
        )

        d2.metric(
            "Categorical Columns",
            analysis["categorical_count"],
        )

        d3.metric(
            "Date / Time Columns",
            len(
                analysis[
                    "datetime_columns"
                ]
            ),
        )

        with st.expander(
            "Date / Time Columns"
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
                    "No date/time columns detected."
                )

        with st.expander(
            "Identifier Columns"
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
            <div class="section-title">
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
                    No invalid values detected by the
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
            f"""
            <div class="notice notice-blue">
                City standardization signals:
                <b>{analysis["city_changes"]:,}</b>
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
            <div class="section-title">
                Anomaly Detection
            </div>

            <div class="section-subtitle">
                Statistical and machine-learning screening signals.
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
                IQR outliers and Isolation Forest anomalies
                indicate unusual observations. They are not
                automatically confirmed data errors.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-title">
                IQR Statistical Detection
            </div>
            """,
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
            """
            <div class="section-title">
                Isolation Forest
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="notice notice-blue">
                Current model sensitivity:
                <b>{sensitivity}%</b>.
                Change it from the sidebar to rerun the model.
            </div>
            """,
            unsafe_allow_html=True,
        )

        anomaly_df = df.copy()

        anomaly_df[
            "ML_Anomaly"
        ] = np.where(
            analysis[
                "anomaly_mask"
            ],
            "Anomaly",
            "Normal",
        )

        anomaly_records = (
            anomaly_df[
                anomaly_df[
                    "ML_Anomaly"
                ]
                == "Anomaly"
            ]
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
            <div class="section-title">
                Automated Cleaning
            </div>

            <div class="section-subtitle">
                Transform raw data into an analysis-ready dataset.
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
                Cleaning completed. Duplicate records were
                removed, city values standardized, numeric
                missing values filled using medians and
                categorical missing values filled using modes.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-title">
                Cleaned Dataset
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
            <div class="section-title">
                Business Analytics
            </div>

            <div class="section-subtitle">
                Explore trends and commercial patterns
                from the cleaned dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if business_df is None:

            st.warning(
                "No Sales, Revenue or Amount column was detected."
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
                    Visualization filtering uses an IQR upper
                    bound of <b>{upper_bound:,.2f}</b>.
                    This affects charts only and does not delete
                    records from the cleaned dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

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

            st.markdown(
                """
                <div class="section-title">
                    Sales Distribution
                </div>
                """,
                unsafe_allow_html=True,
            )

            if len(
                business_df
            ) > 1:

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
                            f"{bins[i + 1]:,.0f}"
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

            left, right = st.columns(2)

            with left:

                city_columns = [
                    c
                    for c in cleaned_df.columns
                    if "city"
                    in str(c).lower()
                ]

                if city_columns:

                    city_column = (
                        city_columns[0]
                    )

                    city_sales = (
                        business_df
                        .groupby(
                            city_column
                        )[
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
                                Top locations
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
                    c
                    for c in cleaned_df.columns
                    if "product"
                    in str(c).lower()
                ]

                if product_columns:

                    product_column = (
                        product_columns[0]
                    )

                    product_sales = (
                        business_df
                        .groupby(
                            product_column
                        )[
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
                                Top products
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        product_sales,
                        height=300,
                    )

            category_columns = [
                c
                for c in cleaned_df.columns
                if "category"
                in str(c).lower()
            ]

            if category_columns:

                category_column = (
                    category_columns[0]
                )

                category_sales = (
                    business_df
                    .groupby(
                        category_column
                    )[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    """
                    <div class="section-title">
                        Sales by Category
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.bar_chart(
                    category_sales,
                    height=300,
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

                monthly_df = cleaned_df.copy()

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
                        Month=monthly_df[
                            date_column
                        ]
                        .dt.to_period("M")
                        .astype(str)
                    )
                    .groupby(
                        "Month"
                    )[
                        sales_column
                    ]
                    .sum()
                )

                if len(
                    monthly_sales
                ) > 0:

                    st.markdown(
                        """
                        <div class="section-title">
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
            <div class="section-title">
                Power BI Export Center
            </div>

            <div class="section-subtitle">
                Business-ready datasets prepared for Power BI.
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
                Import these CSV files into Power BI Desktop
                using <b>Get Data → Text/CSV</b>.
                The ZIP package contains the complete export set.
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
                            Power BI-ready CSV
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
            <div class="section-title">
                Complete Package
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
            <div class="section-title">
                AI Analysis
            </div>

            <div class="section-subtitle">
                Turn measured DataGuard signals into
                an understandable professional assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_text = f"""
DataGuard AI Report

Dataset:
{st.session_state.file_name}

Rows:
{analysis["rows"]}

Columns:
{analysis["columns"]}

Total Cells:
{analysis["total_cells"]}

Missing:
{analysis["missing_count"]}

Duplicates:
{analysis["duplicate_count"]}

Invalid:
{analysis["invalid_count"]}

Quality Score:
{analysis["quality_score"]}/100

Numeric Columns:
{analysis["numeric_count"]}

Categorical Columns:
{analysis["categorical_count"]}

Date/Time Columns:
{analysis["datetime_columns"]}

Identifier Columns:
{analysis["identifier_columns"]}

IQR Outliers:
{analysis["iqr_outlier_count"]}

Isolation Forest Sensitivity:
{sensitivity}%

ML Anomalies:
{analysis["ml_anomaly_count"]}

Normal Records:
{analysis["normal_count"]}

Rows Removed:
{analysis["rows_removed"]}

Values Filled:
{analysis["values_filled"]}

City Values Standardized:
{analysis["city_changes"]}

Sales Column:
{analysis["sales_column"]}

Business Chart IQR Upper Bound:
{analysis["upper_bound"]}
"""

        if not GEMINI_API_KEY:

            st.markdown(
                """
                <div class="notice notice-yellow">
                    Gemini is not configured.
                    Add <b>GEMINI_API_KEY</b> to your
                    Streamlit secrets to enable AI analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(
            "Generate AI Analysis",
            type="primary",
        ):

            with st.spinner(
                "Generating AI assessment..."
            ):

                st.session_state.gemini_analysis = (
                    get_gemini_analysis(
                        summary_text
                    )
                )

        if st.session_state.gemini_analysis:

            st.markdown(
                """
                <div class="card">
                    <div class="card-title">
                        AI Assessment
                    </div>

                    <div class="card-subtitle">
                        Based on measured DataGuard signals
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
                <div class="card">

                    <div class="feature-icon">
                        AI
                    </div>

                    <div class="feature-title">
                        Generate an intelligent assessment
                    </div>

                    <div class="feature-text">
                        DataGuard will provide a concise
                        interpretation of data quality problems,
                        statistical outliers, ML anomalies,
                        cleaning results and business impact.
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
            <div class="section-title">
                Data Quality Report
            </div>

            <div class="section-subtitle">
                Complete analysis summary for documentation
                and portfolio presentation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        report_text = f"""
DATAGUARD AI — DATA QUALITY REPORT

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET
-------
File: {st.session_state.file_name}
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

ANOMALIES
---------
IQR Outliers: {analysis["iqr_outlier_count"]:,}
Isolation Forest Sensitivity: {sensitivity}%
ML Anomalies: {analysis["ml_anomaly_count"]:,}
Normal Records: {analysis["normal_count"]:,}

CLEANING
--------
Original Rows: {analysis["rows"]:,}
Cleaned Rows: {len(cleaned_df):,}
Rows Removed: {analysis["rows_removed"]:,}
Values Filled: {analysis["values_filled"]:,}
City Values Standardized: {analysis["city_changes"]:,}

BUSINESS ANALYTICS
------------------
Sales Column: {analysis["sales_column"]}
Visualization IQR Upper Bound: {analysis["upper_bound"]}

INTERPRETATION
--------------
IQR outliers and Isolation Forest anomalies are
screening signals and are not automatically confirmed
data errors.

Extreme sales values excluded from business charts
were excluded only for visualization and were not
deleted from the cleaned dataset.
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
                <div class="section-title">
                    AI Assessment
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ <b>DataGuard AI</b>
        &nbsp; • &nbsp;
        AI Data Quality & Anomaly Detection Platform
        <br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
