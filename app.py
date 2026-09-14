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
# PREMIUM DARK UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #080b12;
        color: #e5e7eb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: #080b12;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0d111a;
        border-right: 1px solid #202735;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] * {
        color: #d1d5db;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
    }

    /* ---------- TEXT ---------- */

    .dg-title {
        font-size: 31px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .dg-subtitle {
        color: #8b95a7;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .dg-section {
        font-size: 19px;
        font-weight: 750;
        color: #f1f5f9;
        margin-top: 26px;
        margin-bottom: 4px;
    }

    .dg-section-sub {
        color: #7f8a9d;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* ---------- CARDS ---------- */

    .dg-card {
        background: #111722;
        border: 1px solid #202938;
        border-radius: 15px;
        padding: 19px;
        margin-bottom: 14px;
    }

    .dg-card-title {
        color: #f1f5f9;
        font-size: 15px;
        font-weight: 700;
    }

    .dg-card-subtitle {
        color: #7f8a9d;
        font-size: 12px;
        margin-top: 4px;
    }

    /* ---------- KPI ---------- */

    .dg-kpi {
        background: #111722;
        border: 1px solid #202938;
        border-radius: 15px;
        padding: 17px;
        min-height: 122px;
    }

    .dg-kpi:hover {
        border-color: #344155;
    }

    .dg-kpi-label {
        color: #8994a7;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: .7px;
        text-transform: uppercase;
    }

    .dg-kpi-value {
        color: #f8fafc;
        font-size: 26px;
        font-weight: 800;
        margin-top: 9px;
    }

    .dg-kpi-note {
        color: #667085;
        font-size: 11px;
        margin-top: 3px;
    }

    /* ---------- BADGES ---------- */

    .dg-badge-good {
        display: inline-block;
        background: #0d2b20;
        border: 1px solid #14532d;
        color: #86efac;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    .dg-badge-warning {
        display: inline-block;
        background: #2b2110;
        border: 1px solid #713f12;
        color: #fcd34d;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    .dg-badge-danger {
        display: inline-block;
        background: #2c1317;
        border: 1px solid #7f1d1d;
        color: #fca5a5;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    .dg-badge-info {
        display: inline-block;
        background: #101f38;
        border: 1px solid #1e40af;
        color: #93c5fd;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }

    /* ---------- SCORE ---------- */

    .score-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 16px;
    }

    .score-ring {
        width: 175px;
        height: 175px;
        border-radius: 50%;
        border: 12px solid #243047;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .score-number {
        color: #f8fafc;
        font-size: 39px;
        font-weight: 850;
    }

    .score-label {
        color: #8b95a7;
        font-size: 12px;
    }

    /* ---------- PIPELINE ---------- */

    .pipeline {
        display: flex;
        align-items: center;
        gap: 7px;
        overflow-x: auto;
        padding: 8px 0 18px 0;
    }

    .pipeline-step {
        min-width: 105px;
        text-align: center;
        padding: 9px 8px;
        border-radius: 10px;
        background: #111722;
        border: 1px solid #202938;
        color: #778196;
        font-size: 11px;
    }

    .pipeline-step.active {
        border-color: #6366f1;
        background: #17172d;
        color: #c7d2fe;
    }

    .pipeline-step.done {
        border-color: #14532d;
        background: #0e2119;
        color: #86efac;
    }

    .pipeline-arrow {
        color: #4b5563;
    }

    /* ---------- STATUS ---------- */

    .system-status {
        background: #0e2119;
        border: 1px solid #14532d;
        border-radius: 10px;
        padding: 10px 12px;
        color: #86efac;
        font-size: 12px;
    }

    .notice {
        background: #111b2c;
        border: 1px solid #263b5b;
        border-radius: 11px;
        padding: 13px 15px;
        color: #a9c7f7;
        font-size: 13px;
        margin: 8px 0 15px 0;
    }

    .notice-warning {
        background: #241c0c;
        border: 1px solid #684d16;
        color: #f5d47b;
    }

    .notice-success {
        background: #0d2118;
        border: 1px solid #14532d;
        color: #86efac;
    }

    /* ---------- EMPTY STATE ---------- */

    .empty-state {
        background: #0f141e;
        border: 1px dashed #334155;
        border-radius: 18px;
        padding: 55px 30px;
        text-align: center;
        margin-top: 15px;
    }

    .empty-icon {
        font-size: 46px;
        margin-bottom: 10px;
    }

    .empty-title {
        color: #f1f5f9;
        font-size: 21px;
        font-weight: 750;
    }

    .empty-text {
        color: #7f8a9d;
        font-size: 13px;
        max-width: 560px;
        margin: 8px auto;
    }

    /* ---------- UPLOAD ---------- */

    .upload-box {
        background: #101621;
        border: 1px dashed #46546a;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }

    .upload-title {
        color: #f1f5f9;
        font-size: 17px;
        font-weight: 700;
    }

    .upload-subtitle {
        color: #7f8a9d;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ---------- ACTIVITY ---------- */

    .activity {
        border-left: 2px solid #293548;
        padding-left: 15px;
        margin: 12px 0;
    }

    .activity-title {
        color: #dbe4f0;
        font-size: 13px;
        font-weight: 650;
    }

    .activity-time {
        color: #667085;
        font-size: 11px;
        margin-top: 3px;
    }

    /* ---------- BUTTONS ---------- */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid #303b4d;
        background: #171e2b;
        color: #e5e7eb;
        font-weight: 650;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #6366f1;
        color: white;
        background: #1b2132;
    }

    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div {
        background: #111722;
        border-color: #303b4d;
    }

    .stTextInput input {
        background: #111722;
        color: #e5e7eb;
        border-color: #303b4d;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #202938;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background: #111722;
        border: 1px solid #202938;
        border-radius: 13px;
        padding: 13px;
    }

    [data-testid="stMetricLabel"] {
        color: #8994a7 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* ---------- EXPANDER ---------- */

    [data-testid="stExpander"] {
        background: #111722;
        border: 1px solid #202938;
        border-radius: 12px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #596579;
        font-size: 11px;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #202938;
    }

    @media (max-width: 768px) {

        .main .block-container {
            padding: 1rem;
        }

        .dg-title {
            font-size: 25px;
        }

        .pipeline {
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

defaults = {
    "df": None,
    "cleaned_df": None,
    "business_df": None,
    "file_name": None,
    "analysis_complete": False,
    "analysis": None,
    "gemini_analysis": None,
    "contamination_pct": 5,
    "selected_page": "Dashboard",
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

        parsed = safe_to_datetime(
            df[column]
        )

        ratio = parsed.notna().mean()

        if any(
            token in name
            for token in tokens
        ):

            if ratio >= 0.50:
                datetime_columns.append(column)

        else:

            if ratio >= 0.95:
                datetime_columns.append(column)

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

                identifier_columns.append(
                    column
                )

    return list(
        dict.fromkeys(
            identifier_columns
        )
    )


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
                    df[column]
                    .notna()
                    .sum()
                )
                for column in df.columns
            ],
            "Missing": [
                int(
                    df[column]
                    .isna()
                    .sum()
                )
                for column in df.columns
            ],
            "Unique Values": [
                int(
                    df[column]
                    .nunique(
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


def standardize_city_column(df):

    result = df.copy()

    total_changes = 0

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

        changes = (
            original.astype("string")
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
        )
        .copy()
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
            min(
                100,
                score,
            ),
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

    sales_column = (
        find_sales_column(df)
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

    business_df = (
        business_df[
            business_df[
                sales_column
            ].notna()
        ]
    )

    business_df = (
        business_df[
            business_df[
                sales_column
            ] >= 0
        ]
    )

    if len(
        business_df
    ) == 0:

        return (
            None,
            sales_column,
            None,
        )

    q1 = (
        business_df[
            sales_column
        ].quantile(0.25)
    )

    q3 = (
        business_df[
            sales_column
        ].quantile(0.75)
    )

    iqr = q3 - q1

    upper_bound = (
        q3 + 1.5 * iqr
    )

    chart_df = (
        business_df[
            business_df[
                sales_column
            ] <= upper_bound
        ]
        .copy()
    )

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
            .groupby(
                sales_column
            )
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
        if "city"
        in str(column).lower()
    ]

    if city_columns:

        city_column = city_columns[0]

        city_summary = (
            cleaned_df
            .groupby(
                city_column
            )
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

        date_column = datetime_columns[0]

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
            monthly_df
            .dropna(
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

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    "",
)

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
# PAGE CONFIGURATION
# ============================================================

PAGES = [
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


def get_page_number(page):

    mapping = {
        "Upload Data": 1,
        "Data Profile": 2,
        "Quality Checks": 3,
        "Anomaly Detection": 4,
        "Data Cleaning": 5,
        "Analytics": 6,
        "Power BI": 7,
        "AI Analysis": 8,
    }

    return mapping.get(page, 0)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:5px 2px 18px 2px;">
            <div style="font-size:29px;">🛡️</div>
            <div style="
                color:#f8fafc;
                font-size:20px;
                font-weight:800;
                margin-top:4px;
            ">
                DataGuard AI
            </div>
            <div style="
                color:#718096;
                font-size:11px;
                margin-top:3px;
            ">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "##### WORKSPACE"
    )

    selected_page = st.radio(
        "Workspace",
        [
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
        ],
        index=PAGES.index(
            st.session_state.selected_page
        ),
        label_visibility="collapsed",
    )

    st.session_state.selected_page = selected_page

    st.markdown("---")

    st.markdown(
        "##### SYSTEM"
    )

    st.caption(
        "DataGuard AI analyzes the complete uploaded "
        "dataset. Charts may apply visualization-only "
        "filters."
    )

    if st.session_state.df is not None:

        st.markdown(
            f"""
            <div class="system-status">
                ● All systems operational
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.caption(
            f"Dataset: {st.session_state.file_name}"
        )

    else:

        st.markdown(
            """
            <div style="
                background:#17130b;
                border:1px solid #5b4515;
                border-radius:10px;
                padding:10px;
                color:#fcd34d;
                font-size:11px;
            ">
                ○ Waiting for dataset
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        "##### ANOMALY SETTINGS"
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
        "Higher sensitivity flags more records as unusual."
    )

    st.markdown("---")

    st.caption(
        "DataGuard AI • Portfolio Edition"
    )


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns(
    [8, 2]
)

with header_left:

    if st.session_state.df is not None:

        st.markdown(
            f"""
            <div style="
                color:#667085;
                font-size:11px;
                margin-bottom:4px;
            ">
                WORKSPACE / {selected_page.upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="dg-title">{selected_page}</div>',
        unsafe_allow_html=True,
    )

with header_right:

    if st.session_state.df is not None:

        st.markdown(
            f"""
            <div style="
                text-align:right;
                padding-top:8px;
                color:#7f8a9d;
                font-size:11px;
            ">
                🟢 LIVE
                <br>
                {st.session_state.file_name}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# WORKFLOW
# ============================================================

if st.session_state.df is not None:

    current_step = get_page_number(
        selected_page
    )

    steps = [
        "Upload",
        "Profile",
        "Quality",
        "Anomalies",
        "Clean",
        "Analytics",
        "Power BI",
        "AI",
    ]

    workflow_html = (
        '<div class="pipeline">'
    )

    for index, step in enumerate(
        steps,
        start=1,
    ):

        if current_step == index:

            cls = "active"
            symbol = f"{index:02d}"

        elif current_step > index:

            cls = "done"
            symbol = "✓"

        else:

            cls = ""
            symbol = f"{index:02d}"

        workflow_html += (
            f'<div class="pipeline-step {cls}">'
            f'<b>{symbol}</b><br>{step}'
            f'</div>'
        )

        if index < len(steps):

            workflow_html += (
                '<div class="pipeline-arrow">›</div>'
            )

    workflow_html += "</div>"

    st.markdown(
        workflow_html,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD PAGE / GLOBAL UPLOAD
# ============================================================

if selected_page == "Upload Data":

    st.markdown(
        """
        <div class="dg-subtitle">
            Upload a dataset and start a complete data-quality assessment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-box">
            <div style="font-size:36px;">☁️</div>
            <div class="upload-title">
                Drop your dataset here
            </div>
            <div class="upload-subtitle">
                CSV, XLSX or XLS • Maximum recommended size 50 MB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Browse files",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

else:

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
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

            st.success(
                f"Dataset uploaded successfully: "
                f"{uploaded_file.name}"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )

            st.stop()


# ============================================================
# NO DATA STATE
# ============================================================

if st.session_state.df is None:

    if selected_page != "Upload Data":

        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">🛡️</div>
                <div class="empty-title">
                    Start with your dataset
                </div>
                <div class="empty-text">
                    Upload a CSV or Excel file to profile your data,
                    identify quality issues, screen for anomalies,
                    clean the dataset and generate business analytics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            (
                "🔍",
                "Profile",
                "Understand structure, data types, missing values and identifiers.",
            ),
            (
                "✓",
                "Quality Checks",
                "Detect missing values, duplicates and invalid values.",
            ),
            (
                "⚡",
                "Anomalies",
                "Use IQR and Isolation Forest to screen unusual records.",
            ),
            (
                "📊",
                "Analytics",
                "Create business views and Power BI-ready exports.",
            ),
        ]

        for col, card in zip(
            [c1, c2, c3, c4],
            cards,
        ):

            with col:

                icon, title, text = card

                st.markdown(
                    f"""
                    <div class="dg-card">
                        <div style="font-size:25px;">
                            {icon}
                        </div>
                        <div class="dg-card-title"
                             style="margin-top:8px;">
                            {title}
                        </div>
                        <div class="dg-card-subtitle">
                            {text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:

        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">📁</div>
                <div class="empty-title">
                    Upload your first dataset
                </div>
                <div class="empty-text">
                    Supported formats: CSV, XLSX and XLS.
                    DataGuard AI will analyze the complete dataset.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ANALYSIS ENGINE
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if not st.session_state.analysis_complete:

        with st.spinner(
            "Analyzing the complete dataset..."
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

            st.session_state.cleaned_df = cleaned_df
            st.session_state.business_df = business_df

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
# MAIN APPLICATION
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

    score = analysis["quality_score"]

    # ========================================================
    # DASHBOARD
    # ========================================================

    if selected_page == "Dashboard":

        st.markdown(
            '<div class="dg-subtitle">'
            'Monitor, analyze and improve the quality of your dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="notice">
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

        # KPI ROW

        k1, k2, k3, k4, k5, k6 = st.columns(6)

        kpis = [
            (
                k1,
                "Data Quality",
                f"{score:.1f}",
                "/ 100",
            ),
            (
                k2,
                "Records",
                f'{analysis["rows"]:,}',
                "total rows",
            ),
            (
                k3,
                "Columns",
                f'{analysis["columns"]}',
                "fields",
            ),
            (
                k4,
                "Missing",
                f'{analysis["missing_count"]:,}',
                "cells",
            ),
            (
                k5,
                "Duplicates",
                f'{analysis["duplicate_count"]:,}',
                "records",
            ),
            (
                k6,
                "Anomalies",
                f'{analysis["ml_anomaly_count"]:,}',
                f"{contamination_pct}% sensitivity",
            ),
        ]

        for col, label, value, note in kpis:

            with col:

                st.markdown(
                    f"""
                    <div class="dg-kpi">
                        <div class="dg-kpi-label">
                            {label}
                        </div>
                        <div class="dg-kpi-value">
                            {value}
                        </div>
                        <div class="dg-kpi-note">
                            {note}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # QUALITY + ISSUES

        st.markdown(
            '<div class="dg-section">Quality Overview</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="dg-section-sub">'
            'A high-level view of dataset health.'
            '</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(
            [1, 1.6]
        )

        with left:

            st.markdown(
                '<div class="dg-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dg-card-title">'
                'Overall Quality Score'
                '</div>',
                unsafe_allow_html=True,
            )

            if score >= 95:
                status = "Excellent"
            elif score >= 85:
                status = "Good"
            elif score >= 70:
                status = "Needs Attention"
            else:
                status = "Critical"

            st.markdown(
                f"""
                <div class="score-wrap">
                    <div class="score-ring">
                        <div class="score-number">
                            {score:.0f}%
                        </div>
                        <div class="score-label">
                            {status}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:5px 10px;
                    color:#8b95a7;
                    font-size:12px;
                ">
                    <span>Completeness</span>
                    <b style="color:#e5e7eb;">
                        {max(0, 100 - (analysis["missing_count"] /
                        max(1, analysis["total_cells"]) * 100)):.1f}%
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:5px 10px;
                    color:#8b95a7;
                    font-size:12px;
                ">
                    <span>Uniqueness</span>
                    <b style="color:#e5e7eb;">
                        {max(0, 100 - (analysis["duplicate_count"] /
                        max(1, analysis["rows"]) * 100)):.1f}%
                    </b>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    padding:5px 10px;
                    color:#8b95a7;
                    font-size:12px;
                ">
                    <span>Validity</span>
                    <b style="color:#e5e7eb;">
                        {max(0, 100 - (analysis["invalid_count"] /
                        max(1, analysis["total_cells"]) * 100)):.1f}%
                    </b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        with right:

            st.markdown(
                '<div class="dg-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dg-card-title">'
                'Issues Detected'
                '</div>',
                unsafe_allow_html=True,
            )

            issue_rows = [
                (
                    "Missing Values",
                    analysis["missing_count"],
                    "warning",
                ),
                (
                    "Duplicate Records",
                    analysis["duplicate_count"],
                    "warning",
                ),
                (
                    "Invalid Values",
                    analysis["invalid_count"],
                    "danger",
                ),
                (
                    "Potential Outliers",
                    analysis["iqr_outlier_count"],
                    "info",
                ),
                (
                    "ML Anomalies",
                    analysis["ml_anomaly_count"],
                    "info",
                ),
            ]

            for title, value, badge in issue_rows:

                badge_class = (
                    "dg-badge-danger"
                    if badge == "danger"
                    else
                    "dg-badge-warning"
                    if badge == "warning"
                    else
                    "dg-badge-info"
                )

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        align-items:center;
                        justify-content:space-between;
                        padding:10px 0;
                        border-bottom:1px solid #202938;
                    ">
                        <span style="
                            color:#cbd5e1;
                            font-size:13px;
                        ">
                            {title}
                        </span>

                        <span class="{badge_class}">
                            {value:,}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        # ANOMALY PANEL

        st.markdown(
            '<div class="dg-section">Anomaly Detection</div>',
            unsafe_allow_html=True,
        )

        anomaly_left, anomaly_right = st.columns(
            [2, 1]
        )

        with anomaly_left:

            st.markdown(
                f"""
                <div class="dg-card">
                    <div class="dg-card-title">
                        Isolation Forest Screening
                    </div>
                    <div class="dg-card-subtitle">
                        ML-based screening for unusual records.
                    </div>

                    <div style="
                        display:flex;
                        gap:35px;
                        margin-top:22px;
                    ">
                        <div>
                            <div style="
                                color:#697586;
                                font-size:11px;
                            ">
                                NORMAL
                            </div>
                            <div style="
                                color:#86efac;
                                font-size:25px;
                                font-weight:800;
                            ">
                                {analysis["normal_count"]:,}
                            </div>
                        </div>

                        <div>
                            <div style="
                                color:#697586;
                                font-size:11px;
                            ">
                                ANOMALIES
                            </div>
                            <div style="
                                color:#fca5a5;
                                font-size:25px;
                                font-weight:800;
                            ">
                                {analysis["ml_anomaly_count"]:,}
                            </div>
                        </div>

                        <div>
                            <div style="
                                color:#697586;
                                font-size:11px;
                            ">
                                RATE
                            </div>
                            <div style="
                                color:#c4b5fd;
                                font-size:25px;
                                font-weight:800;
                            ">
                                {(analysis["ml_anomaly_count"] /
                                max(1, analysis["rows"]) * 100):.1f}%
                            </div>
                        </div>
                    </div>

                    <div style="
                        margin-top:20px;
                        color:#778196;
                        font-size:11px;
                    ">
                        Isolation Forest •
                        {contamination_pct}% sensitivity
                        <br>
                        ML anomalies are screening signals,
                        not confirmed errors.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with anomaly_right:

            st.markdown(
                '<div class="dg-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dg-card-title">'
                'Sensitivity'
                '</div>',
                unsafe_allow_html=True,
            )

            st.write("")

            new_slider = st.slider(
                "Detection sensitivity",
                1,
                20,
                contamination_pct,
                key="dashboard_sensitivity",
            )

            if new_slider != contamination_pct:

                st.session_state.contamination_pct = (
                    new_slider
                )

                st.session_state.analysis_complete = False
                st.session_state.gemini_analysis = None

                st.rerun()

            st.caption(
                "Increase sensitivity to flag more unusual records."
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        # CHARTS

        st.markdown(
            '<div class="dg-section">Quality Monitoring</div>',
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
            use_container_width=True,
        )

        # RECENT ACTIVITY

        st.markdown(
            '<div class="dg-section">Recent Activity</div>',
            unsafe_allow_html=True,
        )

        activities = [
            (
                "Dataset analyzed",
                f"{analysis['rows']:,} records and "
                f"{analysis['columns']} columns processed",
            ),
            (
                "Quality checks completed",
                f"{analysis['missing_count']:,} missing cells, "
                f"{analysis['duplicate_count']:,} duplicates detected",
            ),
            (
                "Anomaly screening completed",
                f"{analysis['ml_anomaly_count']:,} ML anomalies flagged",
            ),
            (
                "Cleaning preview generated",
                f"{analysis['rows_removed']:,} duplicate rows removable, "
                f"{analysis['values_filled']:,} values fillable",
            ),
        ]

        for title, text in activities:

            st.markdown(
                f"""
                <div class="activity">
                    <div class="activity-title">
                        {title}
                    </div>
                    <div class="activity-time">
                        {text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # PREVIEW

        st.markdown(
            '<div class="dg-section">Dataset Preview</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=350,
        )


    # ========================================================
    # UPLOAD DATA
    # ========================================================

    elif selected_page == "Upload Data":

        st.markdown(
            '<div class="dg-subtitle">'
            'Manage your current dataset and start a new analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="dg-card">
                <div class="dg-card-title">
                    Current Dataset
                </div>
                <div class="dg-card-subtitle">
                    {st.session_state.file_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        u1, u2, u3, u4 = st.columns(4)

        u1.metric(
            "Records",
            f'{analysis["rows"]:,}',
        )

        u2.metric(
            "Columns",
            analysis["columns"],
        )

        u3.metric(
            "File",
            st.session_state.file_name.split(".")[-1].upper(),
        )

        u4.metric(
            "Quality",
            f'{analysis["quality_score"]:.1f}',
        )

        st.markdown(
            '<div class="dg-section">Data Ingestion</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "Upload another file above to replace the current dataset. "
            "The previous analysis will automatically reset."
        )

        st.markdown(
            '<div class="dg-section">Current Preview</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(20),
            use_container_width=True,
            height=430,
        )


    # ========================================================
    # DATA PROFILE
    # ========================================================

    elif selected_page == "Data Profile":

        st.markdown(
            '<div class="dg-subtitle">'
            'Explore structure, types, completeness and uniqueness.'
            '</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)

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

        profile = profile_dataset(df)

        st.markdown(
            '<div class="dg-section">Column Profile</div>',
            unsafe_allow_html=True,
        )

        search_column = st.text_input(
            "Search columns",
            placeholder="Search by column name...",
        )

        if search_column:

            profile = profile[
                profile["Column"]
                .astype(str)
                .str.contains(
                    search_column,
                    case=False,
                    na=False,
                )
            ]

        st.dataframe(
            profile,
            use_container_width=True,
            height=480,
        )

        st.markdown(
            '<div class="dg-section">Detected Structure</div>',
            unsafe_allow_html=True,
        )

        d1, d2 = st.columns(2)

        with d1:

            with st.expander(
                "Date / Time columns",
                expanded=True,
            ):

                if analysis["datetime_columns"]:

                    for column in analysis[
                        "datetime_columns"
                    ]:

                        st.write(
                            f"✓ {column}"
                        )

                else:

                    st.caption(
                        "No date/time columns detected."
                    )

        with d2:

            with st.expander(
                "Identifier columns",
                expanded=True,
            ):

                if analysis["identifier_columns"]:

                    for column in analysis[
                        "identifier_columns"
                    ]:

                        st.write(
                            f"✓ {column}"
                        )

                else:

                    st.caption(
                        "No identifier columns detected."
                    )


    # ========================================================
    # QUALITY CHECKS
    # ========================================================

    elif selected_page == "Quality Checks":

        st.markdown(
            '<div class="dg-subtitle">'
            'Rule-based checks for completeness, validity, consistency and uniqueness.'
            '</div>',
            unsafe_allow_html=True,
        )

        checks = [
            (
                "Completeness",
                analysis["missing_count"],
                "Missing cells",
            ),
            (
                "Uniqueness",
                analysis["duplicate_count"],
                "Duplicate records",
            ),
            (
                "Validity",
                analysis["invalid_count"],
                "Invalid values",
            ),
            (
                "Consistency",
                analysis["city_changes"],
                "City values to standardize",
            ),
            (
                "Outliers",
                analysis["iqr_outlier_count"],
                "Statistical outliers",
            ),
        ]

        cols = st.columns(5)

        for col, check in zip(
            cols,
            checks,
        ):

            with col:

                title, value, caption = check

                if value == 0:

                    badge = (
                        '<span class="dg-badge-good">'
                        'PASSED'
                        '</span>'
                    )

                else:

                    badge = (
                        '<span class="dg-badge-warning">'
                        'WARNING'
                        '</span>'
                    )

                st.markdown(
                    f"""
                    <div class="dg-card"
                         style="min-height:145px;">
                        <div class="dg-card-title">
                            {title}
                        </div>
                        <div style="
                            color:#f8fafc;
                            font-size:27px;
                            font-weight:800;
                            margin:12px 0;
                        ">
                            {value:,}
                        </div>
                        {badge}
                        <div class="dg-card-subtitle">
                            {caption}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="dg-section">Invalid Value Details</div>',
            unsafe_allow_html=True,
        )

        if analysis["invalid_details"].empty:

            st.markdown(
                '<div class="notice notice-success">'
                '✓ No invalid values were detected.'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.dataframe(
                analysis["invalid_details"],
                use_container_width=True,
            )

        st.markdown(
            '<div class="dg-section">City Consistency</div>',
            unsafe_allow_html=True,
        )

        if analysis["city_changes"] > 0:

            st.markdown(
                f"""
                <div class="notice notice-warning">
                    {analysis["city_changes"]:,}
                    city values can be standardized.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="notice notice-success">'
                '✓ No city standardization changes detected.'
                '</div>',
                unsafe_allow_html=True,
            )


    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    elif selected_page == "Anomaly Detection":

        st.markdown(
            '<div class="dg-subtitle">'
            'Identify statistically unusual records without automatically labeling them as errors.'
            '</div>',
            unsafe_allow_html=True,
        )

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Total Records",
            f'{analysis["rows"]:,}',
        )

        a2.metric(
            "Normal",
            f'{analysis["normal_count"]:,}',
        )

        a3.metric(
            "Anomalies",
            f'{analysis["ml_anomaly_count"]:,}',
        )

        a4.metric(
            "Anomaly Rate",
            f'{analysis["ml_anomaly_count"] /
            max(1, analysis["rows"]) * 100:.1f}%',
        )

        st.markdown(
            """
            <div class="notice notice-warning">
                ⚠ IQR outliers and Isolation Forest anomalies are
                screening signals. They are not automatically confirmed
                data errors and should be validated against business context.
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns(
            [1, 2]
        )

        with left:

            st.markdown(
                """
                <div class="dg-card">
                    <div class="dg-card-title">
                        Detection Configuration
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                "Algorithm"
            )

            st.selectbox(
                "Algorithm",
                ["Isolation Forest"],
                disabled=True,
                label_visibility="collapsed",
            )

            sensitivity = st.slider(
                "Sensitivity",
                1,
                20,
                contamination_pct,
                key="anomaly_page_sensitivity",
            )

            if sensitivity != contamination_pct:

                st.session_state.contamination_pct = (
                    sensitivity
                )

                st.session_state.analysis_complete = False
                st.session_state.gemini_analysis = None

                st.rerun()

            st.caption(
                "Current numeric features:"
            )

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            for column in numeric_columns:

                st.caption(
                    f"• {column}"
                )

        with right:

            st.markdown(
                '<div class="dg-card">',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="dg-card-title">'
                'IQR Outlier Summary'
                '</div>',
                unsafe_allow_html=True,
            )

            if analysis["iqr_details"].empty:

                st.success(
                    "No IQR outliers detected."
                )

            else:

                st.dataframe(
                    analysis["iqr_details"],
                    use_container_width=True,
                    height=280,
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="dg-section">Isolation Forest Results</div>',
            unsafe_allow_html=True,
        )

        anomaly_display = df.copy()

        anomaly_display["ML_Anomaly"] = np.where(
            analysis["anomaly_mask"],
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

        st.caption(
            f"Showing up to 500 of "
            f"{len(anomaly_records):,} detected anomalies."
        )

        st.dataframe(
            anomaly_records.head(500),
            use_container_width=True,
            height=480,
        )


    # ========================================================
    # DATA CLEANING
    # ========================================================

    elif selected_page == "Data Cleaning":

        st.markdown(
            '<div class="dg-subtitle">'
            'Preview rule-based cleaning and prepare a Power BI-ready dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        before, after = st.columns(2)

        with before:

            st.markdown(
                """
                <div class="dg-card">
                    <div class="dg-card-title">
                        Before Cleaning
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2 = st.columns(2)

            b1.metric(
                "Rows",
                f'{len(df):,}',
            )

            b2.metric(
                "Missing",
                f'{analysis["missing_count"]:,}',
            )

            b3, b4 = st.columns(2)

            b3.metric(
                "Duplicates",
                f'{analysis["duplicate_count"]:,}',
            )

            b4.metric(
                "Invalid",
                f'{analysis["invalid_count"]:,}',
            )

        with after:

            st.markdown(
                """
                <div class="dg-card">
                    <div class="dg-card-title">
                        After Cleaning
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ac1, ac2 = st.columns(2)

            ac1.metric(
                "Rows",
                f'{len(cleaned_df):,}',
            )

            ac2.metric(
                "Values Filled",
                f'{analysis["values_filled"]:,}',
            )

            ac3, ac4 = st.columns(2)

            ac3.metric(
                "Rows Removed",
                f'{analysis["rows_removed"]:,}',
            )

            ac4.metric(
                "Cities Standardized",
                f'{analysis["city_changes"]:,}',
            )

        st.markdown(
            '<div class="dg-section">Cleaning Actions</div>',
            unsafe_allow_html=True,
        )

        actions = [
            (
                "Duplicate removal",
                analysis["rows_removed"],
            ),
            (
                "Numeric missing-value imputation",
                analysis["values_filled"],
            ),
            (
                "City standardization",
                analysis["city_changes"],
            ),
        ]

        for title, value in actions:

            status = (
                "APPLIED"
                if value > 0
                else "NO CHANGE"
            )

            badge = (
                "dg-badge-good"
                if value > 0
                else "dg-badge-info"
            )

            st.markdown(
                f"""
                <div class="dg-card"
                     style="
                     display:flex;
                     align-items:center;
                     justify-content:space-between;
                     ">
                    <div>
                        <div class="dg-card-title">
                            {title}
                        </div>
                        <div class="dg-card-subtitle">
                            {value:,} records/cells affected
                        </div>
                    </div>
                    <span class="{badge}">
                        {status}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="dg-section">Cleaned Dataset Preview</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "⬇ Download Cleaned Dataset",
            data=cleaned_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="DataGuard_Cleaned_Data.csv",
            mime="text/csv",
            type="primary",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif selected_page == "Analytics":

        st.markdown(
            '<div class="dg-subtitle">'
            'Business-focused analytics generated from the cleaned dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        if business_df is None:

            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <div class="empty-title">
                        Business analytics unavailable
                    </div>
                    <div class="empty-text">
                        No Sales, Revenue or Amount column was detected.
                        Data quality analysis is still available.
                    </div>
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

            total_sales = business_df[
                sales_column
            ].sum()

            avg_sales = business_df[
                sales_column
            ].mean()

            st.markdown(
                f"""
                <div class="notice">
                    Business charts use the cleaned dataset.
                    Values above the IQR upper bound
                    <b>{upper_bound:,.2f}</b> are excluded
                    only from visualization.
                    Records are not deleted.
                </div>
                """,
                unsafe_allow_html=True,
            )

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Total Sales",
                f"{total_sales:,.2f}",
            )

            k2.metric(
                "Average Value",
                f"{avg_sales:,.2f}",
            )

            k3.metric(
                "Sales Records",
                f"{len(business_df):,}",
            )

            st.markdown(
                '<div class="dg-section">Sales Distribution</div>',
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
                ),
                use_container_width=True,
            )

            left, right = st.columns(2)

            with left:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_counts = (
                        cleaned_df[
                            city_columns[0]
                        ]
                        .value_counts()
                        .head(10)
                    )

                    st.markdown(
                        '<div class="dg-section">'
                        'Records by City'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        city_counts,
                        use_container_width=True,
                    )

            with right:

                product_columns = [
                    column
                    for column in cleaned_df.columns
                    if "product"
                    in str(column).lower()
                ]

                if product_columns:

                    product_sales = (
                        business_df
                        .groupby(
                            product_columns[0]
                        )[sales_column]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                    )

                    st.markdown(
                        '<div class="dg-section">'
                        'Sales by Product'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.bar_chart(
                        product_sales,
                        use_container_width=True,
                    )

            category_columns = [
                column
                for column in cleaned_df.columns
                if "category"
                in str(column).lower()
            ]

            if category_columns:

                category_sales = (
                    business_df
                    .groupby(
                        category_columns[0]
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    '<div class="dg-section">'
                    'Sales by Category'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.bar_chart(
                    category_sales,
                    use_container_width=True,
                )

            datetime_columns = analysis[
                "datetime_columns"
            ]

            if datetime_columns:

                date_column = datetime_columns[0]

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
                    monthly_df
                    .dropna(
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

                st.markdown(
                    '<div class="dg-section">'
                    'Monthly Sales Trend'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.line_chart(
                    monthly_sales,
                    use_container_width=True,
                )


    # ========================================================
    # POWER BI
    # ========================================================

    elif selected_page == "Power BI":

        st.markdown(
            '<div class="dg-subtitle">'
            'Export cleaned datasets and supporting tables for Power BI.'
            '</div>',
            unsafe_allow_html=True,
        )

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis["sales_column"],
        )

        st.markdown(
            """
            <div class="notice">
                <b>Power BI workflow</b><br>
                Download the export package → extract the CSV files →
                open Power BI Desktop → Get Data → Text/CSV.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="dg-section">Available Exports</div>',
            unsafe_allow_html=True,
        )

        for filename, data in exports.items():

            left, right = st.columns(
                [7, 1]
            )

            with left:

                st.markdown(
                    f"""
                    <div class="dg-card">
                        <div class="dg-card-title">
                            📄 {filename}
                        </div>
                        <div class="dg-card-subtitle">
                            Power BI-ready CSV export
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:

                st.write("")

                st.download_button(
                    "Download",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}",
                )

        st.markdown(
            '<div class="dg-section">Complete Package</div>',
            unsafe_allow_html=True,
        )

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "📦 Download Complete Power BI Package",
            data=zip_data,
            file_name="DataGuard_PowerBI_Exports.zip",
            mime="application/zip",
            type="primary",
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif selected_page == "AI Analysis":

        st.markdown(
            '<div class="dg-subtitle">'
            'Generate a portfolio-ready interpretation using Gemini.'
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

IQR outliers and ML anomalies are screening signals,
not automatically confirmed errors.

Business visualization filtering does not delete records.
"""

        ai_left, ai_right = st.columns(
            [2, 1]
        )

        with ai_left:

            st.markdown(
                """
                <div class="dg-card">
                    <div class="dg-card-title">
                        ✨ Gemini AI Analyst
                    </div>
                    <div class="dg-card-subtitle">
                        Interpret the DataGuard quality report
                        without inventing unsupported facts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if not GEMINI_API_KEY:

                st.markdown(
                    """
                    <div class="notice notice-warning">
                        Gemini API key is not configured.
                        Add GEMINI_API_KEY to Streamlit secrets
                        to enable AI analysis.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.button(
                "✨ Generate AI Analysis",
                type="primary",
            ):

                with st.spinner(
                    "Gemini is analyzing the quality report..."
                ):

                    result = (
                        get_gemini_analysis(
                            summary_text
                        )
                    )

                st.session_state.gemini_analysis = (
                    result
                )

        with ai_right:

            st.markdown(
                """
                <div class="dg-card">
                    <div class="dg-card-title">
                        Analysis Scope
                    </div>
                    <br>
                    <div style="
                        color:#8994a7;
                        font-size:12px;
                        line-height:1.8;
                    ">
                        ✓ Data quality<br>
                        ✓ Statistical outliers<br>
                        ✓ ML anomalies<br>
                        ✓ Cleaning results<br>
                        ✓ Business impact<br>
                        ✓ Power BI recommendations
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.gemini_analysis:

            st.markdown(
                '<div class="dg-section">'
                'AI Assessment'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                st.session_state.gemini_analysis
            )

        else:

            st.info(
                "Generate the AI analysis to see the interpretation here."
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif selected_page == "Reports":

        st.markdown(
            '<div class="dg-subtitle">'
            'Portfolio-ready summary of the complete DataGuard assessment.'
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

Extreme sales values excluded from business charts are excluded
only for visualization and are not deleted from the analytical dataset.
"""

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Quality",
            f'{analysis["quality_score"]:.1f}%',
        )

        r2.metric(
            "Records",
            f'{analysis["rows"]:,}',
        )

        r3.metric(
            "Issues",
            f'{analysis["missing_count"] + analysis["duplicate_count"] + analysis["invalid_count"]:,}',
        )

        r4.metric(
            "Anomalies",
            f'{analysis["ml_anomaly_count"]:,}',
        )

        st.markdown(
            '<div class="dg-section">Executive Summary</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="dg-card">
                <div style="
                    color:#cbd5e1;
                    font-size:14px;
                    line-height:1.8;
                ">
                    <b>{st.session_state.file_name}</b>
                    contains
                    <b>{analysis["rows"]:,}</b> records across
                    <b>{analysis["columns"]}</b> columns.
                    The current DataGuard quality score is
                    <b>{analysis["quality_score"]:.1f}/100</b>.
                    The pipeline identified
                    <b>{analysis["missing_count"]:,}</b> missing cells,
                    <b>{analysis["duplicate_count"]:,}</b> duplicate records,
                    and
                    <b>{analysis["invalid_count"]:,}</b> invalid values.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="dg-section">Full Report</div>',
            unsafe_allow_html=True,
        )

        st.code(
            report_text,
            language="text",
        )

        st.download_button(
            "📄 Download Data Quality Report",
            data=report_text.encode(
                "utf-8"
            ),
            file_name="DataGuard_Data_Quality_Report.txt",
            mime="text/plain",
            type="primary",
        )

        if st.session_state.gemini_analysis:

            st.markdown(
                '<div class="dg-section">'
                'Gemini AI Assessment'
                '</div>',
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
        🛡️ DataGuard AI
        &nbsp;•&nbsp;
        AI Data Quality & Anomaly Detection Platform
        <br><br>
        Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
