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

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

/* ---------- SIDEBAR ---------- */

section[data-testid="stSidebar"] {
    background: #0d111a;
    border-right: 1px solid #1d2635;
}

section[data-testid="stSidebar"] > div {
    background: #0d111a;
}

section[data-testid="stSidebar"] * {
    color: #d8dee9;
}

.sidebar-brand {
    padding: 8px 4px 18px 4px;
}

.sidebar-logo {
    font-size: 30px;
    margin-bottom: 4px;
}

.sidebar-title {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -0.4px;
}

.sidebar-subtitle {
    color: #7f8ba3;
    font-size: 11px;
    margin-top: 3px;
}

.sidebar-section {
    color: #697586;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin: 20px 0 8px 2px;
}

/* ---------- RADIO NAVIGATION ---------- */

div[data-testid="stRadio"] > label {
    display: none;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 3px;
}

div[data-testid="stRadio"] label {
    border-radius: 9px;
    padding: 8px 10px;
    color: #8994a8;
    transition: 0.15s ease;
}

div[data-testid="stRadio"] label:hover {
    background: #151c29;
    color: #f8fafc;
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: #171d2b;
    color: #ffffff;
    border: 1px solid #283247;
}

/* ---------- TYPOGRAPHY ---------- */

h1, h2, h3, h4 {
    color: #f8fafc !important;
}

h1 {
    font-size: 30px !important;
    letter-spacing: -0.8px;
}

h2 {
    font-size: 22px !important;
    letter-spacing: -0.4px;
}

h3 {
    font-size: 17px !important;
}

p, span, label {
    color: #a7b0c0;
}

.page-eyebrow {
    color: #697586;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.page-subtitle {
    color: #7f8ba3;
    font-size: 13px;
    margin-bottom: 20px;
}

/* ---------- TOP BAR ---------- */

.topbar {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 14px;
    padding: 13px 17px;
    margin-bottom: 20px;
}

.topbar-title {
    color: #f8fafc;
    font-size: 13px;
    font-weight: 700;
}

.topbar-dataset {
    color: #78859b;
    font-size: 11px;
    margin-top: 2px;
}

/* ---------- STATUS ---------- */

.status-live {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0e1c18;
    border: 1px solid #173d31;
    color: #86efac;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 10px;
    font-weight: 800;
}

.status-dot {
    width: 6px;
    height: 6px;
    background: #4ade80;
    border-radius: 50%;
}

/* ---------- WORKFLOW ---------- */

.workflow {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 20px;
    overflow-x: auto;
}

.workflow-step {
    min-width: 92px;
    padding: 8px 10px;
    border-radius: 9px;
    background: #111722;
    border: 1px solid #202a3a;
}

.workflow-step.active {
    background: #17152a;
    border-color: #6d5dfc;
}

.workflow-number {
    color: #697586;
    font-size: 9px;
    font-weight: 800;
}

.workflow-name {
    color: #dce3ee;
    font-size: 11px;
    font-weight: 700;
    margin-top: 2px;
}

.workflow-arrow {
    color: #4b5565;
    font-size: 14px;
}

/* ---------- CARDS ---------- */

.dg-card {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.dg-card-title {
    color: #eef2f7;
    font-size: 14px;
    font-weight: 750;
}

.dg-card-subtitle {
    color: #697586;
    font-size: 11px;
    margin-top: 3px;
}

/* ---------- KPI ---------- */

.kpi-card {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 14px;
    padding: 17px;
    min-height: 118px;
    margin-bottom: 12px;
}

.kpi-label {
    color: #697586;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.kpi-value {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 850;
    margin-top: 9px;
    letter-spacing: -0.6px;
}

.kpi-caption {
    color: #68758a;
    font-size: 10px;
    margin-top: 3px;
}

/* ---------- SCORE ---------- */

.score-card {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 16px;
    padding: 20px;
    min-height: 290px;
}

.score-number {
    color: #f8fafc;
    font-size: 45px;
    font-weight: 900;
    line-height: 1;
}

.score-denom {
    color: #697586;
    font-size: 14px;
}

.score-status {
    color: #86efac;
    font-size: 11px;
    font-weight: 800;
    margin-top: 7px;
}

.progress-bg {
    width: 100%;
    height: 7px;
    background: #1b2432;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 17px;
}

.progress-fill {
    height: 100%;
    background: #6d5dfc;
    border-radius: 20px;
}

/* ---------- QUALITY ITEMS ---------- */

.quality-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 0;
    border-bottom: 1px solid #192231;
}

.quality-item:last-child {
    border-bottom: none;
}

.quality-name {
    color: #cbd5e1;
    font-size: 12px;
}

.badge {
    border-radius: 20px;
    padding: 4px 9px;
    font-size: 10px;
    font-weight: 800;
}

.badge-warning {
    color: #fcd34d;
    background: #29200b;
    border: 1px solid #51400d;
}

.badge-danger {
    color: #fca5a5;
    background: #2a1216;
    border: 1px solid #542027;
}

.badge-info {
    color: #c4b5fd;
    background: #1d1835;
    border: 1px solid #352c63;
}

.badge-good {
    color: #86efac;
    background: #0d2119;
    border: 1px solid #1b4935;
}

/* ---------- ANOMALY ---------- */

.anomaly-card {
    background: #0d121c;
    border: 1px solid #27213d;
    border-radius: 15px;
    padding: 19px;
}

.anomaly-number {
    font-size: 25px;
    font-weight: 850;
}

.anomaly-label {
    color: #697586;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1px;
}

.anomaly-note {
    color: #697586;
    font-size: 10px;
    line-height: 1.5;
    margin-top: 16px;
}

/* ---------- ACTIVITY ---------- */

.activity-item {
    padding: 12px 0 12px 15px;
    border-left: 2px solid #252f40;
    margin-left: 4px;
}

.activity-title {
    color: #dbe3ee;
    font-size: 12px;
    font-weight: 700;
}

.activity-text {
    color: #697586;
    font-size: 10px;
    margin-top: 3px;
}

/* ---------- INFO ---------- */

.dg-info {
    background: #0d1828;
    border: 1px solid #1c3655;
    color: #93c5fd;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 11px;
    margin-bottom: 13px;
}

.dg-warning {
    background: #20190b;
    border: 1px solid #4c3b0d;
    color: #fcd34d;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 11px;
    margin-bottom: 13px;
}

.dg-success {
    background: #0c1d17;
    border: 1px solid #194633;
    color: #86efac;
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 11px;
    margin-bottom: 13px;
}

/* ---------- UPLOAD ---------- */

.upload-card {
    background: #0d121c;
    border: 1px dashed #354157;
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-bottom: 18px;
}

.upload-icon {
    font-size: 30px;
    margin-bottom: 8px;
}

.upload-title {
    color: #f8fafc;
    font-size: 16px;
    font-weight: 800;
}

.upload-text {
    color: #697586;
    font-size: 11px;
    margin-top: 5px;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #1d2635;
    border-radius: 10px;
    overflow: hidden;
}

/* ---------- BUTTONS ---------- */

.stButton button,
.stDownloadButton button {
    border-radius: 9px !important;
    border: 1px solid #293449 !important;
    background: #151c29 !important;
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

.stButton button:hover,
.stDownloadButton button:hover {
    border-color: #6d5dfc !important;
    color: white !important;
    background: #1a1d32 !important;
}

button[kind="primary"] {
    background: #6254e8 !important;
    border-color: #6254e8 !important;
}

/* ---------- SLIDER ---------- */

.stSlider > div > div > div {
    color: #8b7cff;
}

/* ---------- METRIC ---------- */

[data-testid="stMetric"] {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 12px;
    padding: 13px;
}

[data-testid="stMetricLabel"] {
    color: #697586 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* ---------- EXPANDER ---------- */

[data-testid="stExpander"] {
    background: #0d121c;
    border: 1px solid #1d2635;
    border-radius: 10px;
}

/* ---------- FOOTER ---------- */

.dg-footer {
    text-align: center;
    color: #4e5a6e;
    font-size: 10px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #192231;
}

/* ---------- MOBILE ---------- */

@media (max-width: 768px) {

    .block-container {
        padding-left: 12px;
        padding-right: 12px;
    }

    h1 {
        font-size: 25px !important;
    }

    .workflow {
        overflow-x: auto;
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
                int(df[column].nunique(dropna=True))
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

        if not any(token in name for token in numeric_tokens):
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
            negative_values = numeric_values < 0

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

        standardized = cleaned.map(CITY_MAPPING)
        standardized = standardized.fillna(
            cleaned.str.title()
        )

        changes = (
            original.astype("string").fillna("")
            != standardized.fillna("")
        )

        total_changes += int(changes.sum())

        result[column] = standardized

    return result, total_changes


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


def detect_ml_anomalies(df, contamination_pct):

    numeric_df = (
        df.select_dtypes(include=np.number)
        .copy()
    )

    if numeric_df.shape[1] == 0:

        return (
            pd.Series(False, index=df.index),
            0,
        )

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

    predictions = model.fit_predict(numeric_df)

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
        duplicate_count / max(1, total_cells)
    ) * 100

    invalid_penalty = (
        invalid_count / max(1, total_cells)
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

    cleaned, city_changes = standardize_city_column(
        cleaned
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
                )
                .astype("float64")
            )

        except Exception:
            continue

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            median_value = cleaned[column].median()

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

                values_filled += missing_before

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
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            mode_values = cleaned[column].mode(
                dropna=True
            )

            if len(mode_values) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode_values.iloc[0])
                )

                values_filled += missing_before

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

    q1 = business_df[sales_column].quantile(0.25)
    q3 = business_df[sales_column].quantile(0.75)
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


def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

    profile = profile_dataset(cleaned_df)

    exports["DataGuard_Cleaned_Data.csv"] = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    exports["DataGuard_Data_Profile.csv"] = (
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

        exports["DataGuard_Sales_Summary.csv"] = (
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

        city_column = city_columns[0]

        city_summary = (
            cleaned_df
            .groupby(city_column)
            .size()
            .reset_index(
                name="Record_Count"
            )
        )

        exports["DataGuard_City_Summary.csv"] = (
            city_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    datetime_columns = detect_datetime_columns(
        cleaned_df
    )

    if (
        datetime_columns
        and sales_column is not None
    ):

        date_column = datetime_columns[0]

        monthly_df = cleaned_df.copy()

        monthly_df[date_column] = safe_to_datetime(
            monthly_df[date_column]
        )

        monthly_df[sales_column] = pd.to_numeric(
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
                Month=monthly_df[date_column]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month")[sales_column]
            .sum()
            .reset_index()
        )

        exports["DataGuard_Monthly_Sales.csv"] = (
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
# UI HELPERS
# ============================================================

def kpi_card(label, value, caption):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_header(title, subtitle=""):

    if subtitle:

        st.markdown(
            f"""
            <div class="dg-card-title">{title}</div>
            <div class="dg-card-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="dg-card-title">{title}</div>
            """,
            unsafe_allow_html=True,
        )


def quality_badge(value, kind="info"):

    return (
        f'<span class="badge badge-{kind}">{value}</span>'
    )


def quality_status(score):

    if score >= 95:
        return "Excellent"

    if score >= 85:
        return "Good"

    if score >= 70:
        return "Needs Attention"

    return "Critical"


def render_workflow(current_page):

    pages = [
        ("01", "Upload", "Dashboard"),
        ("02", "Profile", "Data Quality"),
        ("03", "Quality", "Data Quality"),
        ("04", "Anomalies", "Anomalies"),
        ("05", "Clean", "Cleaning"),
        ("06", "Analytics", "Analytics"),
        ("07", "Power BI", "Power BI"),
        ("08", "AI", "AI Analysis"),
    ]

    parts = []

    for i, (number, name, page_name) in enumerate(pages):

        active = current_page == page_name

        cls = "workflow-step active" if active else "workflow-step"

        parts.append(
            f"""
            <div class="{cls}">
                <div class="workflow-number">{number}</div>
                <div class="workflow-name">{name}</div>
            </div>
            """
        )

        if i < len(pages) - 1:
            parts.append(
                '<div class="workflow-arrow">›</div>'
            )

    st.markdown(
        '<div class="workflow">'
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_score(score):

    status = quality_status(score)

    st.markdown(
        f"""
        <div class="score-card">

            <div class="dg-card-title">
                Overall Quality Score
            </div>

            <div class="dg-card-subtitle">
                Composite health indicator
            </div>

            <div style="margin-top:30px;">
                <span class="score-number">
                    {score:.1f}
                </span>
                <span class="score-denom">
                    / 100
                </span>

                <div class="score-status">
                    ● {status}
                </div>
            </div>

            <div class="progress-bg">
                <div
                    class="progress-fill"
                    style="width:{min(score,100)}%;"
                ></div>
            </div>

            <div style="
                display:grid;
                grid-template-columns:1fr 1fr;
                gap:14px;
                margin-top:25px;
            ">

                <div>
                    <div class="kpi-label">
                        COMPLETENESS
                    </div>
                    <div style="
                        color:#e5e7eb;
                        font-size:16px;
                        font-weight:800;
                        margin-top:4px;
                    ">
                        {max(0, 100 - (
                            analysis_global["missing_count"]
                            / max(1, analysis_global["total_cells"])
                            * 100
                        )):.1f}%
                    </div>
                </div>

                <div>
                    <div class="kpi-label">
                        UNIQUENESS
                    </div>
                    <div style="
                        color:#e5e7eb;
                        font-size:16px;
                        font-weight:800;
                        margin-top:4px;
                    ">
                        {max(0, 100 - (
                            analysis_global["duplicate_count"]
                            / max(1, analysis_global["rows"])
                            * 100
                        )):.1f}%
                    </div>
                </div>

                <div>
                    <div class="kpi-label">
                        VALIDITY
                    </div>
                    <div style="
                        color:#e5e7eb;
                        font-size:16px;
                        font-weight:800;
                        margin-top:4px;
                    ">
                        {max(0, 100 - (
                            analysis_global["invalid_count"]
                            / max(1, analysis_global["total_cells"])
                            * 100
                        )):.1f}%
                    </div>
                </div>

                <div>
                    <div class="kpi-label">
                        RECORDS
                    </div>
                    <div style="
                        color:#e5e7eb;
                        font-size:16px;
                        font-weight:800;
                        margin-top:4px;
                    ">
                        {analysis_global["rows"]:,}
                    </div>
                </div>

            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🛡️</div>
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
        '<div class="sidebar-section">WORKSPACE</div>',
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
        '<div class="sidebar-section">SYSTEM</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.file_name:

        st.caption(
            "Dataset: "
            + st.session_state.file_name
        )

        st.markdown(
            """
            <div style="
                color:#86efac;
                font-size:10px;
                font-weight:700;
                margin-top:8px;
            ">
                ● All systems operational
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "No dataset loaded"
        )

    st.markdown(
        '<div class="sidebar-section">ANOMALY SETTINGS</div>',
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
        "Higher sensitivity flags more records "
        "as unusual."
    )

    st.markdown(
        '<div class="sidebar-section">PIPELINE</div>',
        unsafe_allow_html=True,
    )

    pipeline = [
        "01  Upload",
        "02  Profile",
        "03  Quality",
        "04  Anomalies",
        "05  Clean",
        "06  Analytics",
        "07  Power BI",
        "08  AI",
    ]

    for item in pipeline:
        st.caption(item)

    st.markdown(
        """
        <div style="
            position:fixed;
            bottom:18px;
            font-size:10px;
            color:#4e5a6e;
        ">
            DataGuard AI • Portfolio Edition
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD / LOAD DATA
# ============================================================

if st.session_state.df is None:

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / GET STARTED</div>',
        unsafe_allow_html=True,
    )

    st.title("Data Quality Command Center")

    st.markdown(
        """
        <div class="page-subtitle">
            Upload a dataset to profile, validate, detect anomalies,
            clean, analyze and prepare it for Power BI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-icon">📂</div>
            <div class="upload-title">
                Upload your dataset
            </div>
            <div class="upload-text">
                CSV, XLSX or XLS • Complete dataset analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Browse files",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.analysis = None
            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

            st.success(
                f"{uploaded_file.name} uploaded successfully."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )
            st.stop()

    st.markdown("### What DataGuard AI does")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            "### 🔍"
        )

        st.markdown(
            "**Profile**"
        )

        st.caption(
            "Understand rows, columns, types, missing values and identifiers."
        )

    with c2:

        st.markdown(
            "### 🛡️"
        )

        st.markdown(
            "**Validate**"
        )

        st.caption(
            "Detect missing, duplicate and invalid data."
        )

    with c3:

        st.markdown(
            "### 🚨"
        )

        st.markdown(
            "**Detect anomalies**"
        )

        st.caption(
            "Use IQR and Isolation Forest screening."
        )

    with c4:

        st.markdown(
            "### 📊"
        )

        st.markdown(
            "**Analyze**"
        )

        st.caption(
            "Generate analytics and Power BI-ready exports."
        )

    st.stop()


# ============================================================
# ANALYSIS ENGINE
# ============================================================

if not st.session_state.analysis_complete:

    df = st.session_state.df

    with st.spinner(
        "Analyzing the complete dataset..."
    ):

        rows = len(df)
        columns = len(df.columns)
        total_cells = rows * columns

        missing_count = int(
            df.isna().sum().sum()
        )

        duplicate_count = int(
            df.duplicated().sum()
        )

        invalid_count, invalid_details = (
            detect_invalid_values(df)
        )

        quality_score = build_quality_score(
            total_cells,
            missing_count,
            duplicate_count,
            invalid_count,
        )

        datetime_columns = detect_datetime_columns(
            df
        )

        identifier_columns = detect_identifier_columns(
            df,
            datetime_columns,
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
            rows - ml_anomaly_count
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
# GLOBAL ANALYSIS REFERENCES
# ============================================================

df = st.session_state.df
cleaned_df = st.session_state.cleaned_df
business_df = st.session_state.business_df
analysis = st.session_state.analysis

analysis_global = analysis

contamination_pct = (
    st.session_state.contamination_pct
)


# ============================================================
# TOP BAR
# ============================================================

st.markdown(
    f"""
    <div class="topbar">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:20px;
        ">

            <div>
                <div class="topbar-title">
                    WORKSPACE / {page.upper()}
                </div>

                <div class="topbar-dataset">
                    {st.session_state.file_name}
                    &nbsp; • &nbsp;
                    {analysis["rows"]:,} records
                    &nbsp; • &nbsp;
                    {analysis["columns"]} columns
                </div>
            </div>

            <div class="status-live">
                <span class="status-dot"></span>
                LIVE
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# WORKFLOW
# ============================================================

render_workflow(page)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / DASHBOARD</div>',
        unsafe_allow_html=True,
    )

    st.title("Data Quality Overview")

    st.markdown(
        """
        <div class="page-subtitle">
            Monitor, analyze and improve the quality of your dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset identity

    st.markdown(
        f"""
        <div class="dg-info">
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

    # KPI row

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        kpi_card(
            "Data Quality",
            f'{analysis["quality_score"]:.1f}',
            "/ 100 score",
        )

    with k2:
        kpi_card(
            "Records",
            f'{analysis["rows"]:,}',
            "total rows",
        )

    with k3:
        kpi_card(
            "Columns",
            analysis["columns"],
            "fields",
        )

    with k4:
        kpi_card(
            "Missing",
            f'{analysis["missing_count"]:,}',
            "cells",
        )

    with k5:
        kpi_card(
            "Duplicates",
            f'{analysis["duplicate_count"]:,}',
            "records",
        )

    with k6:
        kpi_card(
            "Anomalies",
            f'{analysis["ml_anomaly_count"]:,}',
            f"{contamination_pct}% sensitivity",
        )

    # Quality overview

    st.markdown("### Quality Overview")

    left, right = st.columns(
        [1, 1.15],
        gap="large",
    )

    with left:

        render_score(
            analysis["quality_score"]
        )

    with right:

        st.markdown(
            '<div class="dg-card">',
            unsafe_allow_html=True,
        )

        card_header(
            "Issues Detected",
            "Automated quality checks across the dataset.",
        )

        st.markdown(
            f"""
            <div class="quality-item">
                <span class="quality-name">
                    Missing Values
                </span>
                {quality_badge(
                    f'{analysis["missing_count"]:,}',
                    "warning"
                )}
            </div>

            <div class="quality-item">
                <span class="quality-name">
                    Duplicate Records
                </span>
                {quality_badge(
                    f'{analysis["duplicate_count"]:,}',
                    "warning"
                )}
            </div>

            <div class="quality-item">
                <span class="quality-name">
                    Invalid Values
                </span>
                {quality_badge(
                    f'{analysis["invalid_count"]:,}',
                    "danger"
                )}
            </div>

            <div class="quality-item">
                <span class="quality-name">
                    Potential Outliers
                </span>
                {quality_badge(
                    f'{analysis["iqr_outlier_count"]:,}',
                    "info"
                )}
            </div>

            <div class="quality-item">
                <span class="quality-name">
                    ML Anomalies
                </span>
                {quality_badge(
                    f'{analysis["ml_anomaly_count"]:,}',
                    "info"
                )}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # Anomaly section

    st.markdown("### Anomaly Detection")

    left, right = st.columns(
        [1.25, 0.75],
        gap="large",
    )

    with left:

        st.markdown(
            '<div class="anomaly-card">',
            unsafe_allow_html=True,
        )

        card_header(
            "Isolation Forest Screening",
            "ML-based screening for unusual records.",
        )

        n1, n2, n3 = st.columns(3)

        with n1:

            st.markdown(
                f"""
                <div style="margin-top:22px;">
                    <div class="anomaly-label">
                        NORMAL
                    </div>
                    <div class="anomaly-number"
                         style="color:#86efac;">
                        {analysis["normal_count"]:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with n2:

            st.markdown(
                f"""
                <div style="margin-top:22px;">
                    <div class="anomaly-label">
                        ANOMALIES
                    </div>
                    <div class="anomaly-number"
                         style="color:#fca5a5;">
                        {analysis["ml_anomaly_count"]:,}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with n3:

            rate = (
                analysis["ml_anomaly_count"]
                / max(1, analysis["rows"])
                * 100
            )

            st.markdown(
                f"""
                <div style="margin-top:22px;">
                    <div class="anomaly-label">
                        RATE
                    </div>
                    <div class="anomaly-number"
                         style="color:#c4b5fd;">
                        {rate:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="anomaly-note">
                Isolation Forest • {contamination_pct}% sensitivity
                <br>
                ML anomalies are screening signals,
                not confirmed errors.
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
            '<div class="dg-card">',
            unsafe_allow_html=True,
        )

        card_header(
            "Sensitivity",
            "Detection sensitivity",
        )

        st.slider(
            "Sensitivity",
            min_value=1,
            max_value=20,
            value=contamination_pct,
            disabled=True,
            label_visibility="collapsed",
        )

        st.caption(
            "Increase sensitivity to flag more unusual records."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # Quality monitoring

    st.markdown("### Quality Monitoring")

    quality_df = pd.DataFrame(
        {
            "Metric": [
                "Duplicates",
                "IQR Outliers",
                "Invalid",
                "ML Anomalies",
                "Missing",
            ],
            "Count": [
                analysis["duplicate_count"],
                analysis["iqr_outlier_count"],
                analysis["invalid_count"],
                analysis["ml_anomaly_count"],
                analysis["missing_count"],
            ],
        }
    )

    st.bar_chart(
        quality_df.set_index("Metric"),
        height=300,
    )

    # Recent activity

    st.markdown("### Recent Activity")

    st.markdown(
        f"""
        <div class="dg-card">

            <div class="activity-item">
                <div class="activity-title">
                    Dataset analyzed
                </div>
                <div class="activity-text">
                    {analysis["rows"]:,} records and
                    {analysis["columns"]} columns processed
                </div>
            </div>

            <div class="activity-item">
                <div class="activity-title">
                    Quality checks completed
                </div>
                <div class="activity-text">
                    {analysis["missing_count"]:,} missing cells,
                    {analysis["duplicate_count"]:,} duplicates detected
                </div>
            </div>

            <div class="activity-item">
                <div class="activity-title">
                    Anomaly screening completed
                </div>
                <div class="activity-text">
                    {analysis["ml_anomaly_count"]:,}
                    ML anomalies flagged
                </div>
            </div>

            <div class="activity-item">
                <div class="activity-title">
                    Cleaning preview generated
                </div>
                <div class="activity-text">
                    {analysis["rows_removed"]:,} duplicate rows removable,
                    {analysis["values_filled"]:,} values fillable
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset preview

    st.markdown("### Dataset Preview")

    st.caption(
        f"Showing 10 of {len(df):,} records"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=350,
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / PROFILE</div>',
        unsafe_allow_html=True,
    )

    st.title("Data Profile & Quality")

    st.markdown(
        """
        <div class="page-subtitle">
            Understand structure, completeness, validity and uniqueness.
        </div>
        """,
        unsafe_allow_html=True,
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Quality Score",
        f'{analysis["quality_score"]:.1f}/100',
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

    st.markdown("### Dataset Profile")

    profile = profile_dataset(df)

    st.dataframe(
        profile,
        use_container_width=True,
        height=450,
    )

    st.markdown("### Data Types")

    a, b, c, d = st.columns(4)

    a.metric(
        "Numeric",
        analysis["numeric_count"],
    )

    b.metric(
        "Categorical",
        analysis["categorical_count"],
    )

    c.metric(
        "Date / Time",
        len(analysis["datetime_columns"]),
    )

    d.metric(
        "Identifiers",
        len(analysis["identifier_columns"]),
    )

    with st.expander(
        "View detected Date / Time columns"
    ):

        if analysis["datetime_columns"]:
            st.write(
                analysis["datetime_columns"]
            )
        else:
            st.info(
                "No Date/Time columns detected."
            )

    with st.expander(
        "View detected Identifier columns"
    ):

        if analysis["identifier_columns"]:
            st.write(
                analysis["identifier_columns"]
            )
        else:
            st.info(
                "No identifier columns detected."
            )

    st.markdown("### Invalid Values")

    if analysis["invalid_details"].empty:

        st.markdown(
            """
            <div class="dg-success">
                ✓ No invalid values were detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.dataframe(
            analysis["invalid_details"],
            use_container_width=True,
        )

    st.markdown("### City Standardization")

    if analysis["city_changes"] > 0:

        st.markdown(
            f"""
            <div class="dg-info">
                {analysis["city_changes"]:,} city values
                can be standardized during cleaning.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="dg-success">
                ✓ No city standardization changes detected.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ANOMALIES
# ============================================================

elif page == "Anomalies":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / ANOMALIES</div>',
        unsafe_allow_html=True,
    )

    st.title("Anomaly Detection")

    st.markdown(
        """
        <div class="page-subtitle">
            Identify statistically unusual records using IQR and Isolation Forest.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3, a4 = st.columns(4)

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

    rate = (
        analysis["ml_anomaly_count"]
        / max(1, analysis["rows"])
        * 100
    )

    a4.metric(
        "Anomaly Rate",
        f"{rate:.1f}%",
    )

    st.markdown(
        """
        <div class="dg-warning">
            ⚠ IQR outliers and ML anomalies are not automatically
            data errors. They identify statistically unusual
            observations that may require business validation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### IQR Statistical Outliers")

    if analysis["iqr_details"].empty:

        st.markdown(
            """
            <div class="dg-success">
                ✓ No IQR outliers detected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.dataframe(
            analysis["iqr_details"],
            use_container_width=True,
        )

    st.markdown("### Isolation Forest")

    st.markdown(
        f"""
        <div class="dg-info">
            Current sensitivity: <b>{contamination_pct}%</b>
            <br>
            ML anomalies are screening signals, not confirmed errors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    anomaly_display = df.copy()

    anomaly_display["ML_Anomaly"] = np.where(
        analysis["anomaly_mask"],
        "Anomaly",
        "Normal",
    )

    anomaly_records = anomaly_display[
        anomaly_display["ML_Anomaly"] == "Anomaly"
    ]

    with st.expander(
        f"View {len(anomaly_records):,} ML anomaly records"
    ):

        st.caption(
            "Showing up to 500 detected anomalies."
        )

        st.dataframe(
            anomaly_records.head(500),
            use_container_width=True,
            height=450,
        )


# ============================================================
# CLEANING
# ============================================================

elif page == "Cleaning":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / CLEAN</div>',
        unsafe_allow_html=True,
    )

    st.title("Automated Data Cleaning")

    st.markdown(
        """
        <div class="page-subtitle">
            Transform the dataset into a cleaner, Power BI-ready version.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

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
        f'{analysis["rows_removed"]:,}',
    )

    c4.metric(
        "Values Filled",
        f'{analysis["values_filled"]:,}',
    )

    st.markdown(
        """
        <div class="dg-success">
            ✓ Duplicate removal • city standardization •
            numeric median imputation • categorical mode imputation
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analysis["city_changes"] > 0:

        st.markdown(
            f"""
            <div class="dg-info">
                {analysis["city_changes"]:,}
                city values were standardized.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Before → After")

    before, after = st.columns(2)

    with before:

        st.markdown(
            '<div class="dg-card">',
            unsafe_allow_html=True,
        )

        card_header(
            "Original Dataset",
            "Before cleaning",
        )

        st.metric(
            "Rows",
            f'{len(df):,}',
        )

        st.metric(
            "Missing",
            f'{analysis["missing_count"]:,}',
        )

        st.metric(
            "Duplicates",
            f'{analysis["duplicate_count"]:,}',
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with after:

        cleaned_missing = int(
            cleaned_df.isna().sum().sum()
        )

        cleaned_duplicates = int(
            cleaned_df.duplicated().sum()
        )

        st.markdown(
            '<div class="dg-card">',
            unsafe_allow_html=True,
        )

        card_header(
            "Cleaned Dataset",
            "After automated cleaning",
        )

        st.metric(
            "Rows",
            f'{len(cleaned_df):,}',
        )

        st.metric(
            "Missing",
            f'{cleaned_missing:,}',
        )

        st.metric(
            "Duplicates",
            f'{cleaned_duplicates:,}',
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Cleaned Dataset Preview")

    st.caption(
        f"Showing 10 of {len(cleaned_df):,} cleaned records"
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


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.title("Business Analytics")

    st.markdown(
        """
        <div class="page-subtitle">
            Explore business patterns generated from the cleaned dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if business_df is None:

        st.markdown(
            """
            <div class="dg-warning">
                No Sales, Revenue or Amount column was detected.
                Business analytics are unavailable for this dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        sales_column = analysis["sales_column"]
        upper_bound = analysis["upper_bound"]

        total_sales = business_df[
            sales_column
        ].sum()

        average_sales = business_df[
            sales_column
        ].mean()

        max_sales = business_df[
            sales_column
        ].max()

        records = len(business_df)

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Total Sales",
            f"{total_sales:,.2f}",
        )

        m2.metric(
            "Average Value",
            f"{average_sales:,.2f}",
        )

        m3.metric(
            "Maximum Value",
            f"{max_sales:,.2f}",
        )

        m4.metric(
            "Business Records",
            f"{records:,}",
        )

        st.markdown(
            f"""
            <div class="dg-info">
                Business charts use the cleaned dataset.
                Values above the IQR visualization upper bound
                <b>{upper_bound:,.2f}</b> are excluded from charts only.
                These records are not deleted.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Sales Distribution")

        hist, bins = np.histogram(
            business_df[sales_column],
            bins=10,
        )

        hist_df = pd.DataFrame(
            {
                "Sales Range": [
                    f"{bins[i]:,.0f} - {bins[i+1]:,.0f}"
                    for i in range(len(bins) - 1)
                ],
                "Records": hist,
            }
        )

        st.bar_chart(
            hist_df.set_index("Sales Range"),
            height=320,
        )

        left, right = st.columns(2)

        with left:

            city_columns = [
                column
                for column in cleaned_df.columns
                if "city" in str(column).lower()
            ]

            if city_columns:

                city_counts = (
                    cleaned_df[
                        city_columns[0]
                    ]
                    .value_counts()
                    .head(10)
                )

                st.markdown("### Records by City")

                st.bar_chart(
                    city_counts,
                    height=320,
                )

        with right:

            product_columns = [
                column
                for column in cleaned_df.columns
                if "product" in str(column).lower()
            ]

            if product_columns:

                product_sales = (
                    business_df
                    .groupby(product_columns[0])[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )

                st.markdown("### Sales by Product")

                st.bar_chart(
                    product_sales,
                    height=320,
                )

        category_columns = [
            column
            for column in cleaned_df.columns
            if "category" in str(column).lower()
        ]

        if category_columns:

            category_sales = (
                business_df
                .groupby(category_columns[0])[
                    sales_column
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.markdown("### Sales by Category")

            st.bar_chart(
                category_sales,
                height=300,
            )

        datetime_columns = analysis[
            "datetime_columns"
        ]

        if datetime_columns:

            date_column = datetime_columns[0]

            monthly_df = cleaned_df.copy()

            monthly_df[date_column] = safe_to_datetime(
                monthly_df[date_column]
            )

            monthly_df[sales_column] = pd.to_numeric(
                monthly_df[sales_column],
                errors="coerce",
            )

            monthly_df = monthly_df.dropna(
                subset=[
                    date_column,
                    sales_column,
                ]
            )

            monthly_sales = (
                monthly_df
                .assign(
                    Month=monthly_df[date_column]
                    .dt.to_period("M")
                    .astype(str)
                )
                .groupby("Month")[sales_column]
                .sum()
            )

            st.markdown("### Monthly Sales Trend")

            st.line_chart(
                monthly_sales,
                height=320,
            )


# ============================================================
# POWER BI
# ============================================================

elif page == "Power BI":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / POWER BI</div>',
        unsafe_allow_html=True,
    )

    st.title("Power BI Export Center")

    st.markdown(
        """
        <div class="page-subtitle">
            Download cleaned and supporting datasets for Power BI Desktop.
        </div>
        """,
        unsafe_allow_html=True,
    )

    exports = create_powerbi_exports(
        cleaned_df,
        business_df,
        analysis["sales_column"],
    )

    st.markdown(
        """
        <div class="dg-info">
            Download the export package, extract the CSV files,
            then use Power BI Desktop → Get Data → Text/CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Available Exports")

    for filename, data in exports.items():

        col1, col2 = st.columns(
            [5, 1]
        )

        with col1:

            st.markdown(
                f"""
                <div class="dg-card">
                    <div class="dg-card-title">
                        {filename}
                    </div>
                    <div class="dg-card-subtitle">
                        Power BI-ready CSV export
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

    st.markdown("### Complete Export Package")

    zip_data = create_zip(exports)

    st.download_button(
        "Download Power BI ZIP",
        data=zip_data,
        file_name="DataGuard_PowerBI_Exports.zip",
        mime="application/zip",
        type="primary",
    )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / AI</div>',
        unsafe_allow_html=True,
    )

    st.title("Gemini AI Analysis")

    st.markdown(
        """
        <div class="page-subtitle">
            AI-generated interpretation based only on the DataGuard quality report.
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
            <div class="dg-warning">
                Gemini API key is not configured.
                Add GEMINI_API_KEY to Streamlit Secrets
                to enable AI analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button(
        "Generate Gemini AI Analysis",
        type="primary",
    ):

        with st.spinner(
            "Gemini is analyzing the quality report..."
        ):

            result = get_gemini_analysis(
                summary_text
            )

        st.session_state.gemini_analysis = result

    if st.session_state.gemini_analysis:

        st.markdown(
            '<div class="dg-card">',
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
            "Generate an AI analysis to receive a portfolio-friendly interpretation."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        '<div class="page-eyebrow">WORKSPACE / REPORTS</div>',
        unsafe_allow_html=True,
    )

    st.title("Data Quality Report")

    st.markdown(
        """
        <div class="page-subtitle">
            Portfolio-ready summary of the DataGuard AI analysis.
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

Extreme sales values excluded from business charts are excluded
only for visualization and are not deleted from the analytical dataset.
"""

    st.markdown("### Report Preview")

    st.code(
        report_text,
        language="text",
    )

    st.download_button(
        "Download Report",
        data=report_text.encode("utf-8"),
        file_name="DataGuard_Data_Quality_Report.txt",
        mime="text/plain",
        type="primary",
    )

    if st.session_state.gemini_analysis:

        st.markdown("### Gemini Analysis")

        st.markdown(
            st.session_state.gemini_analysis
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dg-footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br><br>
        Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
