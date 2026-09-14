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
# SIMPLE PREMIUM CSS
# Only CSS is used here. No HTML UI is required.
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #0b1220;
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

section[data-testid="stSidebar"] .stRadio label {
    color: #e5e7eb !important;
}

/* Main titles */

h1 {
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}

/* Metrics */

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
}

div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 800 !important;
}

/* Buttons */

.stButton > button {
    border-radius: 9px;
    font-weight: 700;
}

/* Download buttons */

.stDownloadButton > button {
    border-radius: 9px;
    font-weight: 700;
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* File uploader */

section[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #94a3b8;
    border-radius: 14px;
    background: white;
}

/* Alerts */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Horizontal line */

hr {
    border-color: #e2e8f0;
}

/* Sidebar divider */

.sidebar-divider {
    border-top: 1px solid #1e293b;
    margin: 18px 0;
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

    except Exception:

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


def detect_datetime_columns(df):

    result = []

    keywords = [
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

        column_name = str(column).lower()

        parsed = safe_to_datetime(
            df[column]
        )

        ratio = parsed.notna().mean()

        if any(
            word in column_name
            for word in keywords
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

    keywords = [
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
            word in name
            for word in keywords
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

            if unique_ratio >= 0.98:

                result.append(column)

    return list(dict.fromkeys(result))


def profile_dataset(df):

    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non-Null": [
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
            "Unique": [
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

    details = []
    total_invalid = 0

    keywords = [
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
            word in name
            for word in keywords
        ):

            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        conversion_failures = (
            numeric.isna()
            & df[column].notna()
        )

        negative_values = numeric < 0

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
            original
            .astype("string")
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
        contamination=contamination_pct / 100,
        random_state=42,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    mask = predictions == -1

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
        return 100

    missing_penalty = (
        missing_count
        / total_cells
        * 100
    )

    duplicate_penalty = (
        duplicate_count
        / total_cells
        * 100
    )

    invalid_penalty = (
        invalid_count
        / total_cells
        * 100
    )

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

        missing = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing > 0:

            median = cleaned[
                column
            ].median()

            if pd.notna(median):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median)
                )

                values_filled += missing

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

        missing = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing > 0:

            modes = cleaned[
                column
            ].mode(dropna=True)

            if len(modes) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(modes.iloc[0])
                )

                values_filled += missing

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
        str(column).lower(): column
        for column in df.columns
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
            word in name
            for word in [
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

    upper_bound = q3 + 1.5 * iqr

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


def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

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
        profile_dataset(
            cleaned_df
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    if (
        business_df is not None
        and sales_column is not None
    ):

        summary = pd.DataFrame(
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
            summary
            .to_csv(index=False)
            .encode("utf-8")
        )

        city_columns = [
            column
            for column in cleaned_df.columns
            if "city" in str(column).lower()
        ]

        if city_columns:

            city = city_columns[0]

            city_summary = (
                business_df
                .groupby(city)[
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
            column
            for column in cleaned_df.columns
            if "category"
            in str(column).lower()
        ]

        if category_columns:

            category = category_columns[0]

            category_summary = (
                business_df
                .groupby(category)[
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

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for filename, data in exports.items():

            archive.writestr(
                filename,
                data,
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 🛡️ DataGuard AI"
    )

    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    st.markdown(
        "**WORKSPACE**"
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

    st.divider()

    st.markdown(
        "**DETECTION SETTINGS**"
    )

    contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
    )

    if (
        contamination
        != st.session_state.contamination_pct
    ):

        st.session_state.contamination_pct = (
            contamination
        )

        st.session_state.analysis_complete = False
        st.session_state.gemini_analysis = None

        st.rerun()

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.divider()

    st.markdown(
        "**PIPELINE**"
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

    for step in pipeline:

        st.caption(step)

    st.divider()

    st.caption(
        "DataGuard AI\n\n"
        "Portfolio Edition\n\n"
        "Python • Pandas • Scikit-learn\n"
        "Streamlit • Gemini"
    )


# ============================================================
# MAIN HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.caption(
        "DATA INTELLIGENCE WORKSPACE"
    )

    st.title(
        "DataGuard AI"
    )

    st.write(
        "AI-powered data quality, anomaly detection "
        "and business analytics."
    )

with header_right:

    st.success(
        "● Engine Online"
    )


# ============================================================
# UPLOAD SCREEN
# ============================================================

if st.session_state.df is None:

    st.divider()

    st.subheader(
        "Protect the quality of your data"
    )

    st.write(
        "Upload your dataset and let DataGuard AI "
        "profile, validate, detect anomalies, clean "
        "and prepare your data for business analytics."
    )

    feature1, feature2, feature3, feature4 = st.columns(
        4
    )

    with feature1:

        st.info(
            "🔍 **Profile Data**\n\n"
            "Understand columns, data types, "
            "missing values and unique values."
        )

    with feature2:

        st.info(
            "◈ **Detect Anomalies**\n\n"
            "Combine IQR statistical detection "
            "with Isolation Forest."
        )

    with feature3:

        st.info(
            "🧹 **Clean Data**\n\n"
            "Remove duplicates, standardize values "
            "and handle missing data."
        )

    with feature4:

        st.info(
            "📊 **Analyze & Export**\n\n"
            "Explore business trends and create "
            "Power BI-ready datasets."
        )

    st.divider()

    st.subheader(
        "Upload your dataset"
    )

    st.caption(
        "Supported formats: CSV, XLSX, XLS"
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
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

            st.session_state.analysis_complete = False
            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.analysis = None
            st.session_state.gemini_analysis = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read file: {error}"
            )


# ============================================================
# RUN ANALYSIS
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

        columns = len(
            df.columns
        )

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
        ) = detect_invalid_values(df)

        quality_score = build_quality_score(
            total_cells,
            missing_count,
            duplicate_count,
            invalid_count,
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

        (
            iqr_count,
            iqr_details,
        ) = detect_iqr_outliers(df)

        (
            anomaly_mask,
            ml_count,
        ) = detect_ml_anomalies(
            df,
            st.session_state.contamination_pct,
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
# ANALYZED DATASET
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

    st.divider()

    file_col, status_col = st.columns(
        [4, 1]
    )

    with file_col:

        st.markdown(
            f"### 📄 {st.session_state.file_name}"
        )

        st.caption(
            f'{analysis["rows"]:,} rows • '
            f'{analysis["columns"]} columns • '
            f'Analysis complete'
        )

    with status_col:

        st.success(
            "✓ Ready"
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.subheader(
            "Dashboard"
        )

        st.caption(
            "Monitor dataset health and quality signals."
        )

        score = analysis[
            "quality_score"
        ]

        if score >= 95:

            score_status = "Good"

        elif score >= 85:

            score_status = "Needs Attention"

        else:

            score_status = "Critical"

        score_col, kpi_col = st.columns(
            [1, 2]
        )

        with score_col:

            st.metric(
                "Data Quality Score",
                f"{score}/100",
                score_status,
            )

            st.progress(
                score / 100
            )

        with kpi_col:

            k1, k2 = st.columns(2)

            with k1:

                st.metric(
                    "Missing Cells",
                    f'{analysis["missing_count"]:,}',
                    "Missing observations",
                )

            with k2:

                st.metric(
                    "Duplicate Records",
                    f'{analysis["duplicate_count"]:,}',
                    "Duplicate rows",
                )

            k3, k4 = st.columns(2)

            with k3:

                st.metric(
                    "IQR Outliers",
                    f'{analysis["iqr_outlier_count"]:,}',
                    "Statistical signals",
                )

            with k4:

                st.metric(
                    "ML Anomalies",
                    f'{analysis["ml_anomaly_count"]:,}',
                    f'{st.session_state.contamination_pct}% sensitivity',
                )

        st.divider()

        st.subheader(
            "Dataset Profile"
        )

        p1, p2, p3, p4, p5 = st.columns(
            5
        )

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

        st.divider()

        st.subheader(
            "Quality Monitoring"
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

            st.divider()

            st.subheader(
                "Business Snapshot"
            )

            b1, b2, b3 = st.columns(
                3
            )

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

        st.divider()

        st.subheader(
            "Data Preview"
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

        st.subheader(
            "Data Quality"
        )

        st.caption(
            "Detailed profiling and rule-based validation."
        )

        q1, q2, q3, q4 = st.columns(
            4
        )

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

        st.divider()

        st.subheader(
            "Column Profile"
        )

        st.dataframe(
            profile_dataset(df),
            use_container_width=True,
            height=450,
        )

        st.divider()

        st.subheader(
            "Invalid Values"
        )

        if analysis[
            "invalid_details"
        ].empty:

            st.success(
                "No invalid values detected by the configured validation rules."
            )

        else:

            st.dataframe(
                analysis[
                    "invalid_details"
                ],
                use_container_width=True,
            )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        st.subheader(
            "Anomaly Detection"
        )

        st.caption(
            "Statistical and machine-learning screening signals."
        )

        st.warning(
            "IQR outliers and Isolation Forest anomalies "
            "are screening signals and are not automatically "
            "confirmed data errors."
        )

        a1, a2, a3 = st.columns(
            3
        )

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

        st.divider()

        st.subheader(
            "IQR Statistical Detection"
        )

        if analysis[
            "iqr_details"
        ].empty:

            st.success(
                "No IQR outliers detected."
            )

        else:

            st.dataframe(
                analysis["iqr_details"],
                use_container_width=True,
            )

        st.divider()

        st.subheader(
            "Isolation Forest"
        )

        st.info(
            f'Current sensitivity: '
            f'{st.session_state.contamination_pct}%. '
            f'Change it from the sidebar to rerun the model.'
        )

        anomaly_df = df.copy()

        anomaly_df[
            "ML_Anomaly"
        ] = np.where(
            analysis["anomaly_mask"],
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

        st.subheader(
            "Automated Cleaning"
        )

        st.caption(
            "Transform raw data into an analysis-ready dataset."
        )

        st.success(
            "Cleaning completed. Duplicate records were "
            "removed, city values standardized and missing "
            "values handled."
        )

        c1, c2, c3, c4 = st.columns(
            4
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

        st.divider()

        st.subheader(
            "Cleaned Dataset"
        )

        st.dataframe(
            cleaned_df.head(10),
            use_container_width=True,
            height=350,
        )

        st.download_button(
            "⬇ Download Cleaned CSV",
            data=(
                cleaned_df
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "DataGuard_Cleaned_Data.csv"
            ),
            mime="text/csv",
            type="primary",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "Analytics":

        st.subheader(
            "Business Analytics"
        )

        st.caption(
            "Explore trends and commercial patterns from the cleaned dataset."
        )

        if business_df is None:

            st.warning(
                "No Sales, Revenue or Amount column was detected."
            )

        else:

            sales_column = analysis[
                "sales_column"
            ]

            st.info(
                f'Business charts use an IQR upper bound of '
                f'{analysis["upper_bound"]:,.2f}. '
                f'This affects visualization only and does not '
                f'delete records from the cleaned dataset.'
            )

            s1, s2, s3, s4 = st.columns(
                4
            )

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

            st.divider()

            st.subheader(
                "Sales Distribution"
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

            city_columns = [
                column
                for column in cleaned_df.columns
                if "city"
                in str(column).lower()
            ]

            if city_columns:

                city = city_columns[0]

                city_sales = (
                    business_df
                    .groupby(city)[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )

                st.divider()

                st.subheader(
                    "Sales by City"
                )

                st.bar_chart(
                    city_sales,
                    height=300,
                )

            category_columns = [
                column
                for column in cleaned_df.columns
                if "category"
                in str(column).lower()
            ]

            if category_columns:

                category = (
                    category_columns[0]
                )

                category_sales = (
                    business_df
                    .groupby(category)[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.divider()

                st.subheader(
                    "Sales by Category"
                )

                st.bar_chart(
                    category_sales,
                    height=300,
                )


    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        st.subheader(
            "Power BI Export Center"
        )

        st.caption(
            "Business-ready datasets prepared for Power BI."
        )

        st.info(
            "Import these CSV files into Power BI Desktop "
            "using Get Data → Text/CSV."
        )

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis["sales_column"],
        )

        for filename, data in exports.items():

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                st.markdown(
                    f"**{filename}**"
                )

                st.caption(
                    "Power BI-ready CSV"
                )

            with col2:

                st.download_button(
                    "Download",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=(
                        f"download_{filename}"
                    ),
                )

            st.divider()

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "⬇ Download Complete Power BI Package",
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

        st.subheader(
            "AI Analysis"
        )

        st.caption(
            "Interpret measured DataGuard signals using Gemini."
        )

        try:

            GEMINI_API_KEY = (
                st.secrets.get(
                    "GEMINI_API_KEY",
                    "",
                )
            )

        except Exception:

            GEMINI_API_KEY = ""

        if not GEMINI_API_KEY:

            GEMINI_API_KEY = os.getenv(
                "GEMINI_API_KEY",
                "",
            )

        if not GEMINI_API_KEY:

            st.warning(
                "Gemini is not configured. Add "
                "GEMINI_API_KEY to Streamlit secrets "
                "to enable AI analysis."
            )

        else:

            if st.button(
                "Generate AI Analysis",
                type="primary",
            ):

                try:

                    from google import genai

                    client = genai.Client(
                        api_key=GEMINI_API_KEY
                    )

                    prompt = f"""
Analyze this DataGuard AI report.

Do not invent facts.

IQR and Isolation Forest findings
are screening signals.

Quality Score:
{analysis["quality_score"]}/100

Rows:
{analysis["rows"]}

Columns:
{analysis["columns"]}

Missing:
{analysis["missing_count"]}

Duplicates:
{analysis["duplicate_count"]}

Invalid:
{analysis["invalid_count"]}

IQR Outliers:
{analysis["iqr_outlier_count"]}

ML Anomalies:
{analysis["ml_anomaly_count"]}

Rows Removed:
{analysis["rows_removed"]}

Values Filled:
{analysis["values_filled"]}

Provide:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations
"""

                    with st.spinner(
                        "Generating AI assessment..."
                    ):

                        response = (
                            client.interactions.create(
                                model="gemini-3.6-flash",
                                input=prompt,
                                generation_config={
                                    "temperature": 0.1
                                },
                            )
                        )

                    st.session_state.gemini_analysis = (
                        response.output_text
                    )

                except Exception as error:

                    st.error(
                        f"Gemini error: {error}"
                    )

            if (
                st.session_state.gemini_analysis
            ):

                st.divider()

                st.subheader(
                    "AI Assessment"
                )

                st.markdown(
                    st.session_state.gemini_analysis
                )


    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.subheader(
            "Data Quality Report"
        )

        st.caption(
            "Complete analysis summary for documentation and portfolio presentation."
        )

        report = f"""
DATAGUARD AI
DATA QUALITY REPORT

Dataset:
{st.session_state.file_name}

Rows:
{analysis["rows"]:,}

Columns:
{analysis["columns"]}

Quality Score:
{analysis["quality_score"]}/100

Missing Cells:
{analysis["missing_count"]:,}

Duplicate Records:
{analysis["duplicate_count"]:,}

Invalid Values:
{analysis["invalid_count"]:,}

IQR Outliers:
{analysis["iqr_outlier_count"]:,}

Isolation Forest Sensitivity:
{st.session_state.contamination_pct}%

ML Anomalies:
{analysis["ml_anomaly_count"]:,}

Rows Removed:
{analysis["rows_removed"]:,}

Values Filled:
{analysis["values_filled"]:,}

City Values Standardized:
{analysis["city_changes"]:,}

Sales Column:
{analysis["sales_column"]}

Business Visualization Upper Bound:
{analysis["upper_bound"]}

NOTE:
IQR outliers and Isolation Forest anomalies are
screening signals and are not automatically confirmed
data errors.
"""

        st.code(
            report,
            language="text",
        )

        st.download_button(
            "⬇ Download Report",
            data=report.encode("utf-8"),
            file_name=(
                "DataGuard_Data_Quality_Report.txt"
            ),
            mime="text/plain",
            type="primary",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform"
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)
