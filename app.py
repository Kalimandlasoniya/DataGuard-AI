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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "page": "Dashboard",
    "df": None,
    "cleaned_df": None,
    "file_name": None,
    "analysis": None,
    "gemini_analysis": None,
    "cleaning_result": None,
    "sensitivity": 5,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# IMPORTANT:
# CSS only. NO HTML MARKUP IS USED FOR THE UI.
# ============================================================

st.markdown(
    """
    <style>

    /* Overall application */

    .stApp {
        background-color: #070b12;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background-color: #090e16;
        border-right: 1px solid #1d2735;
    }

    /* Main headings */

    h1, h2, h3 {
        color: #f5f7fb !important;
    }

    p, label {
        color: #9aa7b8 !important;
    }

    /* Buttons */

    .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid #263244;
        background-color: #0d141f;
        color: #b7c2d1;
        min-height: 38px;
    }

    .stButton > button:hover {
        border-color: #4f7fff;
        color: white;
        background-color: #111c2d;
    }

    /* Metrics */

    [data-testid="stMetric"] {
        background-color: #0d141f;
        border: 1px solid #1d2938;
        border-radius: 12px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"] {
        color: #8391a5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f7fb !important;
    }

    /* Dataframes */

    [data-testid="stDataFrame"] {
        border: 1px solid #1d2938;
        border-radius: 10px;
    }

    /* Inputs */

    input,
    textarea {
        background-color: #0d141f !important;
        color: white !important;
    }

    /* File uploader */

    [data-testid="stFileUploader"] {
        background-color: #0d141f;
        border: 1px dashed #334155;
        border-radius: 12px;
        padding: 10px;
    }

    /* Progress */

    [data-testid="stProgressBar"] {
        border-radius: 10px;
    }

    /* Divider */

    hr {
        border-color: #1d2938;
    }

    /* Sidebar navigation */

    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        border-color: transparent;
        background-color: transparent;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #111a27;
        border-color: #263244;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA HELPERS
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read CSV/XLSX/XLS safely."""

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()


def datetime_columns(df):
    result = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            result.append(column)
            continue

        if df[column].dtype == "object":

            converted = pd.to_datetime(
                df[column],
                errors="coerce"
            )

            if converted.notna().mean() >= 0.80:
                result.append(column)

    return result


def invalid_numeric_values(df):
    invalid = 0

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        # Only inspect columns that appear numeric.
        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        original_non_null = df[column].notna()

        invalid += int(
            (
                converted.isna()
                & original_non_null
            ).sum()
        )

    return invalid


def quality_metrics(df):
    rows = len(df)
    columns = len(df.columns)

    if rows == 0 or columns == 0:
        return {
            "score": 0,
            "missing": 0,
            "duplicates": 0,
            "invalid": 0,
        }

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    invalid = invalid_numeric_values(df)

    missing_rate = missing / (rows * columns)
    duplicate_rate = duplicates / rows
    invalid_rate = invalid / rows

    score = (
        100
        - missing_rate * 50
        - duplicate_rate * 30
        - invalid_rate * 20
    )

    score = max(
        0,
        min(100, score)
    )

    return {
        "score": round(score, 2),
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def iqr_outliers(df):
    total = 0
    details = {}

    for column in numeric_columns(df):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (series < lower)
            | (series > upper)
        )

        count = int(mask.sum())

        total += count

        details[column] = {
            "count": count,
            "lower": float(lower),
            "upper": float(upper),
        }

    return total, details


def isolation_forest(df, sensitivity):
    """Return normal count, anomaly count and mask."""

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty or len(numeric) < 10:
        return (
            len(df),
            0,
            pd.Series(
                False,
                index=df.index
            ),
        )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.fillna(
        numeric.median(numeric_only=True)
    )

    numeric = numeric.fillna(0)

    contamination = max(
        0.001,
        min(0.25, sensitivity / 100)
    )

    try:

        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )

        predictions = model.fit_predict(
            numeric
        )

        mask = pd.Series(
            predictions == -1,
            index=df.index
        )

        anomalies = int(mask.sum())
        normal = int((~mask).sum())

        return normal, anomalies, mask

    except Exception:
        return (
            len(df),
            0,
            pd.Series(
                False,
                index=df.index
            ),
        )


def find_column(df, keywords):

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", " ")
            .strip()
        )

        for keyword in keywords:

            if keyword in name:
                return column

    return None


# ============================================================
# CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    duplicate_count = int(
        cleaned.duplicated().sum()
    )

    cleaned = cleaned.drop_duplicates()

    rows_removed = (
        original_rows
        - len(cleaned)
    )

    missing_before = int(
        cleaned.isna().sum().sum()
    )

    # Numeric missing values
    for column in numeric_columns(cleaned):

        if cleaned[column].isna().any():

            median = cleaned[column].median()

            if pd.notna(median):
                cleaned[column] = (
                    cleaned[column]
                    .fillna(median)
                )

    # Categorical missing values
    for column in categorical_columns(cleaned):

        if cleaned[column].isna().any():

            mode = cleaned[column].mode()

            if len(mode) > 0:
                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode.iloc[0])
                )
            else:
                cleaned[column] = (
                    cleaned[column]
                    .fillna("Unknown")
                )

    missing_after = int(
        cleaned.isna().sum().sum()
    )

    values_filled = (
        missing_before
        - missing_after
    )

    # Standardize City if present
    city_column = find_column(
        cleaned,
        ["city"]
    )

    city_standardized = 0

    if city_column:

        before = (
            cleaned[city_column]
            .astype(str)
            .copy()
        )

        cleaned[city_column] = (
            cleaned[city_column]
            .astype(str)
            .str.strip()
            .str.title()
        )

        city_standardized = int(
            (
                before
                != cleaned[city_column]
            ).sum()
        )

    result = {
        "rows_removed": rows_removed,
        "duplicates_removed": duplicate_count,
        "values_filled": values_filled,
        "city_standardized": city_standardized,
    }

    return cleaned, result


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

def business_metrics(df):

    sales_column = find_column(
        df,
        [
            "sales",
            "revenue",
            "amount",
        ]
    )

    profit_column = find_column(
        df,
        [
            "profit",
            "net profit",
        ]
    )

    quantity_column = find_column(
        df,
        [
            "quantity",
            "units",
        ]
    )

    result = {
        "sales_column": sales_column,
        "profit_column": profit_column,
        "quantity_column": quantity_column,
    }

    if sales_column:

        sales = pd.to_numeric(
            df[sales_column],
            errors="coerce"
        )

        result["total_sales"] = float(
            sales.sum()
        )

        result["average_sale"] = float(
            sales.mean()
        )

        result["highest_sale"] = float(
            sales.max()
        )

    if profit_column:

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce"
        )

        result["total_profit"] = float(
            profit.sum()
        )

    if (
        sales_column
        and profit_column
    ):

        sales = pd.to_numeric(
            df[sales_column],
            errors="coerce"
        )

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce"
        )

        total_sales = sales.sum()

        if total_sales != 0:

            result["profit_margin"] = (
                profit.sum()
                / total_sales
                * 100
            )

    if quantity_column:

        quantity = pd.to_numeric(
            df[quantity_column],
            errors="coerce"
        )

        result["total_quantity"] = float(
            quantity.sum()
        )

    return result


# ============================================================
# POWER BI EXPORT
# ============================================================

def powerbi_excel(df):

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Cleaned_Data",
            index=False
        )

        profile = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(df[c].dtype)
                for c in df.columns
            ],
            "Missing Values": [
                int(df[c].isna().sum())
                for c in df.columns
            ],
            "Unique Values": [
                int(df[c].nunique())
                for c in df.columns
            ],
        })

        profile.to_excel(
            writer,
            sheet_name="Data_Profile",
            index=False
        )

        business = business_metrics(df)

        pd.DataFrame(
            [business]
        ).to_excel(
            writer,
            sheet_name="Business_Summary",
            index=False
        )

    return buffer.getvalue()


def create_zip(df):

    excel_data = powerbi_excel(df)

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "DataGuard_Cleaned_Data.csv",
            csv_data
        )

        archive.writestr(
            "DataGuard_PowerBI.xlsx",
            excel_data
        )

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

def generate_gemini_analysis(
    dataset_name,
    df,
    quality,
    outliers,
    normal,
    anomalies,
    sensitivity,
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "Gemini API key is not configured.\n\n"
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable AI analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        business = business_metrics(df)

        prompt = f"""
You are a senior data quality and business analytics consultant.

Analyze ONLY the measured information below.

Dataset:
{dataset_name}

Rows:
{len(df)}

Columns:
{len(df.columns)}

Quality Score:
{quality["score"]}%

Missing Values:
{quality["missing"]}

Duplicates:
{quality["duplicates"]}

Invalid Values:
{quality["invalid"]}

IQR Outliers:
{outliers}

Isolation Forest Normal Records:
{normal}

Isolation Forest Anomalies:
{anomalies}

Isolation Forest Sensitivity:
{sensitivity}%

Business Metrics:
{business}

Rules:
1. Do not invent numbers.
2. Do not change measured numbers.
3. Do not claim anomalies are confirmed errors.
4. IQR outliers are statistical signals.
5. Isolation Forest anomalies are screening signals.
6. Give practical recommendations.

Use these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Recommendations
7. Business Impact
8. Power BI Recommendations
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = getattr(
            response,
            "text",
            None
        )

        if text:
            return text

        return str(response)

    except Exception as error:

        return (
            "Gemini analysis failed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ DataGuard AI")
    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    st.subheader("WORKSPACE")

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

    for page_name in pages:

        if st.button(
            page_name,
            key=f"page_{page_name}",
            type=(
                "primary"
                if st.session_state.page == page_name
                else "secondary"
            ),
        ):

            st.session_state.page = page_name
            st.rerun()

    st.divider()

    st.subheader(
        "DETECTION SETTINGS"
    )

    st.session_state.sensitivity = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=int(
            st.session_state.sensitivity
        ),
        help=(
            "Higher sensitivity flags more "
            "records as unusual."
        ),
    )

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.divider()

    st.subheader("SYSTEM")

    st.success(
        "All systems operational"
    )


# ============================================================
# COMMON HEADER
# ============================================================

st.caption(
    "DATA INTELLIGENCE WORKSPACE"
)

st.title(
    st.session_state.page
)

st.caption(
    "AI-powered data quality, anomaly detection "
    "and business analytics."
)


# ============================================================
# WORKFLOW
# ============================================================

workflow = [
    "Upload",
    "Profile",
    "Quality",
    "Anomalies",
    "Clean",
    "Analytics",
    "Power BI",
    "AI",
]

current_page = st.session_state.page

workflow_map = {
    "Dashboard": 0,
    "Upload Data": 1,
    "Data Profile": 2,
    "Quality Checks": 3,
    "Anomaly Detection": 4,
    "Data Cleaning": 5,
    "Analytics": 6,
    "Power BI": 7,
    "AI Analysis": 8,
    "Reports": 8,
}

current_step = workflow_map.get(
    current_page,
    0
)

workflow_columns = st.columns(
    len(workflow)
)

for index, (
    column,
    step_name
) in enumerate(
    zip(
        workflow_columns,
        workflow
    ),
    start=1,
):

    with column:

        if current_step == 0:
            label = f"{index:02d} {step_name}"

        elif index < current_step:
            label = f"✓ {step_name}"

        elif index == current_step:
            label = f"● {step_name}"

        else:
            label = f"{index:02d} {step_name}"

        st.caption(label)


st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if current_page == "Dashboard":

    if st.session_state.df is None:

        st.subheader(
            "Welcome to DataGuard AI"
        )

        st.write(
            "Upload a dataset to begin your "
            "data intelligence workflow."
        )

        st.info(
            "Supported formats: CSV, XLSX, XLS • "
            "Maximum upload size: 50 MB"
        )

        st.subheader(
            "Upload your dataset"
        )

        uploaded = st.file_uploader(
            "Choose a dataset",
            type=[
                "csv",
                "xlsx",
                "xls",
            ],
        )

        if uploaded:

            try:

                df = read_uploaded_file(
                    uploaded
                )

                st.session_state.df = df
                st.session_state.file_name = (
                    uploaded.name
                )

                st.success(
                    f"{uploaded.name} uploaded successfully."
                )

                st.session_state.page = (
                    "Data Profile"
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Could not read file: {error}"
                )

    else:

        df = st.session_state.df

        quality = quality_metrics(
            df
        )

        normal, anomalies, mask = (
            isolation_forest(
                df,
                st.session_state.sensitivity,
            )
        )

        st.caption(
            f"Dataset: {st.session_state.file_name}"
        )

        columns = st.columns(6)

        metrics = [
            (
                "Quality Score",
                f'{quality["score"]:.2f}%'
            ),
            (
                "Records",
                f"{len(df):,}"
            ),
            (
                "Columns",
                f"{len(df.columns):,}"
            ),
            (
                "Missing",
                f'{quality["missing"]:,}'
            ),
            (
                "Duplicates",
                f'{quality["duplicates"]:,}'
            ),
            (
                "Anomalies",
                f"{anomalies:,}"
            ),
        ]

        for column, (
            label,
            value
        ) in zip(
            columns,
            metrics
        ):

            with column:

                st.metric(
                    label,
                    value
                )

        st.subheader(
            "Data Quality Overview"
        )

        left, right = st.columns(2)

        with left:

            st.metric(
                "Quality Score",
                f'{quality["score"]:.2f}%'
            )

            st.progress(
                int(
                    quality["score"]
                )
            )

            st.caption(
                "Score is based on missing values, "
                "duplicates and invalid values."
            )

        with right:

            st.metric(
                "Anomaly Rate",
                f"{anomalies / len(df) * 100:.2f}%"
                if len(df)
                else "0.00%"
            )

            st.write(
                f"Normal records: {normal:,}"
            )

            st.write(
                f"Unusual records: {anomalies:,}"
            )

            st.caption(
                "Isolation Forest results are "
                "screening signals."
            )


# ============================================================
# UPLOAD PAGE
# ============================================================

elif current_page == "Upload Data":

    st.subheader(
        "Upload your dataset"
    )

    st.write(
        "CSV, XLSX or XLS files"
    )

    st.info(
        "Maximum file size: 50 MB"
    )

    uploaded = st.file_uploader(
        "Choose a dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
    )

    if uploaded:

        try:

            df = read_uploaded_file(
                uploaded
            )

            st.session_state.df = df
            st.session_state.cleaned_df = None
            st.session_state.file_name = (
                uploaded.name
            )
            st.session_state.analysis = None
            st.session_state.gemini_analysis = None

            st.success(
                f"{uploaded.name} uploaded successfully."
            )

            st.write(
                f"Rows: {len(df):,}"
            )

            st.write(
                f"Columns: {len(df.columns):,}"
            )

        except Exception as error:

            st.error(
                f"Could not read file: {error}"
            )


# ============================================================
# DATA PROFILE
# ============================================================

elif current_page == "Data Profile":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        st.subheader(
            "Dataset Overview"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Rows",
                f"{len(df):,}"
            )

        with c2:
            st.metric(
                "Columns",
                f"{len(df.columns):,}"
            )

        with c3:
            st.metric(
                "Numeric",
                len(numeric_columns(df))
            )

        with c4:
            st.metric(
                "Categorical",
                len(categorical_columns(df))
            )

        st.subheader(
            "Column Profile"
        )

        profile = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
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
                    df[column].nunique()
                )
                for column in df.columns
            ],
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Preview"
        )

        st.dataframe(
            df.head(50),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

elif current_page == "Quality Checks":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        quality = quality_metrics(df)

        outliers, details = (
            iqr_outliers(df)
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Quality Score",
                f'{quality["score"]:.2f}%'
            )

        with c2:
            st.metric(
                "Missing",
                f'{quality["missing"]:,}'
            )

        with c3:
            st.metric(
                "Duplicates",
                f'{quality["duplicates"]:,}'
            )

        with c4:
            st.metric(
                "Invalid",
                f'{quality["invalid"]:,}'
            )

        with c5:
            st.metric(
                "IQR Outliers",
                f"{outliers:,}"
            )

        st.subheader(
            "Quality Assessment"
        )

        st.progress(
            int(quality["score"])
        )

        st.write(
            f"Current quality score: "
            f"**{quality['score']:.2f}%**"
        )

        if quality["missing"] == 0:
            st.success(
                "No missing values detected."
            )
        else:
            st.warning(
                f'{quality["missing"]:,} missing values detected.'
            )

        if quality["duplicates"] == 0:
            st.success(
                "No duplicate rows detected."
            )
        else:
            st.warning(
                f'{quality["duplicates"]:,} duplicate rows detected.'
            )

        if quality["invalid"] == 0:
            st.success(
                "No invalid numeric values detected."
            )
        else:
            st.warning(
                f'{quality["invalid"]:,} invalid values detected.'
            )

        st.subheader(
            "IQR Outlier Details"
        )

        if details:

            outlier_table = pd.DataFrame([
                {
                    "Column": column,
                    "Outliers": value["count"],
                    "Lower Bound": value["lower"],
                    "Upper Bound": value["upper"],
                }
                for column, value in details.items()
            ])

            st.dataframe(
                outlier_table,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No IQR outliers detected."
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif current_page == "Anomaly Detection":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        normal, anomalies, mask = (
            isolation_forest(
                df,
                st.session_state.sensitivity,
            )
        )

        rate = (
            anomalies / len(df) * 100
            if len(df)
            else 0
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Normal Records",
                f"{normal:,}"
            )

        with c2:
            st.metric(
                "Anomalies",
                f"{anomalies:,}"
            )

        with c3:
            st.metric(
                "Anomaly Rate",
                f"{rate:.2f}%"
            )

        st.subheader(
            "Isolation Forest"
        )

        st.write(
            "Isolation Forest identifies records "
            "that behave unusually compared with "
            "the rest of the dataset."
        )

        st.caption(
            f"Sensitivity: "
            f"{st.session_state.sensitivity}%"
        )

        anomaly_df = df.loc[
            mask
        ].copy()

        anomaly_df.insert(
            0,
            "Anomaly",
            "Yes"
        )

        st.dataframe(
            anomaly_df.head(100),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# DATA CLEANING
# ============================================================

elif current_page == "Data Cleaning":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        st.subheader(
            "Data Cleaning"
        )

        st.write(
            "The cleaning process removes duplicate rows, "
            "fills missing values and standardizes city names "
            "when a City column exists."
        )

        if st.button(
            "Run Data Cleaning",
            type="primary",
        ):

            cleaned, result = clean_dataset(
                st.session_state.df
            )

            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_result = result

            st.success(
                "Cleaning completed successfully."
            )

        if (
            st.session_state.cleaned_df
            is not None
        ):

            result = (
                st.session_state.cleaning_result
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Rows Removed",
                    result["rows_removed"]
                )

            with c2:
                st.metric(
                    "Values Filled",
                    result["values_filled"]
                )

            with c3:
                st.metric(
                    "City Standardized",
                    result["city_standardized"]
                )

            with c4:
                st.metric(
                    "Final Rows",
                    f"{len(st.session_state.cleaned_df):,}"
                )

            st.subheader(
                "Cleaned Data Preview"
            )

            st.dataframe(
                st.session_state.cleaned_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

            csv_data = (
                st.session_state.cleaned_df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                "Download Cleaned CSV",
                csv_data,
                file_name="DataGuard_Cleaned_Data.csv",
                mime="text/csv",
            )


# ============================================================
# ANALYTICS
# ============================================================

elif current_page == "Analytics":

    df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.df
    )

    if df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        st.subheader(
            "Business Analytics"
        )

        metrics = business_metrics(df)

        if not metrics.get(
            "sales_column"
        ):

            st.info(
                "No Sales, Revenue or Amount column "
                "was detected."
            )

        else:

            sales = metrics[
                "sales_column"
            ]

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "Total Sales",
                    f'{metrics["total_sales"]:,.2f}'
                )

            with c2:
                st.metric(
                    "Average Sale",
                    f'{metrics["average_sale"]:,.2f}'
                )

            with c3:
                st.metric(
                    "Highest Sale",
                    f'{metrics["highest_sale"]:,.2f}'
                )

            with c4:

                if "profit_margin" in metrics:

                    st.metric(
                        "Profit Margin",
                        f'{metrics["profit_margin"]:.2f}%'
                    )

                else:

                    st.metric(
                        "Profit Margin",
                        "N/A"
                    )

            st.subheader(
                "Sales Distribution"
            )

            sales_series = pd.to_numeric(
                df[sales],
                errors="coerce"
            ).dropna()

            if not sales_series.empty:

                st.line_chart(
                    sales_series.reset_index(
                        drop=True
                    )
                )

            # Category analysis
            category = find_column(
                df,
                ["category"]
            )

            if category:

                category_sales = (
                    df.groupby(category)[sales]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.subheader(
                    "Sales by Category"
                )

                st.bar_chart(
                    category_sales
                )


# ============================================================
# POWER BI
# ============================================================

elif current_page == "Power BI":

    df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.df
    )

    if df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        st.subheader(
            "Power BI Export Center"
        )

        st.write(
            "Export your data and supporting tables "
            "for Power BI reporting."
        )

        excel_data = powerbi_excel(
            df
        )

        zip_data = create_zip(
            df
        )

        st.download_button(
            "Download Power BI Excel",
            excel_data,
            file_name="DataGuard_PowerBI.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            type="primary",
        )

        st.download_button(
            "Download Complete Power BI Package",
            zip_data,
            file_name="DataGuard_PowerBI_Package.zip",
            mime="application/zip",
        )

        st.write(
            "The package contains:"
        )

        st.write(
            "• Cleaned data"
        )

        st.write(
            "• Data profile"
        )

        st.write(
            "• Business summary"
        )


# ============================================================
# AI ANALYSIS
# ============================================================

elif current_page == "AI Analysis":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else st.session_state.df
        )

        quality = quality_metrics(
            df
        )

        outliers, _ = (
            iqr_outliers(df)
        )

        normal, anomalies, _ = (
            isolation_forest(
                df,
                st.session_state.sensitivity,
            )
        )

        st.subheader(
            "Gemini AI Analysis"
        )

        st.write(
            "Generate an AI-assisted interpretation "
            "using the measured results from DataGuard AI."
        )

        if st.button(
            "Generate AI Analysis",
            type="primary",
        ):

            with st.spinner(
                "Generating analysis..."
            ):

                result = generate_gemini_analysis(
                    st.session_state.file_name,
                    df,
                    quality,
                    outliers,
                    normal,
                    anomalies,
                    st.session_state.sensitivity,
                )

                st.session_state.gemini_analysis = (
                    result
                )

        if st.session_state.gemini_analysis:

            st.markdown(
                st.session_state.gemini_analysis
            )


# ============================================================
# REPORTS
# ============================================================

elif current_page == "Reports":

    if st.session_state.df is None:

        st.warning(
            "Please upload a dataset first."
        )

    else:

        df = st.session_state.df

        quality = quality_metrics(
            df
        )

        outliers, _ = (
            iqr_outliers(df)
        )

        normal, anomalies, _ = (
            isolation_forest(
                df,
                st.session_state.sensitivity,
            )
        )

        report = f"""
DATAGUARD AI
DATA QUALITY REPORT

Dataset:
{st.session_state.file_name}

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M")}

----------------------------------------
DATASET OVERVIEW
----------------------------------------

Rows: {len(df):,}
Columns: {len(df.columns):,}

----------------------------------------
DATA QUALITY
----------------------------------------

Quality Score: {quality["score"]:.2f}%
Missing Values: {quality["missing"]:,}
Duplicates: {quality["duplicates"]:,}
Invalid Values: {quality["invalid"]:,}

----------------------------------------
STATISTICAL OUTLIERS
----------------------------------------

IQR Outliers: {outliers:,}

----------------------------------------
ML ANOMALIES
----------------------------------------

Normal Records: {normal:,}
Anomalies: {anomalies:,}
Sensitivity: {st.session_state.sensitivity}%

----------------------------------------
END OF REPORT
----------------------------------------
"""

        st.subheader(
            "DataGuard AI Report"
        )

        st.text_area(
            "Report Preview",
            report,
            height=400,
        )

        st.download_button(
            "Download Report",
            report,
            file_name="DataGuard_AI_Report.txt",
            mime="text/plain",
            type="primary",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ DataGuard AI • AI Data Quality & "
    "Anomaly Detection Platform"
)

st.caption(
    "Python • Pandas • Scikit-learn • "
    "Streamlit • Gemini"
)
