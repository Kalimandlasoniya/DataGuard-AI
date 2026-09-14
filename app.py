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
# PREMIUM UI
# ============================================================

st.markdown(
    """
    <style>

    /* ================= GLOBAL ================= */

    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Hide Streamlit heading anchor icons */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -1px;
    }

    h2 {
        color: #0f172a;
        font-weight: 750;
    }

    h3 {
        color: #1e293b;
        font-weight: 700;
    }

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding: 8px 10px;
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #1e293b;
    }

    /* ================= METRICS ================= */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-size: 0.84rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 800;
    }

    /* ================= DATASET BANNER ================= */

    .dataset-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px 22px;
        margin: 15px 0 25px 0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .dataset-name {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
    }

    .dataset-meta {
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 5px;
    }

    .status-good {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        padding: 4px 11px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* ================= SCORE ================= */

    .score-card {
        background: #ffffff;
        border: 1px solid #dbe4ee;
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        margin: 12px 0 20px 0;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }

    .score-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .score-number {
        color: #0f172a;
        font-size: 3.2rem;
        font-weight: 850;
        line-height: 1.1;
        margin: 8px 0;
    }

    .score-status {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* ================= CARDS ================= */

    .dg-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .dg-card-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .dg-card-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 5px;
    }

    .dg-card-subtitle {
        color: #64748b;
        font-size: 0.86rem;
        margin-top: 3px;
    }

    /* ================= BUTTONS ================= */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }

    /* ================= TABLE ================= */

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ================= FOOTER ================= */

    .dg-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        padding: 35px 0 10px 0;
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
        converted = pd.to_datetime(series, errors="coerce")
        if converted.notna().mean() >= 0.60:
            return converted
    except Exception:
        pass

    return None


def detect_datetime_columns(df):
    columns = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            columns.append(col)
            continue

        if df[col].dtype == "object":
            converted = safe_to_datetime(df[col])

            if converted is not None:
                columns.append(col)

    return columns


def detect_identifier_columns(df, datetime_columns=None):

    datetime_columns = datetime_columns or []

    identifiers = []

    for col in df.columns:

        if col in datetime_columns:
            continue

        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        name = str(col).lower()

        if (
            unique_ratio > 0.95
            or name.endswith("id")
            or name == "id"
            or "order id" in name
            or "customer id" in name
            or "product id" in name
        ):
            identifiers.append(col)

    return identifiers


def profile_dataset(df):

    datetime_columns = detect_datetime_columns(df)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_count": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
    }


def detect_invalid_values(df):

    invalid_count = 0
    invalid_details = {}

    for col in df.columns:

        series = df[col]

        if pd.api.types.is_numeric_dtype(series):

            invalid = series.isin([np.inf, -np.inf]).sum()

            if invalid:
                invalid_details[col] = int(invalid)
                invalid_count += int(invalid)

    return invalid_count, invalid_details


def standardize_city_column(df):

    result = df.copy()

    for col in result.columns:

        if result[col].dtype == "object":

            name = str(col).lower()

            if "city" in name:

                result[col] = (
                    result[col]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .replace(CITY_MAPPING)
                )

    return result


def detect_iqr_outliers(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    total_outliers = 0
    details = {}

    for col in numeric_columns:

        series = df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (df[col] < lower) | (df[col] > upper)

        count = int(mask.sum())

        if count:
            details[col] = {
                "count": count,
                "lower": float(lower),
                "upper": float(upper),
            }

            total_outliers += count

    return total_outliers, details


def detect_ml_anomalies(df, contamination_pct):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) < 2:
        return 0, pd.Series(False, index=df.index)

    work = df[numeric_columns].copy()

    work = work.replace([np.inf, -np.inf], np.nan)

    work = work.fillna(work.median(numeric_only=True))

    if len(work) < 20:
        return 0, pd.Series(False, index=df.index)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_pct / 100,
        random_state=42,
    )

    predictions = model.fit_predict(work)

    anomaly_mask = predictions == -1

    return int(anomaly_mask.sum()), pd.Series(
        anomaly_mask,
        index=df.index,
    )


def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
):

    if total_cells == 0:
        return 0

    missing_penalty = (
        missing_count / total_cells * 100
    )

    duplicate_penalty = (
        duplicate_count / total_cells * 100
    )

    invalid_penalty = (
        invalid_count / total_cells * 100
    )

    score = 100 - (
        missing_penalty * 0.50
        + duplicate_penalty * 0.30
        + invalid_penalty * 0.20
    )

    return max(0, min(100, score))


def clean_dataset(df):

    cleaned = df.copy()

    cleaned = standardize_city_column(cleaned)

    cleaned = cleaned.drop_duplicates()

    for col in cleaned.columns:

        if pd.api.types.is_numeric_dtype(cleaned[col]):

            median = cleaned[col].median()

            if pd.notna(median):
                cleaned[col] = cleaned[col].fillna(median)

        elif cleaned[col].dtype == "object":

            mode = cleaned[col].mode(dropna=True)

            if not mode.empty:
                cleaned[col] = cleaned[col].fillna(mode.iloc[0])

    cleaned = cleaned.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return cleaned


def find_sales_column(df):

    keywords = [
        "sales",
        "sale",
        "revenue",
        "amount",
        "total",
        "price",
        "value",
    ]

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for keyword in keywords:

        for col in numeric_columns:

            if keyword in str(col).lower():
                return col

    return None


def prepare_business_data(df):

    sales_column = find_sales_column(df)

    if sales_column is None:
        return df.copy(), None

    business = df.copy()

    return business, sales_column


def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

    profile = profile_dataset(cleaned_df)

    profile_df = pd.DataFrame(
        [
            ["Rows", profile["rows"]],
            ["Columns", profile["columns"]],
            ["Numeric Columns", len(profile["numeric_columns"])],
            ["Categorical Columns", len(profile["categorical_columns"])],
            ["Date Columns", len(profile["datetime_columns"])],
            ["Missing Cells", profile["missing_count"]],
            ["Duplicate Rows", profile["duplicate_count"]],
        ],
        columns=["Metric", "Value"],
    )

    exports["DataGuard_Cleaned_Data.csv"] = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    exports["DataGuard_Data_Profile.csv"] = profile_df.to_csv(
        index=False
    ).encode("utf-8")

    if sales_column:

        sales_summary = pd.DataFrame(
            {
                "Metric": [
                    "Total Sales",
                    "Average Sale",
                    "Minimum Sale",
                    "Maximum Sale",
                    "Sales Records",
                ],
                "Value": [
                    business_df[sales_column].sum(),
                    business_df[sales_column].mean(),
                    business_df[sales_column].min(),
                    business_df[sales_column].max(),
                    business_df[sales_column].count(),
                ],
            }
        )

        exports["DataGuard_Sales_Summary.csv"] = (
            sales_summary.to_csv(index=False).encode("utf-8")
        )

        city_columns = [
            c for c in business_df.columns
            if "city" in str(c).lower()
        ]

        if city_columns:

            city_col = city_columns[0]

            city_summary = (
                business_df
                .groupby(city_col)[sales_column]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Sales_Records="count",
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False,
                )
            )

            exports["DataGuard_City_Summary.csv"] = (
                city_summary.to_csv(index=False)
                .encode("utf-8")
            )

        category_columns = [
            c for c in business_df.columns
            if "category" in str(c).lower()
        ]

        if category_columns:

            category_col = category_columns[0]

            category_summary = (
                business_df
                .groupby(category_col)[sales_column]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Sales_Records="count",
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False,
                )
            )

            exports["DataGuard_Category_Summary.csv"] = (
                category_summary.to_csv(index=False)
                .encode("utf-8")
            )

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        for filename, content in exports.items():
            z.writestr(filename, content)

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

def get_gemini_api_key():

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


def generate_gemini_analysis(analysis):

    api_key = get_gemini_api_key()

    if not api_key:
        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are a senior data quality analyst.

Analyze ONLY the supplied DataGuard AI report.

Do not invent facts.

IQR outliers and Isolation Forest anomalies are
screening signals, not confirmed errors.

Cleaning has already been performed.

Sales visualization filtering does not delete records.

DATA QUALITY REPORT:

{analysis}

Return these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

Clearly label hypotheses as possible explanations.
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "temperature": 0.1,
            },
        )

        return response.output_text

    except Exception as e:

        return f"Gemini analysis failed: {e}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            margin-bottom:3px;
        ">
            🛡️ DataGuard AI
        </div>

        <div style="
            color:#94a3b8;
            font-size:13px;
            margin-bottom:25px;
        ">
            AI Data Quality Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### WORKSPACE")

    pages = [
        "Dashboard",
        "Data Quality",
        "Anomalies",
        "Cleaning",
        "Analytics",
        "Power BI",
        "AI Analysis",
        "Reports",
    ]

    page = st.radio(
        "Navigation",
        pages,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### DETECTION SETTINGS")

    contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.contamination_pct,
        help="Higher sensitivity flags more records as unusual.",
    )

    st.session_state.contamination_pct = contamination

    st.divider()

    st.markdown("### PIPELINE")

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

    st.caption("DataGuard AI")
    st.caption("Portfolio Edition")
    st.caption(
        "Python • Pandas • Scikit-learn"
    )
    st.caption(
        "Streamlit • Gemini"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("DataGuard AI")

st.caption(
    "AI-powered data quality, anomaly detection and business analytics."
)

if st.session_state.file_name:

    st.markdown(
        f"""
        <div class="dataset-banner">

            <div class="dataset-name">
                📄 {st.session_state.file_name}
            </div>

            <div class="dataset-meta">
                {len(st.session_state.df):,} rows
                •
                {len(st.session_state.df.columns):,} columns
                •
                Analysis complete
                &nbsp;&nbsp;
                <span class="status-good">✓ Ready</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD
# ============================================================

with st.expander(
    "Upload dataset",
    expanded=st.session_state.df is None,
):

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:

        try:

            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.analysis_complete = False
            st.session_state.analysis = None
            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.gemini_analysis = None

            st.success(
                f"Loaded {len(df):,} rows × "
                f"{len(df.columns):,} columns."
            )

        except Exception as e:

            st.error(
                f"Unable to read file: {e}"
            )


# ============================================================
# RUN ANALYSIS
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if st.button(
        "Run DataGuard Analysis",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Profiling dataset and detecting quality issues..."
        ):

            total_cells = (
                df.shape[0] * df.shape[1]
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

            iqr_outlier_count, iqr_details = (
                detect_iqr_outliers(df)
            )

            ml_anomaly_count, anomaly_mask = (
                detect_ml_anomalies(
                    df,
                    st.session_state.contamination_pct,
                )
            )

            quality_score = build_quality_score(
                total_cells,
                missing_count,
                duplicate_count,
                invalid_count,
            )

            cleaned_df = clean_dataset(df)

            business_df, sales_column = (
                prepare_business_data(
                    cleaned_df
                )
            )

            analysis = {
                "quality_score": quality_score,
                "total_cells": total_cells,
                "missing_count": missing_count,
                "duplicate_count": duplicate_count,
                "invalid_count": invalid_count,
                "invalid_details": invalid_details,
                "iqr_outlier_count": iqr_outlier_count,
                "iqr_details": iqr_details,
                "ml_anomaly_count": ml_anomaly_count,
                "sales_column": sales_column,
                "anomaly_mask": anomaly_mask,
            }

            st.session_state.analysis = analysis
            st.session_state.cleaned_df = cleaned_df
            st.session_state.business_df = business_df
            st.session_state.analysis_complete = True

        st.success(
            "Analysis completed successfully."
        )


# ============================================================
# CHECK DATA
# ============================================================

if st.session_state.df is None:

    st.info(
        "Upload a CSV or Excel dataset to begin."
    )

    st.markdown(
        """
        ### DataGuard AI Pipeline

        **Upload → Profile → Quality → Anomalies → Clean → Analytics → Power BI → AI**

        Use the sidebar to navigate between analysis modules.
        """
    )

    st.stop()


df = st.session_state.df


# ============================================================
# REQUIRE ANALYSIS
# ============================================================

if not st.session_state.analysis_complete:

    st.warning(
        "Run DataGuard Analysis to unlock the dashboard."
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


analysis = st.session_state.analysis
cleaned_df = st.session_state.cleaned_df
business_df = st.session_state.business_df


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader("Dashboard")

    st.caption(
        "Monitor dataset health and quality signals."
    )

    # Quality score

    score = analysis["quality_score"]

    if score >= 90:
        status = "Good"

    elif score >= 75:
        status = "Needs Attention"

    else:
        status = "Critical"

    st.markdown(
        f"""
        <div class="score-card">

            <div class="score-label">
                Overall Data Quality
            </div>

            <div class="score-number">
                {score:.2f}/100
            </div>

            <span class="score-status">
                {status}
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        min(score / 100, 1.0)
    )

    # Quality signals

    st.subheader("Quality Signals")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Missing Cells",
            f"{analysis['missing_count']:,}",
        )

    with c2:
        st.metric(
            "Duplicate Records",
            f"{analysis['duplicate_count']:,}",
        )

    with c3:
        st.metric(
            "IQR Outliers",
            f"{analysis['iqr_outlier_count']:,}",
        )

    with c4:
        st.metric(
            "ML Anomalies",
            f"{analysis['ml_anomaly_count']:,}",
        )

    # Dataset profile

    st.subheader("Dataset Profile")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    datetime_columns = detect_datetime_columns(
        df
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.metric(
            "Rows",
            f"{len(df):,}",
        )

    with p2:
        st.metric(
            "Columns",
            f"{len(df.columns):,}",
        )

    with p3:
        st.metric(
            "Numeric",
            f"{len(numeric_columns):,}",
        )

    with p4:
        st.metric(
            "Categorical",
            f"{len(categorical_columns):,}",
        )

    with p5:
        st.metric(
            "Date / Time",
            f"{len(datetime_columns):,}",
        )

    # Quality chart

    st.subheader("Quality Monitoring")

    quality_data = pd.DataFrame(
        {
            "Signal": [
                "Missing",
                "Duplicate",
                "IQR Outliers",
                "ML Anomalies",
            ],
            "Count": [
                analysis["missing_count"],
                analysis["duplicate_count"],
                analysis["iqr_outlier_count"],
                analysis["ml_anomaly_count"],
            ],
        }
    )

    st.bar_chart(
        quality_data.set_index("Signal"),
        horizontal=True,
    )

    # Business snapshot

    st.subheader("Business Snapshot")

    sales_column = analysis["sales_column"]

    if sales_column:

        total_sales = (
            business_df[sales_column].sum()
        )

        average_sale = (
            business_df[sales_column].mean()
        )

        sales_records = (
            business_df[sales_column].count()
        )

        b1, b2, b3 = st.columns(3)

        with b1:
            st.metric(
                "Total Sales",
                f"{total_sales:,.0f}",
            )

        with b2:
            st.metric(
                "Average Sale",
                f"{average_sale:,.2f}",
            )

        with b3:
            st.metric(
                "Sales Records",
                f"{sales_records:,}",
            )

    else:

        st.info(
            "No obvious sales/revenue column was detected."
        )

    # Preview

    st.subheader("Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        height=420,
        hide_index=True,
    )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.subheader("Data Quality")

    st.caption(
        "Detailed monitoring of missing values, duplicates and invalid values."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Missing Cells",
            f"{analysis['missing_count']:,}",
        )

    with c2:
        st.metric(
            "Duplicate Rows",
            f"{analysis['duplicate_count']:,}",
        )

    with c3:
        st.metric(
            "Invalid Values",
            f"{analysis['invalid_count']:,}",
        )

    st.divider()

    missing_table = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing_table.columns = [
        "Column",
        "Missing_Count",
    ]

    missing_table = missing_table[
        missing_table["Missing_Count"] > 0
    ].sort_values(
        "Missing_Count",
        ascending=False,
    )

    st.subheader("Missing Values by Column")

    if missing_table.empty:

        st.success(
            "No missing values detected."
        )

    else:

        st.dataframe(
            missing_table,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Invalid Values")

    if analysis["invalid_details"]:

        invalid_table = pd.DataFrame(
            [
                {
                    "Column": col,
                    "Invalid_Count": count,
                }
                for col, count
                in analysis["invalid_details"].items()
            ]
        )

        st.dataframe(
            invalid_table,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No invalid numeric values detected."
        )


# ============================================================
# ANOMALIES
# ============================================================

elif page == "Anomalies":

    st.subheader("Anomaly Detection")

    st.caption(
        "Statistical and machine-learning anomaly screening."
    )

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "IQR Outliers",
            f"{analysis['iqr_outlier_count']:,}",
        )

        if analysis["iqr_details"]:

            iqr_table = pd.DataFrame(
                [
                    {
                        "Column": col,
                        "Outliers": details["count"],
                        "Lower Bound": round(
                            details["lower"],
                            2,
                        ),
                        "Upper Bound": round(
                            details["upper"],
                            2,
                        ),
                    }
                    for col, details
                    in analysis["iqr_details"].items()
                ]
            )

            st.dataframe(
                iqr_table,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No IQR outliers detected."
            )

    with c2:

        st.metric(
            "Isolation Forest Anomalies",
            f"{analysis['ml_anomaly_count']:,}",
        )

        st.info(
            f"Current sensitivity: "
            f"{st.session_state.contamination_pct}%"
        )

        st.caption(
            "Isolation Forest identifies unusual combinations "
            "of numeric values. These are screening signals, "
            "not automatically confirmed errors."
        )


# ============================================================
# CLEANING
# ============================================================

elif page == "Cleaning":

    st.subheader("Data Cleaning")

    st.caption(
        "Review the cleaned dataset before exporting."
    )

    original_rows = len(df)
    cleaned_rows = len(cleaned_df)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Original Rows",
            f"{original_rows:,}",
        )

    with c2:
        st.metric(
            "Cleaned Rows",
            f"{cleaned_rows:,}",
        )

    with c3:
        st.metric(
            "Rows Removed",
            f"{original_rows - cleaned_rows:,}",
        )

    st.divider()

    st.subheader("Cleaned Data Preview")

    st.dataframe(
        cleaned_df.head(25),
        use_container_width=True,
        height=500,
        hide_index=True,
    )

    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Cleaned CSV",
        data=csv_data,
        file_name="DataGuard_Cleaned_Data.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.subheader("Business Analytics")

    sales_column = analysis["sales_column"]

    if not sales_column:

        st.warning(
            "No sales/revenue column was automatically detected."
        )

    else:

        st.caption(
            f"Detected business metric: {sales_column}"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total",
                f"{business_df[sales_column].sum():,.0f}",
            )

        with c2:
            st.metric(
                "Average",
                f"{business_df[sales_column].mean():,.2f}",
            )

        with c3:
            st.metric(
                "Maximum",
                f"{business_df[sales_column].max():,.2f}",
            )

        st.divider()

        st.subheader(
            f"{sales_column} Distribution"
        )

        st.bar_chart(
            business_df[sales_column]
            .value_counts()
            .head(20)
        )

        city_columns = [
            c
            for c in business_df.columns
            if "city" in str(c).lower()
        ]

        if city_columns:

            city_col = city_columns[0]

            city_summary = (
                business_df
                .groupby(city_col)[sales_column]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(15)
            )

            st.subheader(
                "Sales by City"
            )

            st.bar_chart(
                city_summary
            )

        category_columns = [
            c
            for c in business_df.columns
            if "category" in str(c).lower()
        ]

        if category_columns:

            category_col = category_columns[0]

            category_summary = (
                business_df
                .groupby(category_col)[sales_column]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.subheader(
                "Sales by Category"
            )

            st.bar_chart(
                category_summary
            )


# ============================================================
# POWER BI
# ============================================================

elif page == "Power BI":

    st.subheader("Power BI Export Center")

    st.caption(
        "Download clean, structured datasets ready for Power BI."
    )

    exports = create_powerbi_exports(
        cleaned_df,
        business_df,
        analysis["sales_column"],
    )

    st.success(
        f"{len(exports)} Power BI-ready files generated."
    )

    for filename, content in exports.items():

        st.download_button(
            filename.replace(
                "DataGuard_",
                "",
            ),
            data=content,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )

    zip_data = create_zip(
        exports
    )

    st.divider()

    st.download_button(
        "Download Complete Power BI Package",
        data=zip_data,
        file_name="DataGuard_PowerBI_Package.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.subheader("AI Analysis")

    st.caption(
        "Gemini-generated interpretation of DataGuard findings."
    )

    report_for_ai = {
        "Quality Score": round(
            analysis["quality_score"],
            2,
        ),
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Cells": analysis["missing_count"],
        "Duplicate Rows": analysis["duplicate_count"],
        "Invalid Values": analysis["invalid_count"],
        "IQR Outliers": analysis["iqr_outlier_count"],
        "Isolation Forest Anomalies": analysis[
            "ml_anomaly_count"
        ],
        "Isolation Forest Sensitivity": (
            st.session_state.contamination_pct
        ),
        "Sales Column": analysis[
            "sales_column"
        ],
    }

    st.json(report_for_ai)

    if st.button(
        "Generate Gemini Analysis",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Gemini is analyzing the DataGuard report..."
        ):

            result = generate_gemini_analysis(
                report_for_ai
            )

            st.session_state.gemini_analysis = result

    if st.session_state.gemini_analysis:

        st.divider()

        st.subheader(
            "AI Assessment"
        )

        st.markdown(
            st.session_state.gemini_analysis
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.subheader("DataGuard Report")

    st.caption(
        "Complete dataset health summary."
    )

    report_text = f"""
DATAGUARD AI
AI DATA QUALITY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset
-------
File: {st.session_state.file_name}
Rows: {len(df):,}
Columns: {len(df.columns):,}

Data Quality
------------
Quality Score: {analysis["quality_score"]:.2f}/100
Missing Cells: {analysis["missing_count"]:,}
Duplicate Rows: {analysis["duplicate_count"]:,}
Invalid Values: {analysis["invalid_count"]:,}

Anomaly Detection
-----------------
IQR Outliers: {analysis["iqr_outlier_count"]:,}
ML Anomalies: {analysis["ml_anomaly_count"]:,}
Isolation Forest Sensitivity:
{st.session_state.contamination_pct}%

Business
--------
Sales Column: {analysis["sales_column"]}

DataGuard AI
AI Data Quality & Anomaly Detection Platform
"""

    st.text_area(
        "Report Preview",
        report_text,
        height=500,
    )

    st.download_button(
        "Download DataGuard Report",
        data=report_text,
        file_name="DataGuard_AI_Report.txt",
        mime="text/plain",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dg-footer">
        🛡️ <strong>DataGuard AI</strong>
        &nbsp;•&nbsp;
        AI Data Quality & Anomaly Detection Platform
        <br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
