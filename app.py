import io
import os
import zipfile
import re

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

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #080D18;
        color: #F5F7FA;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #0B1120;
        border-right: 1px solid #202A40;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    /* ---------- TEXT ---------- */

    .dg-title {
        font-size: 30px;
        font-weight: 800;
        color: #F5F7FA;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    .dg-subtitle {
        font-size: 14px;
        color: #8E9AAF;
        margin-bottom: 25px;
    }

    .dg-section {
        font-size: 18px;
        font-weight: 750;
        color: #F5F7FA;
        margin-top: 22px;
        margin-bottom: 8px;
    }

    .dg-muted {
        font-size: 13px;
        color: #8995AA;
    }

    /* ---------- CARDS ---------- */

    .dg-card {
        background: #11192B;
        border: 1px solid #222D45;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .dg-card-title {
        font-size: 15px;
        font-weight: 700;
        color: #F5F7FA;
        margin-bottom: 5px;
    }

    .dg-card-subtitle {
        font-size: 12px;
        color: #8995AA;
        margin-bottom: 15px;
    }

    /* ---------- KPI ---------- */

    .dg-kpi {
        background: #11192B;
        border: 1px solid #222D45;
        border-radius: 14px;
        padding: 17px;
        min-height: 105px;
    }

    .dg-kpi-label {
        font-size: 12px;
        color: #8995AA;
        margin-bottom: 8px;
    }

    .dg-kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #F5F7FA;
    }

    /* ---------- SIDEBAR BRAND ---------- */

    .brand-box {
        padding: 5px 5px 22px 5px;
    }

    .brand-icon {
        font-size: 30px;
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        color: #F5F7FA;
    }

    .brand-sub {
        font-size: 11px;
        color: #7F8BA0;
        margin-top: 2px;
    }

    /* ---------- STATUS ---------- */

    .status-box {
        background: #10192A;
        border: 1px solid #24314B;
        border-radius: 12px;
        padding: 12px;
        margin-top: 15px;
    }

    .status-dot {
        color: #48D597;
        font-weight: 700;
    }

    /* ---------- WORKFLOW ---------- */

    .workflow {
        display: flex;
        gap: 6px;
        align-items: center;
        overflow-x: auto;
        padding: 10px 0 20px 0;
    }

    .workflow-item {
        white-space: nowrap;
        padding: 9px 13px;
        border-radius: 9px;
        font-size: 12px;
        font-weight: 650;
        background: #11192B;
        border: 1px solid #202A40;
        color: #707C91;
    }

    .workflow-complete {
        color: #64DCA5;
        border-color: #23523F;
    }

    .workflow-active {
        color: #F5F7FA;
        background: #17233A;
        border-color: #496A9F;
    }

    .workflow-arrow {
        color: #536078;
    }

    /* ---------- UPLOAD ---------- */

    [data-testid="stFileUploader"] {
        background: #10182A;
        border: 1px dashed #40506D;
        border-radius: 15px;
        padding: 8px;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 9px;
        font-weight: 650;
        border: 1px solid #2C3B57;
    }

    /* ---------- TABLE ---------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #222D45;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- FOOTER ---------- */

    .dg-footer {
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #202A40;
        color: #68748A;
        font-size: 12px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "df": None,
    "cleaned_df": None,
    "business_df": None,
    "chart_df": None,
    "file_name": None,
    "analysis": None,
    "gemini_analysis": None,
    "contamination_pct": 5,
    "page": "Dashboard",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def page_title(title, subtitle=""):
    st.markdown(
        f"""
        <div class="dg-title">{title}</div>
        <div class="dg-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def card(title, subtitle=""):
    html = f"""
    <div class="dg-card">
        <div class="dg-card-title">{title}</div>
    """

    if subtitle:
        html += f"""
        <div class="dg-card-subtitle">{subtitle}</div>
        """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def close_card():
    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="dg-kpi">
            <div class="dg-kpi-label">{label}</div>
            <div class="dg-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(value):
    if value is None:
        return "Not available"

    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"

    if isinstance(value, (int, np.integer)):
        return f"{value:,}"

    return str(value)


# ============================================================
# DATA HELPERS
# ============================================================

def detect_datetime_columns(df):

    datetime_cols = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        if df[col].dtype == "object":

            sample = df[col].dropna().astype(str).head(300)

            if len(sample) == 0:
                continue

            parsed = pd.to_datetime(
                sample,
                errors="coerce"
            )

            if parsed.notna().mean() >= 0.75:
                datetime_cols.append(col)

    return datetime_cols


def detect_invalid_values(df):

    invalid = 0

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        series = df[col]

        invalid += int(
            np.isinf(series).sum()
        )

        invalid += int(
            (series < 0).sum()
            if col.lower() in [
                "quantity",
                "sales",
                "amount",
                "revenue",
                "price",
                "profit",
            ]
            else 0
        )

    return invalid


def detect_iqr_outliers(df):

    total = 0

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        series = pd.to_numeric(
            df[col],
            errors="coerce"
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

        total += int(
            ((series < lower) | (series > upper)).sum()
        )

    return total


def detect_ml_anomalies(df, contamination_pct):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty or len(df) < 10:
        return 0, len(df)

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.fillna(
        numeric.median(numeric_only=True)
    )

    numeric = numeric.fillna(0)

    contamination = contamination_pct / 100

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )

    predictions = model.fit_predict(numeric)

    anomalies = int(
        (predictions == -1).sum()
    )

    normal = int(
        (predictions == 1).sum()
    )

    return anomalies, normal


def calculate_quality_score(
    rows,
    columns,
    missing,
    duplicates,
    invalid
):

    if rows == 0 or columns == 0:
        return 0

    missing_rate = missing / (rows * columns)
    duplicate_rate = duplicates / rows
    invalid_rate = invalid / rows

    score = (
        100
        - missing_rate * 50
        - duplicate_rate * 30
        - invalid_rate * 20
    )

    return max(
        0,
        min(100, score)
    )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_dataset(df, contamination_pct):

    rows = len(df)
    columns = len(df.columns)

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    invalid = detect_invalid_values(df)

    iqr_outliers = detect_iqr_outliers(df)

    ml_anomalies, normal_records = detect_ml_anomalies(
        df,
        contamination_pct
    )

    quality_score = calculate_quality_score(
        rows,
        columns,
        missing,
        duplicates,
        invalid
    )

    numeric_cols = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_cols = list(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    datetime_cols = detect_datetime_columns(df)

    anomaly_rate = (
        ml_anomalies / rows * 100
        if rows > 0
        else 0
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing_values": missing,
        "duplicates": duplicates,
        "invalid_values": invalid,
        "iqr_outliers": iqr_outliers,
        "ml_anomalies": ml_anomalies,
        "normal_records": normal_records,
        "anomaly_rate": anomaly_rate,
        "quality_score": quality_score,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "numeric_names": numeric_cols,
        "categorical_names": categorical_cols,
        "datetime_names": datetime_cols,
    }


# ============================================================
# CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    rows_before = len(cleaned)

    duplicates_removed = int(
        cleaned.duplicated().sum()
    )

    cleaned = cleaned.drop_duplicates()

    values_filled = 0

    numeric_cols = cleaned.select_dtypes(
        include=np.number
    ).columns

    categorical_cols = cleaned.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in numeric_cols:

        missing_before = int(
            cleaned[col].isna().sum()
        )

        if missing_before > 0:

            median_value = cleaned[col].median()

            cleaned[col] = cleaned[col].fillna(
                median_value
            )

            values_filled += missing_before

    for col in categorical_cols:

        missing_before = int(
            cleaned[col].isna().sum()
        )

        if missing_before > 0:

            mode = cleaned[col].mode()

            if len(mode) > 0:

                cleaned[col] = cleaned[col].fillna(
                    mode.iloc[0]
                )

                values_filled += missing_before

    return (
        cleaned,
        {
            "rows_before": rows_before,
            "rows_after": len(cleaned),
            "rows_removed": rows_before - len(cleaned),
            "duplicates_removed": duplicates_removed,
            "values_filled": values_filled,
        },
    )


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

def find_sales_column(df):

    candidates = [
        "Sales",
        "sales",
        "Revenue",
        "revenue",
        "Amount",
        "amount",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    return None


def prepare_business_data(df):

    sales_col = find_sales_column(df)

    if sales_col is None:
        return None

    business = df.copy()

    business[sales_col] = pd.to_numeric(
        business[sales_col],
        errors="coerce"
    )

    business = business.dropna(
        subset=[sales_col]
    )

    return business


# ============================================================
# POWER BI EXPORTS
# ============================================================

def create_powerbi_exports(df):

    files = {}

    cleaned_bytes = io.BytesIO()

    df.to_csv(
        cleaned_bytes,
        index=False
    )

    files["DataGuard_Cleaned_Data.csv"] = (
        cleaned_bytes.getvalue()
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

    profile_bytes = io.BytesIO()

    profile.to_csv(
        profile_bytes,
        index=False
    )

    files["DataGuard_Profile.csv"] = (
        profile_bytes.getvalue()
    )

    sales_col = find_sales_column(df)

    if sales_col:

        summary = pd.DataFrame({
            "Metric": [
                "Total Sales",
                "Average Sale",
                "Highest Sale",
                "Records",
            ],
            "Value": [
                df[sales_col].sum(),
                df[sales_col].mean(),
                df[sales_col].max(),
                len(df),
            ],
        })

        summary_bytes = io.BytesIO()

        summary.to_csv(
            summary_bytes,
            index=False
        )

        files["DataGuard_Sales_Summary.csv"] = (
            summary_bytes.getvalue()
        )

    return files


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
# GEMINI
# ============================================================

def run_gemini(df, analysis, cleaning=None):

    try:

        from google import genai

    except Exception:

        return (
            "Gemini SDK is not installed. "
            "Install the Google GenAI package first."
        )

    api_key = None

    try:
        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        pass

    if not api_key:
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:

        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit secrets."
        )

    sales_col = find_sales_column(df)

    total_sales = None
    average_sale = None
    highest_sale = None
    upper_bound = None

    if sales_col:

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).dropna()

        if len(sales):

            total_sales = float(
                sales.sum()
            )

            average_sale = float(
                sales.mean()
            )

            highest_sale = float(
                sales.max()
            )

            q1 = sales.quantile(0.25)
            q3 = sales.quantile(0.75)

            iqr = q3 - q1

            upper_bound = float(
                q3 + 1.5 * iqr
            )

    cleaning_text = "Not performed"

    if cleaning:

        cleaning_text = f"""
Rows before: {cleaning['rows_before']:,}
Rows after: {cleaning['rows_after']:,}
Rows removed: {cleaning['rows_removed']:,}
Values filled: {cleaning['values_filled']:,}
"""

    prompt = f"""
You are the AI analysis engine inside DataGuard AI,
a professional data quality platform.

Analyze ONLY the measured values provided below.

IMPORTANT:
- Do not recalculate numbers.
- Do not estimate numbers.
- Do not invent numbers.
- Do not replace a provided number with another number.
- Use the exact values provided.
- Clearly distinguish statistical outliers from ML anomaly signals.
- Isolation Forest anomalies are screening signals, not confirmed errors.

DATASET:
Rows: {analysis['rows']}
Columns: {analysis['columns']}

QUALITY:
Quality score: {analysis['quality_score']:.2f}%
Missing values: {analysis['missing_values']}
Duplicates: {analysis['duplicates']}
Invalid values: {analysis['invalid_values']}
IQR outliers: {analysis['iqr_outliers']}

MACHINE LEARNING:
Normal records: {analysis['normal_records']}
ML anomalies: {analysis['ml_anomalies']}
Anomaly rate: {analysis['anomaly_rate']:.2f}%
Algorithm: Isolation Forest
Sensitivity: {st.session_state.contamination_pct}%

DATA TYPES:
Numeric columns: {analysis['numeric_columns']}
Categorical columns: {analysis['categorical_columns']}
Datetime columns: {analysis['datetime_columns']}

BUSINESS METRICS:
Sales column: {sales_col}
Total sales: {total_sales}
Average sale: {average_sale}
Highest sale: {highest_sale}
IQR upper bound: {upper_bound}

CLEANING:
{cleaning_text}

Return exactly these sections:

## Overall Assessment

## Confirmed Data Quality Problems

## Statistical Outlier Findings

## ML Anomaly Findings

## Possible Root Causes

## Cleaning Results

## Business Impact

## Power BI Recommendations

Keep the explanation professional and concise.
"""

    try:

        client = genai.Client(
            api_key=api_key
        )

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

    except Exception as e:

        return f"Gemini analysis failed: {e}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-icon">🛡️</div>
            <div class="brand-name">DataGuard AI</div>
            <div class="brand-sub">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "**WORKSPACE**"
    )

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

    for item in pages:

        if st.button(
            item,
            key=f"nav_{item}",
            use_container_width=True
        ):

            st.session_state.page = item
            st.rerun()

    st.divider()

    st.markdown(
        "**DETECTION SETTINGS**"
    )

    st.session_state.contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=int(
            st.session_state.contamination_pct
        ),
        help=(
            "Higher sensitivity flags more "
            "records as unusual."
        ),
    )

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.markdown(
        """
        <div class="status-box">
            <div style="font-size:11px;color:#77839A;">
                SYSTEM
            </div>
            <div style="margin-top:5px;">
                <span class="status-dot">●</span>
                All systems operational
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

if st.session_state.df is not None:

    st.session_state.analysis = analyze_dataset(
        st.session_state.df,
        st.session_state.contamination_pct
    )


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.markdown(
        """
        <div style="
            font-size:13px;
            color:#7F8BA0;
            margin-bottom:3px;
        ">
            Data Intelligence Workspace
        </div>

        <div style="
            font-size:13px;
            color:#9AA5B8;
        ">
            AI-powered data quality, anomaly detection
            and business analytics.
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    if st.session_state.file_name:

        st.markdown(
            f"""
            <div style="
                text-align:right;
                font-size:12px;
                color:#AAB4C5;
                padding-top:10px;
            ">
                📄 {st.session_state.file_name}
            </div>
            """,
            unsafe_allow_html=True
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

page_to_step = {
    "Dashboard": 3,
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

current_step = page_to_step.get(
    st.session_state.page,
    1
)

workflow_html = '<div class="workflow">'

for i, step in enumerate(
    workflow,
    start=1
):

    if i < current_step:
        cls = "workflow-item workflow-complete"
        symbol = "✓"

    elif i == current_step:
        cls = "workflow-item workflow-active"
        symbol = f"{i:02d}"

    else:
        cls = "workflow-item"
        symbol = f"{i:02d}"

    workflow_html += (
        f'<div class="{cls}">'
        f'{symbol} {step}'
        f'</div>'
    )

    if i < len(workflow):

        workflow_html += (
            '<div class="workflow-arrow">→</div>'
        )

workflow_html += "</div>"

st.markdown(
    workflow_html,
    unsafe_allow_html=True
)


# ============================================================
# NO DATA STATE
# ============================================================

if st.session_state.df is None:

    page_title(
        "Welcome to DataGuard AI",
        "Upload a dataset to begin your data intelligence workflow."
    )

    card(
        "Upload your dataset",
        "CSV, XLSX or XLS files • Maximum 50 MB"
    )

    uploaded = st.file_uploader(
        "Upload CSV, XLSX or XLS",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    close_card()

    if uploaded:

        size_mb = uploaded.size / (
            1024 * 1024
        )

        if size_mb > 50:

            st.error(
                "File exceeds the 50 MB limit."
            )

        else:

            try:

                if uploaded.name.lower().endswith(
                    ".csv"
                ):

                    df = pd.read_csv(
                        uploaded
                    )

                else:

                    df = pd.read_excel(
                        uploaded
                    )

                st.session_state.df = df
                st.session_state.file_name = (
                    uploaded.name
                )

                st.session_state.analysis = (
                    analyze_dataset(
                        df,
                        st.session_state.contamination_pct
                    )
                )

                st.session_state.page = (
                    "Data Profile"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to read file: {e}"
                )


# ============================================================
# DATA EXISTS
# ============================================================

else:

    df = st.session_state.df
    analysis = st.session_state.analysis


    # ========================================================
    # DASHBOARD
    # ========================================================

    if st.session_state.page == "Dashboard":

        page_title(
            "Data Quality Overview",
            "Monitor, analyze, and improve the quality of your dataset."
        )

        cols = st.columns(6)

        values = [
            ("Data Quality", f"{analysis['quality_score']:.2f}%"),
            ("Total Records", f"{analysis['rows']:,}"),
            ("Columns", f"{analysis['columns']:,}"),
            ("Missing Values", f"{analysis['missing_values']:,}"),
            ("Duplicates", f"{analysis['duplicates']:,}"),
            ("Anomalies", f"{analysis['ml_anomalies']:,}"),
        ]

        for col, (label, value) in zip(
            cols,
            values
        ):

            with col:
                metric_card(label, value)

        st.markdown(
            '<div style="height:12px;"></div>',
            unsafe_allow_html=True
        )

        left, right = st.columns(
            [1, 1.4]
        )

        with left:

            card(
                "Overall Data Quality Score",
                "Measured quality score from the uploaded dataset."
            )

            st.markdown(
                f"""
                <div style="
                    font-size:42px;
                    font-weight:850;
                    text-align:center;
                    color:#F5F7FA;
                    padding:10px;
                ">
                    {analysis['quality_score']:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                analysis["quality_score"] / 100
            )

            close_card()

        with right:

            card(
                "Issues Detected",
                "Measured signals from the uploaded dataset."
            )

            issue_cols = st.columns(5)

            issues = [
                ("Missing", analysis["missing_values"]),
                ("Duplicates", analysis["duplicates"]),
                ("Outliers", analysis["iqr_outliers"]),
                ("Invalid", analysis["invalid_values"]),
                ("ML Anomalies", analysis["ml_anomalies"]),
            ]

            for col, (label, value) in zip(
                issue_cols,
                issues
            ):

                with col:
                    metric_card(
                        label,
                        f"{value:,}"
                    )

            close_card()

        card(
            "Anomaly Detection",
            "Isolation Forest screening results."
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            metric_card(
                "Normal Records",
                f"{analysis['normal_records']:,}"
            )

        with a2:
            metric_card(
                "Anomalies",
                f"{analysis['ml_anomalies']:,}"
            )

        with a3:
            metric_card(
                "Anomaly Rate",
                f"{analysis['anomaly_rate']:.2f}%"
            )

        with a4:
            metric_card(
                "Algorithm",
                "Isolation Forest"
            )

        st.caption(
            "ML anomalies are screening signals, not confirmed errors."
        )

        close_card()


    # ========================================================
    # UPLOAD
    # ========================================================

    elif st.session_state.page == "Upload Data":

        page_title(
            "Upload Data",
            "Upload CSV or Excel data to begin the DataGuard workflow."
        )

        card(
            "Upload your dataset",
            "CSV, XLSX or XLS files • Maximum 50 MB"
        )

        uploaded = st.file_uploader(
            "Choose a dataset",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

        close_card()

        if uploaded:

            size_mb = uploaded.size / (
                1024 * 1024
            )

            if size_mb > 50:

                st.error(
                    "File exceeds the 50 MB maximum."
                )

            else:

                try:

                    if uploaded.name.lower().endswith(
                        ".csv"
                    ):

                        new_df = pd.read_csv(
                            uploaded
                        )

                    else:

                        new_df = pd.read_excel(
                            uploaded
                        )

                    st.session_state.df = new_df
                    st.session_state.file_name = (
                        uploaded.name
                    )
                    st.session_state.cleaned_df = None
                    st.session_state.business_df = None
                    st.session_state.gemini_analysis = None

                    st.session_state.analysis = (
                        analyze_dataset(
                            new_df,
                            st.session_state.contamination_pct
                        )
                    )

                    st.success(
                        f"{uploaded.name} uploaded successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Unable to read file: {e}"
                    )

        card(
            "Current Dataset"
        )

        st.markdown(
            f"""
            <div style="
                font-size:15px;
                color:#D4DAE4;
            ">
                📄 {st.session_state.file_name}
            </div>

            <div style="
                font-size:12px;
                color:#7F8BA0;
                margin-top:8px;
            ">
                {len(df):,} rows • {len(df.columns):,} columns
            </div>
            """,
            unsafe_allow_html=True
        )

        close_card()


    # ========================================================
    # DATA PROFILE
    # ========================================================

    elif st.session_state.page == "Data Profile":

        page_title(
            "Data Profile",
            "Understand the structure, types and completeness of your dataset."
        )

        profile = pd.DataFrame({
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
                int(df[c].nunique())
                for c in df.columns
            ],
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # QUALITY CHECKS
    # ========================================================

    elif st.session_state.page == "Quality Checks":

        page_title(
            "Quality Checks",
            "Measure completeness, consistency, validity and uniqueness."
        )

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            metric_card(
                "Completeness",
                f"{100 - (analysis['missing_values'] / max(1, analysis['rows'] * analysis['columns']) * 100):.2f}%"
            )

        with q2:
            metric_card(
                "Duplicates",
                f"{analysis['duplicates']:,}"
            )

        with q3:
            metric_card(
                "Invalid Values",
                f"{analysis['invalid_values']:,}"
            )

        with q4:
            metric_card(
                "IQR Outliers",
                f"{analysis['iqr_outliers']:,}"
            )

        st.write("")

        quality_table = pd.DataFrame({
            "Check": [
                "Completeness",
                "Consistency",
                "Validity",
                "Uniqueness",
                "Outliers",
            ],
            "Result": [
                analysis["missing_values"] == 0,
                analysis["duplicates"] == 0,
                analysis["invalid_values"] == 0,
                analysis["duplicates"] == 0,
                analysis["iqr_outliers"] == 0,
            ],
            "Count": [
                analysis["missing_values"],
                analysis["duplicates"],
                analysis["invalid_values"],
                analysis["duplicates"],
                analysis["iqr_outliers"],
            ],
        })

        st.dataframe(
            quality_table,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    elif st.session_state.page == "Anomaly Detection":

        page_title(
            "Anomaly Detection",
            "Identify unusual records using Isolation Forest."
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            metric_card(
                "Normal Records",
                f"{analysis['normal_records']:,}"
            )

        with a2:
            metric_card(
                "Anomalies",
                f"{analysis['ml_anomalies']:,}"
            )

        with a3:
            metric_card(
                "Anomaly Rate",
                f"{analysis['anomaly_rate']:.2f}%"
            )

        with a4:
            metric_card(
                "Sensitivity",
                f"{st.session_state.contamination_pct}%"
            )

        st.write("")

        numeric_cols = analysis[
            "numeric_names"
        ]

        if len(numeric_cols) >= 2:

            x_col = st.selectbox(
                "X-axis",
                numeric_cols,
                index=0
            )

            y_col = st.selectbox(
                "Y-axis",
                numeric_cols,
                index=min(
                    1,
                    len(numeric_cols) - 1
                )
            )

            chart_data = df[
                [x_col, y_col]
            ].copy()

            chart_data[x_col] = pd.to_numeric(
                chart_data[x_col],
                errors="coerce"
            )

            chart_data[y_col] = pd.to_numeric(
                chart_data[y_col],
                errors="coerce"
            )

            chart_data = chart_data.dropna()

            st.scatter_chart(
                chart_data,
                x=x_col,
                y=y_col,
                use_container_width=True
            )

        else:

            st.info(
                "At least two numeric columns are required for anomaly visualization."
            )

        st.caption(
            "Important: Isolation Forest identifies unusual patterns. "
            "An anomaly is not automatically a data error."
        )


    # ========================================================
    # CLEANING
    # ========================================================

    elif st.session_state.page == "Data Cleaning":

        page_title(
            "Data Cleaning",
            "Review and apply safe automated data-cleaning operations."
        )

        if st.session_state.cleaned_df is None:

            st.info(
                "Recommended cleaning includes duplicate removal "
                "and missing-value treatment."
            )

            if st.button(
                "Run Cleaning",
                type="primary"
            ):

                cleaned, cleaning_info = clean_dataset(
                    df
                )

                st.session_state.cleaned_df = (
                    cleaned
                )

                st.session_state.cleaning_info = (
                    cleaning_info
                )

                st.rerun()

        else:

            cleaned = st.session_state.cleaned_df
            info = st.session_state.cleaning_info

            b1, b2, b3, b4 = st.columns(4)

            with b1:
                metric_card(
                    "Rows Before",
                    f"{info['rows_before']:,}"
                )

            with b2:
                metric_card(
                    "Rows After",
                    f"{info['rows_after']:,}"
                )

            with b3:
                metric_card(
                    "Rows Removed",
                    f"{info['rows_removed']:,}"
                )

            with b4:
                metric_card(
                    "Values Filled",
                    f"{info['values_filled']:,}"
                )

            st.write("")

            st.markdown(
                "**Cleaned Dataset Preview**"
            )

            st.dataframe(
                cleaned.head(20),
                use_container_width=True,
                hide_index=True
            )

            csv = cleaned.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Cleaned Dataset",
                data=csv,
                file_name="DataGuard_Cleaned_Data.csv",
                mime="text/csv"
            )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif st.session_state.page == "Analytics":

        page_title(
            "Business Analytics",
            "Explore business trends and sales performance."
        )

        business = prepare_business_data(
            df
        )

        if business is None:

            st.warning(
                "No Sales, Revenue or Amount column was detected."
            )

        else:

            sales_col = find_sales_column(
                business
            )

            total_sales = business[
                sales_col
            ].sum()

            average_sale = business[
                sales_col
            ].mean()

            highest_sale = business[
                sales_col
            ].max()

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card(
                    "Total Sales",
                    f"{total_sales:,.2f}"
                )

            with c2:
                metric_card(
                    "Average Sale",
                    f"{average_sale:,.2f}"
                )

            with c3:
                metric_card(
                    "Highest Sale",
                    f"{highest_sale:,.2f}"
                )

            st.write("")

            st.markdown(
                "**Sales Distribution**"
            )

            st.bar_chart(
                business[
                    [sales_col]
                ].head(100),
                use_container_width=True
            )


    # ========================================================
    # POWER BI
    # ========================================================

    elif st.session_state.page == "Power BI":

        page_title(
            "Power BI",
            "Export clean, analysis-ready datasets for Power BI."
        )

        source_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        files = create_powerbi_exports(
            source_df
        )

        zip_data = create_zip(
            files
        )

        card(
            "Power BI Export Package",
            "Download the prepared data and supporting tables."
        )

        st.download_button(
            "Download Power BI Package",
            data=zip_data,
            file_name="DataGuard_PowerBI_Package.zip",
            mime="application/zip",
            type="primary"
        )

        close_card()

        st.markdown(
            "**Included files**"
        )

        for filename in files:

            st.write(
                f"✓ {filename}"
            )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif st.session_state.page == "AI Analysis":

        page_title(
            "AI Analysis",
            "Use Gemini to explain measured data-quality and business findings."
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            metric_card(
                "Quality",
                f"{analysis['quality_score']:.2f}%"
            )

        with a2:
            metric_card(
                "Records",
                f"{analysis['rows']:,}"
            )

        with a3:
            metric_card(
                "ML Anomalies",
                f"{analysis['ml_anomalies']:,}"
            )

        with a4:
            metric_card(
                "Missing",
                f"{analysis['missing_values']:,}"
            )

        st.write("")

        if st.button(
            "Generate Gemini Analysis",
            type="primary"
        ):

            cleaning_info = getattr(
                st.session_state,
                "cleaning_info",
                None
            )

            with st.spinner(
                "Gemini is analyzing the measured results..."
            ):

                result = run_gemini(
                    df,
                    analysis,
                    cleaning_info
                )

            if isinstance(result, str):

                st.session_state.gemini_analysis = (
                    result
                )

            else:

                st.session_state.gemini_analysis = (
                    str(result)
                )

        if st.session_state.gemini_analysis:

            st.markdown(
                """
                <div class="dg-card">
                    <div style="
                        font-size:19px;
                        font-weight:800;
                        color:#F5F7FA;
                    ">
                        DataGuard AI Report
                    </div>

                    <div style="
                        font-size:12px;
                        color:#8995AA;
                        margin-top:4px;
                    ">
                        AI-generated interpretation of measured dataset signals.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            text = st.session_state.gemini_analysis

            # Render headings without Markdown # headings.
            lines = text.splitlines()

            current_content = []

            for line in lines:

                match = re.match(
                    r"^\s*#{1,6}\s*(.+?)\s*$",
                    line
                )

                if match:

                    if current_content:

                        st.markdown(
                            "\n".join(
                                current_content
                            ),
                            unsafe_allow_html=False
                        )

                        current_content = []

                    heading = match.group(1).strip()

                    st.markdown(
                        f"""
                        <div style="
                            font-size:17px;
                            font-weight:800;
                            color:#F5F7FA;
                            margin-top:25px;
                            margin-bottom:8px;
                        ">
                            {heading}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif line.strip():

                    current_content.append(
                        line
                    )

            if current_content:

                st.markdown(
                    "\n".join(
                        current_content
                    ),
                    unsafe_allow_html=False
                )

        else:

            st.info(
                "Click 'Generate Gemini Analysis' to create the AI report."
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif st.session_state.page == "Reports":

        page_title(
            "Reports",
            "Download DataGuard AI results and cleaned datasets."
        )

        source_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        report = pd.DataFrame({
            "Metric": [
                "Data Quality Score",
                "Total Records",
                "Columns",
                "Missing Values",
                "Duplicates",
                "Invalid Values",
                "IQR Outliers",
                "ML Anomalies",
                "Anomaly Rate",
            ],
            "Value": [
                f"{analysis['quality_score']:.2f}%",
                analysis["rows"],
                analysis["columns"],
                analysis["missing_values"],
                analysis["duplicates"],
                analysis["invalid_values"],
                analysis["iqr_outliers"],
                analysis["ml_anomalies"],
                f"{analysis['anomaly_rate']:.2f}%",
            ],
        })

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )

        report_csv = report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Quality Report",
            data=report_csv,
            file_name="DataGuard_Quality_Report.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dg-footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br>
        Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
