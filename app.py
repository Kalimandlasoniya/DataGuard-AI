import io
import os
import re
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

DEFAULTS = {
    "df": None,
    "cleaned_df": None,
    "business_df": None,
    "chart_df": None,
    "file_name": None,
    "analysis_complete": False,
    "analysis": None,
    "gemini_analysis": None,
    "contamination_pct": 5,
    "page": "Dashboard",
    "cleaning_result": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #070b12;
    --panel: #0d131d;
    --panel2: #111925;
    --border: #1e2a39;
    --text: #f4f7fb;
    --muted: #8d9aad;
    --green: #36d399;
    --blue: #5b8cff;
    --purple: #9b7cff;
    --orange: #ffb454;
    --red: #ff6b7a;
}

html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 0%, rgba(91,140,255,.08), transparent 28%),
        radial-gradient(circle at 10% 20%, rgba(54,211,153,.04), transparent 25%),
        var(--bg);
    color: var(--text);
}

/* Hide Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #090e16;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.3rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}

.brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(
        135deg,
        rgba(91,140,255,.25),
        rgba(54,211,153,.18)
    );
    border: 1px solid rgba(91,140,255,.25);
    font-size: 20px;
}

.brand-name {
    font-size: 18px;
    font-weight: 700;
    color: white;
}

.brand-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 1px;
}

.sidebar-label {
    color: #647287;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.3px;
    margin-top: 28px;
    margin-bottom: 9px;
}

.system-box {
    margin-top: 30px;
    padding: 13px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: rgba(255,255,255,.02);
}

.system-title {
    font-size: 10px;
    color: #647287;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 7px;
}

.system-status {
    color: var(--green);
    font-size: 12px;
}

/* Streamlit buttons */

.stButton > button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid transparent;
    background: transparent;
    color: #9ba8ba;
    text-align: left;
    padding: 9px 12px;
    font-size: 13px;
}

.stButton > button:hover {
    border-color: var(--border);
    background: rgba(255,255,255,.035);
    color: white;
}

.stButton > button[kind="primary"] {
    background: rgba(91,140,255,.13);
    border-color: rgba(91,140,255,.3);
    color: white;
}

/* Main content */

.main-container {
    max-width: 1500px;
    margin: auto;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3px 0 22px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 25px;
}

.page-kicker {
    color: #69778b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}

.page-title {
    color: white;
    font-size: 25px;
    font-weight: 750;
}

.page-description {
    color: var(--muted);
    font-size: 13px;
    margin-top: 3px;
}

/* Cards */

.card {
    background: linear-gradient(
        145deg,
        rgba(17,25,37,.96),
        rgba(11,17,27,.96)
    );
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 18px;
}

.card-title {
    color: white;
    font-size: 15px;
    font-weight: 650;
    margin-bottom: 5px;
}

.card-description {
    color: var(--muted);
    font-size: 12px;
}

.kpi-card {
    background: linear-gradient(
        145deg,
        rgba(17,25,37,.98),
        rgba(10,15,23,.98)
    );
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px;
    min-height: 105px;
}

.kpi-label {
    color: #8290a4;
    font-size: 11px;
    margin-bottom: 8px;
}

.kpi-value {
    color: white;
    font-size: 25px;
    font-weight: 750;
}

.kpi-sub {
    color: #657286;
    font-size: 10px;
    margin-top: 4px;
}

/* Upload */

.upload-box {
    border: 1px dashed #334155;
    border-radius: 15px;
    padding: 42px 25px;
    text-align: center;
    background:
        radial-gradient(circle at center, rgba(91,140,255,.08), transparent 55%),
        rgba(255,255,255,.015);
}

.upload-icon {
    font-size: 34px;
    margin-bottom: 10px;
}

.upload-title {
    color: white;
    font-size: 18px;
    font-weight: 650;
}

.upload-text {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

/* Workflow */

.workflow {
    display: flex;
    align-items: center;
    gap: 7px;
    overflow-x: auto;
    padding: 3px 0 18px 0;
}

.workflow-step {
    min-width: 88px;
    padding: 8px 9px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: rgba(255,255,255,.015);
    text-align: center;
}

.workflow-step.done {
    border-color: rgba(54,211,153,.35);
    background: rgba(54,211,153,.07);
}

.workflow-step.active {
    border-color: rgba(91,140,255,.5);
    background: rgba(91,140,255,.1);
}

.workflow-number {
    font-size: 9px;
    color: #657286;
}

.workflow-name {
    color: #a9b4c3;
    font-size: 10px;
    margin-top: 3px;
}

.workflow-step.done .workflow-name {
    color: var(--green);
}

.workflow-step.active .workflow-name {
    color: #9ebaff;
}

.workflow-arrow {
    color: #384557;
    font-size: 14px;
}

/* Score */

.score-ring {
    width: 145px;
    height: 145px;
    border-radius: 50%;
    margin: 10px auto 20px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        radial-gradient(circle, #0d131d 58%, transparent 59%),
        conic-gradient(
            var(--green) var(--score),
            #1b2532 0
        );
}

.score-value {
    color: white;
    font-size: 31px;
    font-weight: 800;
}

.score-caption {
    text-align: center;
    color: #7f8da1;
    font-size: 11px;
}

/* Status */

.status-good {
    color: var(--green);
}

.status-warning {
    color: var(--orange);
}

.status-danger {
    color: var(--red);
}

/* Tables */

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* Metrics */

[data-testid="stMetric"] {
    background: rgba(255,255,255,.02);
    border: 1px solid var(--border);
    padding: 13px;
    border-radius: 12px;
}

/* Inputs */

.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stMultiSelect div {
    background: #0c131e !important;
    color: white !important;
    border-color: var(--border) !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    margin-top: -8px;
}

[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 0 !important;
}

/* Expander */

.streamlit-expanderHeader {
    background: #0d131d;
    border: 1px solid var(--border);
    border-radius: 10px;
}

/* Footer */

.footer {
    text-align: center;
    color: #566274;
    font-size: 10px;
    padding: 30px 0 15px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.Series(pd.NaT, index=series.index)


def detect_datetime_columns(df):
    result = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.append(col)
            continue

        if df[col].dtype == "object":
            converted = safe_to_datetime(df[col])

            if converted.notna().mean() >= 0.80:
                result.append(col)

    return result


def detect_identifier_columns(df):
    identifiers = []

    for col in df.columns:
        unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

        if unique_ratio >= 0.95:
            identifiers.append(col)

    return identifiers


def detect_invalid_values(df):
    invalid = 0

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric = pd.to_numeric(df[col], errors="coerce")

            invalid += int(
                ((numeric.isna()) & (df[col].notna())).sum()
            )

    return invalid


def detect_iqr_outliers(df):
    total = 0
    details = {}

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)
        count = int(mask.sum())

        if count > 0:
            details[col] = {
                "count": count,
                "lower": float(lower),
                "upper": float(upper),
            }

        total += count

    return total, details


def detect_ml_anomalies(df, contamination_pct):
    numeric = df.select_dtypes(include=np.number).copy()

    if numeric.shape[1] == 0:
        return 0, 0, pd.Series(False, index=df.index)

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.fillna(numeric.median(numeric_only=True))

    numeric = numeric.select_dtypes(include=np.number)

    if numeric.shape[1] == 0 or len(numeric) < 10:
        return len(df), 0, pd.Series(False, index=df.index)

    contamination = max(
        0.001,
        min(0.25, contamination_pct / 100)
    )

    try:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=150,
            n_jobs=-1
        )

        prediction = model.fit_predict(numeric)

        anomaly_mask = pd.Series(
            prediction == -1,
            index=df.index
        )

        anomalies = int(anomaly_mask.sum())
        normal = int((~anomaly_mask).sum())

        return normal, anomalies, anomaly_mask

    except Exception:
        return len(df), 0, pd.Series(False, index=df.index)


def build_quality_score(df):
    rows = len(df)
    columns = len(df.columns)

    if rows == 0 or columns == 0:
        return 0.0, 0, 0, 0

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    invalid = detect_invalid_values(df)

    missing_rate = missing / (rows * columns)
    duplicate_rate = duplicates / rows
    invalid_rate = invalid / rows

    score = (
        100
        - missing_rate * 50
        - duplicate_rate * 30
        - invalid_rate * 20
    )

    score = max(0, min(100, score))

    return (
        round(score, 2),
        missing,
        duplicates,
        invalid
    )


def find_column(df, keywords):
    for col in df.columns:
        normalized = str(col).lower().replace("_", " ").strip()

        for keyword in keywords:
            if keyword in normalized:
                return col

    return None


def clean_dataset(df):
    cleaned = df.copy()

    original_rows = len(cleaned)

    duplicate_count = int(cleaned.duplicated().sum())

    cleaned = cleaned.drop_duplicates().copy()

    rows_removed = original_rows - len(cleaned)

    numeric_cols = cleaned.select_dtypes(
        include=np.number
    ).columns

    categorical_cols = cleaned.select_dtypes(
        include=["object", "category"]
    ).columns

    before_missing = int(cleaned.isna().sum().sum())

    for col in numeric_cols:
        if cleaned[col].isna().any():
            median = cleaned[col].median()

            if pd.notna(median):
                cleaned[col] = cleaned[col].fillna(median)

    for col in categorical_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode()

            if len(mode) > 0:
                cleaned[col] = cleaned[col].fillna(mode.iloc[0])
            else:
                cleaned[col] = cleaned[col].fillna("Unknown")

    after_missing = int(cleaned.isna().sum().sum())

    values_filled = before_missing - after_missing

    city_standardized = 0

    city_col = find_column(
        cleaned,
        ["city"]
    )

    if city_col:
        original = cleaned[city_col].copy()

        cleaned[city_col] = (
            cleaned[city_col]
            .astype(str)
            .str.strip()
            .str.title()
        )

        city_standardized = int(
            (original.astype(str) != cleaned[city_col]).sum()
        )

    return (
        cleaned,
        {
            "rows_removed": rows_removed,
            "values_filled": values_filled,
            "city_standardized": city_standardized,
            "duplicates_removed": duplicate_count,
        }
    )


def prepare_business_data(df):
    sales_col = find_column(
        df,
        ["sales", "revenue", "amount"]
    )

    profit_col = find_column(
        df,
        ["profit", "net profit"]
    )

    quantity_col = find_column(
        df,
        ["quantity", "units"]
    )

    result = {}

    if sales_col:
        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        )

        result["sales_column"] = sales_col
        result["total_sales"] = float(sales.sum())
        result["average_sale"] = float(sales.mean())
        result["highest_sale"] = float(sales.max())

        q1 = sales.quantile(.25)
        q3 = sales.quantile(.75)
        iqr = q3 - q1

        result["sales_upper_bound"] = float(
            q3 + 1.5 * iqr
        )

        if profit_col:
            profit = pd.to_numeric(
                df[profit_col],
                errors="coerce"
            )

            result["total_profit"] = float(
                profit.sum()
            )

            result["profit_margin"] = (
                float(profit.sum() / sales.sum() * 100)
                if sales.sum() != 0
                else 0
            )

        if quantity_col:
            quantity = pd.to_numeric(
                df[quantity_col],
                errors="coerce"
            )

            result["total_quantity"] = float(
                quantity.sum()
            )

            result["average_order_value"] = (
                float(sales.mean())
            )

    return result


def create_powerbi_exports(df):
    files = {}

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
                str(df[col].dtype)
                for col in df.columns
            ],
            "Missing Values": [
                int(df[col].isna().sum())
                for col in df.columns
            ],
            "Unique Values": [
                int(df[col].nunique())
                for col in df.columns
            ],
        })

        profile.to_excel(
            writer,
            sheet_name="Profile",
            index=False
        )

        business = prepare_business_data(df)

        pd.DataFrame(
            [business]
        ).to_excel(
            writer,
            sheet_name="Business_Summary",
            index=False
        )

    files["PowerBI_Data.xlsx"] = buffer.getvalue()

    return files


def create_zip(files):
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for filename, content in files.items():
            z.writestr(
                filename,
                content
            )

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

def run_gemini(analysis):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is not configured.\n\n"
            "Add GEMINI_API_KEY to your Streamlit secrets "
            "to enable AI analysis."
        )

    try:
        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are a senior data quality and business analytics consultant.

Analyze ONLY the measured information below.

Do not invent values.
Do not recalculate values.
Do not change numerical values.

Dataset:
{analysis}

Create exactly these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

Important:
- Isolation Forest anomalies are screening signals, not confirmed errors.
- IQR outliers are statistical signals, not automatically incorrect values.
- Clearly distinguish data quality problems from unusual observations.
- Give practical recommendations.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
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
        return (
            "Gemini analysis could not be generated.\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div>
                <div class="brand-name">DataGuard AI</div>
                <div class="brand-sub">AI Data Quality Platform</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    navigation = [
        ("Dashboard", "◉"),
        ("Upload Data", "↑"),
        ("Data Profile", "▦"),
        ("Quality Checks", "✓"),
        ("Anomaly Detection", "⌁"),
        ("Data Cleaning", "✦"),
        ("Analytics", "▥"),
        ("Power BI", "▤"),
        ("AI Analysis", "✧"),
        ("Reports", "▣"),
    ]

    for name, icon in navigation:

        is_current = (
            st.session_state.page == name
        )

        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            type="primary" if is_current else "secondary"
        ):
            st.session_state.page = name
            st.rerun()

    st.markdown(
        '<div class="sidebar-label">DETECTION SETTINGS</div>',
        unsafe_allow_html=True
    )

    sensitivity = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=int(
            st.session_state.contamination_pct
        ),
        help=(
            "Higher sensitivity flags more records "
            "as unusual."
        ),
        label_visibility="visible"
    )

    st.session_state.contamination_pct = sensitivity

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.markdown(
        """
        <div class="system-box">
            <div class="system-title">SYSTEM</div>
            <div class="system-status">
                ● All systems operational
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE STATE
# ============================================================

page = st.session_state.page

workflow_step = {
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
}.get(page, 0)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="topbar">
        <div>
            <div class="page-kicker">
                DATA INTELLIGENCE WORKSPACE
            </div>
            <div class="page-title">
                {page}
            </div>
            <div class="page-description">
                AI-powered data quality, anomaly detection
                and business analytics.
            </div>
        </div>

        <div style="
            padding:8px 12px;
            border:1px solid #1e2a39;
            border-radius:10px;
            color:#8d9aad;
            font-size:11px;
        ">
            🛡 DataGuard AI
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKFLOW
# ============================================================

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

st.markdown(
    '<div class="workflow">',
    unsafe_allow_html=True
)

for i, step in enumerate(steps, start=1):

    if workflow_step == 0:
        state = ""
    elif i < workflow_step:
        state = "done"
    elif i == workflow_step:
        state = "active"
    else:
        state = ""

    if state == "done":
        number = "✓"
    else:
        number = f"{i:02d}"

    st.markdown(
        f"""
        <div class="workflow-step {state}">
            <div class="workflow-number">
                {number}
            </div>
            <div class="workflow-name">
                {step}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if i < len(steps):
        st.markdown(
            '<div class="workflow-arrow">→</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD PROCESSING
# ============================================================

if page in ["Dashboard", "Upload Data"]:

    if st.session_state.df is None:

        if page == "Dashboard":
            st.markdown(
                """
                <div class="card">
                    <div style="
                        font-size:28px;
                        margin-bottom:8px;
                    ">
                        👋
                    </div>

                    <div class="card-title"
                         style="font-size:24px;">
                        Welcome to DataGuard AI
                    </div>

                    <div class="card-description"
                         style="font-size:13px;">
                        Upload a dataset to begin your
                        data intelligence workflow.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="upload-box">

                <div class="upload-icon">
                    📊
                </div>

                <div class="upload-title">
                    Upload your dataset
                </div>

                <div class="upload-text">
                    CSV, XLSX or XLS files • Maximum 50 MB
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Choose your dataset",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

        if uploaded is not None:

            try:

                if uploaded.name.lower().endswith(".csv"):

                    df = pd.read_csv(
                        uploaded
                    )

                else:

                    df = pd.read_excel(
                        uploaded
                    )

                st.session_state.df = df
                st.session_state.cleaned_df = None
                st.session_state.file_name = uploaded.name
                st.session_state.analysis_complete = False
                st.session_state.analysis = None
                st.session_state.gemini_analysis = None

                st.success(
                    f"Successfully loaded {uploaded.name}"
                )

                st.session_state.page = "Data Profile"

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to read the file: {e}"
                )

    else:

        st.info(
            f"Dataset loaded: "
            f"**{st.session_state.file_name}**"
        )


# ============================================================
# DATA PROFILE
# ============================================================

if page == "Data Profile":

    df = st.session_state.df

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Dataset Profile
                </div>
                <div class="card-description">
                    Structure, data types, uniqueness
                    and missing-value overview.
                </div>
            </div>
            """,
            unsafe_allow_html=True
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
                f"{len(df.select_dtypes(include=np.number).columns):,}"
            )

        with c4:
            st.metric(
                "Categorical",
                f"{len(df.select_dtypes(include=['object', 'category']).columns):,}"
            )

        st.subheader("Column Profile")

        profile = pd.DataFrame({
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
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

if page == "Quality Checks":

    df = st.session_state.df

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        score, missing, duplicates, invalid = (
            build_quality_score(df)
        )

        outliers, outlier_details = (
            detect_iqr_outliers(df)
        )

        datetime_columns = detect_datetime_columns(df)

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.metric(
                "Quality Score",
                f"{score:.2f}%"
            )

        with c2:
            st.metric(
                "Missing Values",
                f"{missing:,}"
            )

        with c3:
            st.metric(
                "Duplicates",
                f"{duplicates:,}"
            )

        with c4:
            st.metric(
                "Invalid Values",
                f"{invalid:,}"
            )

        with c5:
            st.metric(
                "IQR Outliers",
                f"{outliers:,}"
            )

        left, right = st.columns([1, 1])

        with left:

            score_value = min(
                100,
                max(0, score)
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Overall Data Quality
                    </div>

                    <div class="score-ring"
                         style="--score:{score_value}%">

                        <div class="score-value">
                            {score:.1f}%
                        </div>

                    </div>

                    <div class="score-caption">
                        Based on missing values,
                        duplicates and invalid values.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:

            st.markdown(
                """
                <div class="card">
                    <div class="card-title">
                        Quality Checks
                    </div>
                """,
                unsafe_allow_html=True
            )

            checks = [
                (
                    "Completeness",
                    missing == 0
                ),
                (
                    "Uniqueness",
                    duplicates == 0
                ),
                (
                    "Validity",
                    invalid == 0
                ),
                (
                    "Datetime Detection",
                    len(datetime_columns) > 0
                ),
            ]

            for label, passed in checks:

                if passed:
                    icon = "✓"
                    css = "status-good"
                    text = "Passed"
                else:
                    icon = "!"
                    css = "status-warning"
                    text = "Review"

                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        padding:9px 0;
                        border-bottom:1px solid #182231;
                        font-size:12px;
                    ">
                        <span>{label}</span>
                        <span class="{css}">
                            {icon} {text}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )


# ============================================================
# ANOMALY DETECTION
# ============================================================

if page == "Anomaly Detection":

    df = st.session_state.df

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        normal, anomalies, anomaly_mask = (
            detect_ml_anomalies(
                df,
                st.session_state.contamination_pct
            )
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Normal Records",
                f"{normal:,}"
            )

        with c2:
            st.metric(
                "ML Anomalies",
                f"{anomalies:,}"
            )

        with c3:
            rate = (
                anomalies / len(df) * 100
                if len(df)
                else 0
            )

            st.metric(
                "Anomaly Rate",
                f"{rate:.2f}%"
            )

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Isolation Forest
                </div>

                <div class="card-description">
                    Isolation Forest identifies records
                    that behave unusually compared with
                    the rest of the dataset.
                    These are screening signals,
                    not confirmed data errors.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        result = df.copy()

        result["Anomaly"] = np.where(
            anomaly_mask,
            "Anomaly",
            "Normal"
        )

        st.dataframe(
            result[result["Anomaly"] == "Anomaly"].head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DATA CLEANING
# ============================================================

if page == "Data Cleaning":

    df = st.session_state.df

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Data Cleaning
                </div>
                <div class="card-description">
                    Remove duplicates, fill missing values
                    and standardize categorical values.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Run Data Cleaning",
            type="primary"
        ):

            cleaned, result = clean_dataset(df)

            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_result = result

            st.success(
                "Data cleaning completed successfully."
            )

        if st.session_state.cleaned_df is not None:

            cleaned = st.session_state.cleaned_df
            result = st.session_state.cleaning_result

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
                    f"{len(cleaned):,}"
                )

            st.subheader("Cleaned Data Preview")

            st.dataframe(
                cleaned.head(100),
                use_container_width=True,
                hide_index=True
            )

            csv = cleaned.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Cleaned CSV",
                csv,
                file_name="dataguard_cleaned.csv",
                mime="text/csv"
            )


# ============================================================
# ANALYTICS
# ============================================================

if page == "Analytics":

    df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.df
    )

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        business = prepare_business_data(df)

        if not business:

            st.info(
                "No sales/revenue column was detected "
                "for business analytics."
            )

        else:

            st.markdown(
                """
                <div class="card">
                    <div class="card-title">
                        Business Analytics
                    </div>
                    <div class="card-description">
                        Key commercial indicators derived
                        from the uploaded dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            metrics = [
                (
                    "Total Sales",
                    business.get("total_sales")
                ),
                (
                    "Average Sale",
                    business.get("average_sale")
                ),
                (
                    "Highest Sale",
                    business.get("highest_sale")
                ),
                (
                    "Profit Margin",
                    business.get("profit_margin")
                ),
            ]

            cols = st.columns(4)

            for col, (label, value) in zip(
                cols,
                metrics
            ):

                with col:

                    if value is None:
                        display = "—"
                    elif label == "Profit Margin":
                        display = f"{value:.2f}%"
                    else:
                        display = f"{value:,.2f}"

                    st.metric(
                        label,
                        display
                    )

            sales_col = business.get(
                "sales_column"
            )

            if sales_col:

                chart_df = df[
                    [sales_col]
                ].copy()

                chart_df[sales_col] = pd.to_numeric(
                    chart_df[sales_col],
                    errors="coerce"
                )

                st.subheader(
                    "Sales Distribution"
                )

                st.bar_chart(
                    chart_df[sales_col]
                    .dropna()
                    .head(100)
                )


# ============================================================
# POWER BI
# ============================================================

if page == "Power BI":

    df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.df
    )

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Power BI Export Center
                </div>

                <div class="card-description">
                    Prepare cleaned and profiled data
                    for Power BI reporting.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        files = create_powerbi_exports(
            df
        )

        zip_data = create_zip(
            files
        )

        st.download_button(
            "Download Power BI Package",
            zip_data,
            file_name="DataGuard_PowerBI_Package.zip",
            mime="application/zip",
            type="primary"
        )

        st.write(
            "Package includes:"
        )

        st.write(
            "• Cleaned data\n"
            "• Data profile\n"
            "• Business summary"
        )


# ============================================================
# AI ANALYSIS
# ============================================================

if page == "AI Analysis":

    df = (
        st.session_state.cleaned_df
        if st.session_state.cleaned_df is not None
        else st.session_state.df
    )

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        score, missing, duplicates, invalid = (
            build_quality_score(df)
        )

        outliers, _ = detect_iqr_outliers(df)

        normal, anomalies, _ = (
            detect_ml_anomalies(
                df,
                st.session_state.contamination_pct
            )
        )

        business = prepare_business_data(df)

        analysis = {
            "dataset_name": st.session_state.file_name,
            "rows": len(df),
            "columns": len(df.columns),
            "quality_score": score,
            "missing_values": missing,
            "duplicates": duplicates,
            "invalid_values": invalid,
            "iqr_outliers": outliers,
            "ml_normal_records": normal,
            "ml_anomalies": anomalies,
            "isolation_forest_sensitivity_percent":
                st.session_state.contamination_pct,
            "business_metrics": business,
        }

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Gemini AI Data Intelligence
                </div>

                <div class="card-description">
                    Generate an AI-assisted interpretation
                    of the measured data quality and
                    business findings.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Generate AI Analysis",
            type="primary"
        ):

            with st.spinner(
                "Gemini is analyzing the dataset..."
            ):

                result = run_gemini(
                    analysis
                )

                st.session_state.gemini_analysis = result

        if st.session_state.gemini_analysis:

            text = st.session_state.gemini_analysis

            lines = text.splitlines()

            current_section = None
            content = []

            for line in lines:

                match = re.match(
                    r"^\s*#{1,6}\s*(.+?)\s*$",
                    line
                )

                if match:

                    if current_section and content:

                        st.markdown(
                            "\n".join(content)
                        )

                    current_section = match.group(1)

                    st.markdown(
                        f"""
                        <div style="
                            margin-top:20px;
                            margin-bottom:8px;
                            color:white;
                            font-size:16px;
                            font-weight:700;
                        ">
                            {current_section}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    content = []

                else:

                    content.append(line)

            if content:
                st.markdown(
                    "\n".join(content)
                )


# ============================================================
# REPORTS
# ============================================================

if page == "Reports":

    df = st.session_state.df

    if df is None:
        st.warning(
            "Upload a dataset first."
        )

    else:

        score, missing, duplicates, invalid = (
            build_quality_score(df)
        )

        outliers, _ = detect_iqr_outliers(df)

        normal, anomalies, _ = (
            detect_ml_anomalies(
                df,
                st.session_state.contamination_pct
            )
        )

        report = f"""
DataGuard AI — Data Quality Report

Dataset:
{st.session_state.file_name}

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M")}

DATASET OVERVIEW
Rows: {len(df):,}
Columns: {len(df.columns):,}

DATA QUALITY
Quality Score: {score:.2f}%
Missing Values: {missing:,}
Duplicates: {duplicates:,}
Invalid Values: {invalid:,}

STATISTICAL ANALYSIS
IQR Outliers: {outliers:,}

ML ANOMALY ANALYSIS
Normal Records: {normal:,}
Anomalies: {anomalies:,}
Sensitivity: {st.session_state.contamination_pct}%

DataGuard AI
AI Data Quality & Anomaly Detection Platform
"""

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Executive Data Quality Report
                </div>

                <div class="card-description">
                    Download a concise summary of the
                    current dataset assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_area(
            "Report Preview",
            report,
            height=420
        )

        st.download_button(
            "Download Report",
            report,
            file_name="DataGuard_AI_Report.txt",
            mime="text/plain",
            type="primary"
        )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard" and st.session_state.df is not None:

    df = st.session_state.df

    score, missing, duplicates, invalid = (
        build_quality_score(df)
    )

    outliers, _ = detect_iqr_outliers(df)

    normal, anomalies, _ = (
        detect_ml_anomalies(
            df,
            st.session_state.contamination_pct
        )
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                Dataset Intelligence Overview
            </div>

            <div class="card-description">
                Current quality and anomaly indicators
                for the uploaded dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(6)

    dashboard_metrics = [
        ("Quality Score", f"{score:.2f}%"),
        ("Records", f"{len(df):,}"),
        ("Columns", f"{len(df.columns):,}"),
        ("Missing", f"{missing:,}"),
        ("Duplicates", f"{duplicates:,}"),
        ("Anomalies", f"{anomalies:,}"),
    ]

    for col, (label, value) in zip(
        cols,
        dashboard_metrics
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

                    <div class="kpi-sub">
                        DataGuard AI
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:

        score_value = max(
            0,
            min(100, score)
        )

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Data Quality Score
                </div>

                <div class="score-ring"
                     style="--score:{score_value}%">

                    <div class="score-value">
                        {score:.1f}%
                    </div>

                </div>

                <div class="score-caption">
                    Completeness • Consistency • Validity
                    • Uniqueness
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        rate = (
            anomalies / len(df) * 100
            if len(df)
            else 0
        )

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Anomaly Detection
                </div>

                <div class="card-description">
                    Isolation Forest screening results.
                </div>

                <div style="
                    display:grid;
                    grid-template-columns:1fr 1fr;
                    gap:10px;
                    margin-top:18px;
                ">

                    <div class="kpi-card">
                        <div class="kpi-label">
                            Normal
                        </div>
                        <div class="kpi-value">
                            {normal:,}
                        </div>
                    </div>

                    <div class="kpi-card">
                        <div class="kpi-label">
                            Anomalies
                        </div>
                        <div class="kpi-value">
                            {anomalies:,}
                        </div>
                    </div>

                </div>

                <div style="
                    color:#78869a;
                    font-size:11px;
                    margin-top:14px;
                ">
                    Anomaly rate: {rate:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br>
        Python • Pandas • Scikit-learn • Streamlit • Gemini
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)
