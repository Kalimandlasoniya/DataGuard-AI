import io
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
# PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #f4f7fb;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #94a3b8 !important;
    }

    .sidebar-brand {
        padding: 8px 4px 20px 4px;
    }

    .sidebar-logo {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
    }

    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 5px;
        line-height: 1.5;
    }

    .sidebar-section {
        color: #60a5fa;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 18px;
        margin-bottom: 8px;
    }

    .pipeline-item {
        padding: 8px 10px;
        margin: 4px 0;
        border-radius: 8px;
        background: #111c2f;
        color: #cbd5e1;
        font-size: 13px;
    }

    /* =========================
       HERO
       ========================= */

    .hero {
        background: linear-gradient(
            135deg,
            #0b1220 0%,
            #172554 55%,
            #1d4ed8 100%
        );
        border-radius: 20px;
        padding: 34px 38px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 850;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 17px;
        color: #dbeafe;
        margin-top: 10px;
        max-width: 800px;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 7px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        color: #dbeafe;
        font-size: 12px;
        font-weight: 700;
    }

    /* =========================
       SECTION TITLES
       ========================= */

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #111827;
        margin: 12px 0 14px 0;
    }

    .section-subtitle {
        color: #667085;
        font-size: 14px;
        margin-top: -7px;
        margin-bottom: 18px;
    }

    /* =========================
       KPI CARDS
       ========================= */

    .kpi {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px 19px;
        min-height: 125px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .kpi-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 700;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 29px;
        font-weight: 850;
        margin-top: 9px;
    }

    .kpi-help {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 5px;
    }

    /* =========================
       FEATURE CARDS
       ========================= */

    .feature {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        min-height: 185px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .feature-icon {
        font-size: 28px;
    }

    .feature-title {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
        margin-top: 10px;
    }

    .feature-text {
        color: #667085;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 7px;
    }

    /* =========================
       STATUS CARDS
       ========================= */

    .status-good {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        color: #05603a;
        padding: 15px 18px;
        border-radius: 12px;
    }

    .status-warning {
        background: #fffaeb;
        border: 1px solid #fedf89;
        color: #854d0e;
        padding: 15px 18px;
        border-radius: 12px;
    }

    .status-danger {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 15px 18px;
        border-radius: 12px;
    }

    .status-title {
        font-weight: 800;
        font-size: 15px;
    }

    .status-text {
        font-size: 13px;
        margin-top: 4px;
    }

    /* =========================
       CONTENT CARDS
       ========================= */

    .content-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    /* =========================
       UPLOAD AREA
       ========================= */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #dbe3ef;
        padding: 8px;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 700;
    }

    /* =========================
       TABS
       ========================= */

    button[data-baseweb="tab"] {
        font-weight: 700;
    }

    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 30px 0 5px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru",
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

    datetime_columns = []

    date_keywords = (
        "date",
        "time",
        "timestamp",
        "created",
        "updated",
        "modified",
        "dob",
        "birth",
        "day",
    )

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Important:
        # Numeric IDs should never automatically become dates.
        if pd.api.types.is_numeric_dtype(series):
            continue

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        converted = safe_to_datetime(series)

        if len(series) == 0:
            continue

        valid_ratio = converted.notna().mean()

        if valid_ratio < 0.85:
            continue

        name = str(column).lower()

        sample = (
            series
            .dropna()
            .astype(str)
            .head(50)
        )

        if len(sample) == 0:
            continue

        looks_date_like = (
            sample
            .str.contains(
                r"[-/:]",
                regex=True
            )
            .mean()
            >= 0.50
        )

        if (
            any(
                keyword in name
                for keyword in date_keywords
            )
            or looks_date_like
        ):
            datetime_columns.append(column)

    return datetime_columns


def load_file(uploaded_file):

    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        if filename.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)

        if filename.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        st.error(
            "Please upload a CSV or Excel file."
        )

        return None

    except Exception as e:

        st.error(
            f"Could not read the file: {e}"
        )

        return None


def missing_summary(df):

    return (
        pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": (
                df.isna().mean().values * 100
            ).round(2),
        })
        .sort_values(
            "Missing Count",
            ascending=False,
        )
    )


def numeric_summary(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return pd.DataFrame()

    return (
        numeric
        .describe()
        .T
        .reset_index()
        .rename(
            columns={"index": "Column"}
        )
    )


def invalid_values(df):

    results = []
    masks = []

    for column in df.columns:

        name = str(column).lower()

        if (
            name == "age"
            or name.endswith("_age")
        ):

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            mask = (
                values.notna()
                &
                (
                    (values < 0)
                    |
                    (values > 100)
                )
            )

            count = int(mask.sum())

            if count:

                results.append({
                    "Column": column,
                    "Issue": "Invalid Age",
                    "Count": count,
                })

                masks.append(mask)

        if (
            name == "quantity"
            or name.endswith("_quantity")
        ):

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            mask = (
                values.notna()
                &
                (values <= 0)
            )

            count = int(mask.sum())

            if count:

                results.append({
                    "Column": column,
                    "Issue": "Invalid Quantity",
                    "Count": count,
                })

                masks.append(mask)

    result_df = pd.DataFrame(
        results,
        columns=[
            "Column",
            "Issue",
            "Count",
        ],
    )

    if masks:

        combined = pd.concat(
            masks,
            axis=1,
        ).any(axis=1)

        invalid_rows = int(
            combined.sum()
        )

    else:

        invalid_rows = 0

    return result_df, invalid_rows


def city_issues(df):

    if "City" not in df.columns:

        return {
            "labels": [],
            "rows": 0,
        }

    city = (
        df["City"]
        .astype("string")
        .str.strip()
    )

    mask = city.isin(
        CITY_MAPPING.keys()
    )

    return {
        "labels": sorted(
            city[mask]
            .dropna()
            .unique()
            .tolist()
        ),
        "rows": int(mask.sum()),
    }


def iqr_outliers(df):

    summary = []
    masks = {}

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        clean = values.dropna()

        if len(clean) < 5:
            continue

        q1 = clean.quantile(0.25)
        q3 = clean.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            values < lower
        ) | (
            values > upper
        )

        masks[column] = mask

        summary.append({
            "Column": column,
            "Lower Bound": round(
                lower, 2
            ),
            "Upper Bound": round(
                upper, 2
            ),
            "Outlier Count": int(
                mask.sum()
            ),
        })

    summary_df = pd.DataFrame(
        summary
    )

    if masks:

        combined = pd.concat(
            [
                mask.rename(column)
                for column, mask
                in masks.items()
            ],
            axis=1,
        ).any(axis=1)

        count = int(
            combined.sum()
        )

    else:

        count = 0

    return summary_df, count


def ml_anomalies(
    df,
    contamination=0.05,
):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    # Do not use IDs as ML features.
    remove = []

    for column in numeric.columns:

        name = str(column).lower()

        if (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or "customer_id" in name
            or "order_id" in name
        ):
            remove.append(column)

    numeric = numeric.drop(
        columns=remove,
        errors="ignore",
    )

    if numeric.empty:
        return None, 0

    numeric = numeric.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric = numeric.fillna(
        numeric.median()
    )

    numeric = numeric.dropna(
        axis=1,
        how="all",
    )

    if numeric.empty:
        return None, 0

    try:

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )

        prediction = model.fit_predict(
            numeric
        )

    except Exception:

        return None, 0

    result = df.copy()

    result["ML_Anomaly"] = (
        prediction == -1
    )

    count = int(
        result["ML_Anomaly"].sum()
    )

    return result, count


def quality_score(
    df,
    invalid_count,
    city_rows,
    outlier_rows,
):

    if df.empty:
        return 0

    rows = max(
        len(df),
        1,
    )

    cells = max(
        len(df) * df.shape[1],
        1,
    )

    missing_rate = (
        df.isna().sum().sum()
        / cells
    )

    duplicate_rate = (
        df.duplicated().sum()
        / rows
    )

    invalid_rate = min(
        invalid_count / rows,
        1,
    )

    city_rate = min(
        city_rows / rows,
        1,
    )

    outlier_rate = min(
        outlier_rows / rows,
        1,
    )

    score = (
        25 * (1 - missing_rate)
        + 20 * (1 - duplicate_rate)
        + 25 * (1 - invalid_rate)
        + 15 * (1 - city_rate)
        + 15 * (1 - outlier_rate)
    )

    return round(
        max(
            0,
            min(
                100,
                score,
            ),
        ),
        1,
    )


def score_status(score):

    if score >= 90:
        return (
            "Excellent",
            "good",
        )

    if score >= 75:
        return (
            "Good",
            "good",
        )

    if score >= 60:
        return (
            "Needs Review",
            "warning",
        )

    return (
        "Critical",
        "danger",
    )


def clean_dataset(df):

    cleaned = df.copy()

    # City standardization
    if "City" in cleaned.columns:

        cleaned["City"] = (
            cleaned["City"]
            .astype("string")
            .str.strip()
            .replace(CITY_MAPPING)
        )

    # Age / Quantity
    for column in cleaned.columns:

        name = str(column).lower()

        if (
            name == "age"
            or name.endswith("_age")
        ):

            values = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            ).astype("float64")

            values.loc[
                (values < 0)
                |
                (values > 100)
            ] = np.nan

            cleaned[column] = values

            cleaned[column] = (
                cleaned[column]
                .fillna(
                    cleaned[column].median()
                )
            )

        elif (
            name == "quantity"
            or name.endswith("_quantity")
        ):

            values = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            ).astype("float64")

            values.loc[
                values <= 0
            ] = np.nan

            cleaned[column] = values

            cleaned[column] = (
                cleaned[column]
                .fillna(
                    cleaned[column].median()
                )
            )

    # General missing values
    for column in cleaned.columns:

        if not cleaned[column].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):

            cleaned[column] = (
                cleaned[column]
                .fillna(
                    cleaned[column].median()
                )
            )

        else:

            mode = (
                cleaned[column]
                .mode(
                    dropna=True
                )
            )

            if not mode.empty:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        mode.iloc[0]
                    )
                )

            else:

                cleaned[column] = (
                    cleaned[column]
                    .fillna("Unknown")
                )

    # Remove exact duplicates
    cleaned = cleaned.drop_duplicates()

    return cleaned


def data_dictionary(df):

    rows = []

    for column in df.columns:

        rows.append({
            "Column": column,
            "Data Type": str(
                df[column].dtype
            ),
            "Non-Null Count": int(
                df[column].notna().sum()
            ),
            "Missing Count": int(
                df[column].isna().sum()
            ),
            "Unique Values": int(
                df[column].nunique(
                    dropna=True
                )
            ),
        })

    return pd.DataFrame(rows)


def zip_package(files):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        for name, content in files.items():

            z.writestr(
                name,
                content,
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🛡️ DataGuard AI</div>
        <div class="sidebar-subtitle">
            Data quality monitoring<br>
            & anomaly detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sidebar-section">Upload</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx",
            "xls",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-section">ML Settings</div>',
        unsafe_allow_html=True,
    )

    anomaly_sensitivity = st.slider(
        "Anomaly sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.caption(
        f"Current setting: **{anomaly_sensitivity}%**"
    )

    st.caption(
        "Controls the expected fraction of records "
        "flagged for ML review."
    )

    st.markdown(
        '<div class="sidebar-section">Pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        "1  Upload",
        "2  Profile",
        "3  Detect issues",
        "4  Analyze anomalies",
        "5  Clean data",
        "6  Export for Power BI",
    ]

    for step in steps:

        st.markdown(
            f'<div class="pipeline-item">{step}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="sidebar-section">Technology</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# LANDING PAGE
# ============================================================

if uploaded_file is None:

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            🛡️ DataGuard AI
        </div>

        <div class="hero-subtitle">
            AI-powered data quality monitoring,
            anomaly detection and intelligent data cleaning.
            Turn raw datasets into reliable, analysis-ready data.
        </div>

        <div class="hero-badge">
            DATA QUALITY • ANOMALY DETECTION • POWER BI READY
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Start with your dataset</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a CSV or Excel file using the sidebar to begin the automated pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="feature">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">
                Detect Data Issues
            </div>
            <div class="feature-text">
                Find missing values, duplicates, invalid values,
                inconsistent categories and statistical outliers.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="feature">
            <div class="feature-icon">🚨</div>
            <div class="feature-title">
                Detect Anomalies
            </div>
            <div class="feature-text">
                Use Isolation Forest to identify unusual records
                that deserve further investigation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="feature">
            <div class="feature-icon">📦</div>
            <div class="feature-title">
                Clean & Export
            </div>
            <div class="feature-text">
                Automatically clean the dataset and create
                Power BI-ready CSV reports.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="content-card">

    <h3>🚀 What DataGuard AI does</h3>

    <p style="color:#667085; line-height:1.7;">
    DataGuard AI provides an end-to-end data quality workflow:
    profiling → validation → anomaly detection → cleaning →
    business analysis → Power BI export.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">'
        'DataGuard AI • Built with Python, Pandas, Scikit-learn & Streamlit'
        '</div>',
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_file(
    uploaded_file
)

if df is None:
    st.stop()


# ============================================================
# CALCULATIONS
# ============================================================

rows = len(df)
columns = df.shape[1]

missing = int(
    df.isna().sum().sum()
)

duplicates = int(
    df.duplicated().sum()
)

numeric_count = len(
    df.select_dtypes(
        include=np.number
    ).columns
)

categorical_count = (
    columns - numeric_count
)

dates = detect_datetime_columns(
    df
)

invalid_df, invalid_count = invalid_values(
    df
)

cities = city_issues(
    df
)

outlier_df, outlier_count = iqr_outliers(
    df
)

ml_df, anomaly_count = ml_anomalies(
    df,
    contamination=anomaly_sensitivity / 100,
)

anomaly_rate = (
    anomaly_count
    / max(rows, 1)
    * 100
)

score = quality_score(
    df,
    invalid_count,
    cities["rows"],
    outlier_count,
)

score_text, score_type = score_status(
    score
)


# ============================================================
# MAIN HEADER AFTER UPLOAD
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🛡️ DataGuard AI
    </div>

    <div class="hero-subtitle">
        Data quality dashboard for
        <b>""" + str(uploaded_file.name) + """</b>
    </div>

    <div class="hero-badge">
        ANALYSIS READY
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

metrics = [
    (
        k1,
        "ROWS",
        f"{rows:,}",
        "Records",
    ),
    (
        k2,
        "COLUMNS",
        f"{columns}",
        "Features",
    ),
    (
        k3,
        "MISSING",
        f"{missing:,}",
        "Missing cells",
    ),
    (
        k4,
        "DUPLICATES",
        f"{duplicates:,}",
        "Duplicate rows",
    ),
    (
        k5,
        "QUALITY",
        f"{score}/100",
        score_text,
    ),
    (
        k6,
        "ML FLAGS",
        f"{anomaly_rate:.1f}%",
        "Screening rate",
    ),
]

for col, label, value, help_text in metrics:

    with col:

        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.write("")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📊 Overview",
    "🔍 Data Quality",
    "🚨 Anomalies",
    "📈 Insights",
    "🧹 Cleaning",
    "🤖 AI Analysis",
    "📦 Power BI",
    "📄 Report",
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=430,
    )

    st.markdown(
        '<div class="section-title">Detected Structure</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Numeric Columns",
        numeric_count,
    )

    b.metric(
        "Categorical Columns",
        categorical_count,
    )

    c.metric(
        "Datetime Columns",
        len(dates),
    )

    d.metric(
        "Invalid Values",
        invalid_count,
    )

    if dates:

        st.markdown(
            f"""
            <div class="status-good">
                <div class="status-title">
                    ✓ Date columns detected
                </div>
                <div class="status-text">
                    {", ".join(dates)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "No reliable datetime column was detected."
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">Data Quality Assessment</div>',
        unsafe_allow_html=True,
    )

    if score_type == "good":

        st.markdown(
            f"""
            <div class="status-good">
                <div class="status-title">
                    🟢 {score_text} — {score}/100
                </div>
                <div class="status-text">
                    The dataset has relatively strong rule-based data quality.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif score_type == "warning":

        st.markdown(
            f"""
            <div class="status-warning">
                <div class="status-title">
                    🟡 {score_text} — {score}/100
                </div>
                <div class="status-text">
                    Some data-quality issues should be reviewed before analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div class="status-danger">
                <div class="status-title">
                    🔴 {score_text} — {score}/100
                </div>
                <div class="status-text">
                    Significant data-quality issues require attention.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    missing_rate = (
        missing
        / max(rows * columns, 1)
        * 100
    )

    duplicate_rate = (
        duplicates
        / max(rows, 1)
        * 100
    )

    invalid_rate = (
        invalid_count
        / max(rows, 1)
        * 100
    )

    city_rate = (
        cities["rows"]
        / max(rows, 1)
        * 100
    )

    outlier_rate = (
        outlier_count
        / max(rows, 1)
        * 100
    )

    breakdown = pd.DataFrame({
        "Metric": [
            "Missing",
            "Duplicates",
            "Invalid",
            "City inconsistencies",
            "IQR outliers",
        ],
        "Rate": [
            missing_rate,
            duplicate_rate,
            invalid_rate,
            city_rate,
            outlier_rate,
        ],
    }).set_index(
        "Metric"
    )

    st.markdown(
        '<div class="section-title">Quality Breakdown</div>',
        unsafe_allow_html=True,
    )

    st.bar_chart(
        breakdown
    )

    st.markdown(
        '<div class="section-title">Missing Values</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        missing_summary(df),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">Invalid Values</div>',
        unsafe_allow_html=True,
    )

    if invalid_df.empty:

        st.success(
            "No invalid Age or Quantity values detected."
        )

    else:

        st.dataframe(
            invalid_df,
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-title">City Consistency</div>',
        unsafe_allow_html=True,
    )

    if cities["rows"]:

        st.warning(
            f"{cities['rows']:,} rows contain mapped city inconsistencies."
        )

        st.write(
            "Values:",
            ", ".join(
                cities["labels"]
            ),
        )

    else:

        st.success(
            "No mapped city inconsistencies detected."
        )

    st.markdown(
        '<div class="section-title">IQR Outliers</div>',
        unsafe_allow_html=True,
    )

    if outlier_df.empty:

        st.info(
            "No suitable numeric columns for IQR analysis."
        )

    else:

        st.dataframe(
            outlier_df,
            use_container_width=True,
        )

        st.caption(
            f"{outlier_count:,} rows contain at least one IQR outlier."
        )


# ============================================================
# ANOMALIES
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">Machine Learning Anomaly Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-warning">
            <div class="status-title">
                🚨 ML screening sensitivity: {anomaly_sensitivity}%
            </div>
            <div class="status-text">
                Isolation Forest flags unusual records for review.
                A flagged record is not automatically a data error.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    a, b, c = st.columns(3)

    a.metric(
        "Records",
        f"{rows:,}",
    )

    b.metric(
        "ML Anomalies",
        f"{anomaly_count:,}",
    )

    c.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%",
    )

    if ml_df is not None:

        st.markdown(
            '<div class="section-title">Flagged Records</div>',
            unsafe_allow_html=True,
        )

        flagged = ml_df[
            ml_df["ML_Anomaly"]
        ]

        st.dataframe(
            flagged.head(100),
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "⬇️ Download ML Results",
            data=ml_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ml_anomaly_results.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "No suitable numeric features were available for ML anomaly detection."
        )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">Business Insights</div>',
        unsafe_allow_html=True,
    )

    if "Sales" in df.columns:

        sales = pd.to_numeric(
            df["Sales"],
            errors="coerce",
        ).dropna()

        if not sales.empty:

            a, b, c, d = st.columns(4)

            a.metric(
                "Total Sales",
                f"{sales.sum():,.2f}",
            )

            b.metric(
                "Average Sale",
                f"{sales.mean():,.2f}",
            )

            c.metric(
                "Median Sale",
                f"{sales.median():,.2f}",
            )

            d.metric(
                "Maximum Sale",
                f"{sales.max():,.2f}",
            )

            st.markdown(
                '<div class="section-title">Sales Distribution</div>',
                unsafe_allow_html=True,
            )

            counts, bins = np.histogram(
                sales,
                bins=20,
            )

            histogram = pd.DataFrame({
                "Sales Range": [
                    f"{bins[i]:,.0f} - {bins[i + 1]:,.0f}"
                    for i in range(
                        len(bins) - 1
                    )
                ],
                "Transactions": counts,
            }).set_index(
                "Sales Range"
            )

            st.bar_chart(
                histogram
            )

            if "Category" in df.columns:

                category_sales = (
                    df.assign(
                        Sales=pd.to_numeric(
                            df["Sales"],
                            errors="coerce",
                        )
                    )
                    .groupby(
                        "Category"
                    )["Sales"]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    '<div class="section-title">Sales by Category</div>',
                    unsafe_allow_html=True,
                )

                st.bar_chart(
                    category_sales
                )

    else:

        st.info(
            "A Sales column was not detected."
        )

    if dates:

        st.markdown(
            '<div class="section-title">Date Trend</div>',
            unsafe_allow_html=True,
        )

        selected_date = st.selectbox(
            "Select date column",
            dates,
        )

        date_values = safe_to_datetime(
            df[selected_date]
        )

        trend_df = df.copy()

        trend_df["_Date"] = date_values

        if "Sales" in trend_df.columns:

            trend_df["Sales"] = pd.to_numeric(
                trend_df["Sales"],
                errors="coerce",
            )

            trend = (
                trend_df
                .dropna(
                    subset=["_Date"]
                )
                .groupby(
                    trend_df["_Date"].dt.date
                )["Sales"]
                .sum()
            )

        else:

            trend = (
                trend_df
                .dropna(
                    subset=["_Date"]
                )
                .groupby(
                    trend_df["_Date"].dt.date
                )
                .size()
            )

        if not trend.empty:

            st.line_chart(
                trend
            )

    else:

        st.info(
            "Trend analysis is unavailable because no reliable date column was detected."
        )


# ============================================================
# CLEANING
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">Automated Data Cleaning</div>',
        unsafe_allow_html=True,
    )

    cleaned = clean_dataset(
        df
    )

    before_missing = int(
        df.isna().sum().sum()
    )

    after_missing = int(
        cleaned.isna().sum().sum()
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Rows Before",
        f"{len(df):,}",
    )

    b.metric(
        "Rows After",
        f"{len(cleaned):,}",
    )

    c.metric(
        "Missing Before",
        f"{before_missing:,}",
    )

    d.metric(
        "Missing After",
        f"{after_missing:,}",
    )

    st.markdown(
        '<div class="section-title">Cleaning Operations</div>',
        unsafe_allow_html=True,
    )

    operations = [
        "Standardized mapped city names",
        "Validated Age values",
        "Validated Quantity values",
        "Filled numeric missing values using median",
        "Filled categorical missing values using mode",
        "Removed exact duplicate rows",
    ]

    for operation in operations:

        st.write(
            f"✓ {operation}"
        )

    st.markdown(
        '<div class="section-title">Cleaned Dataset</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        cleaned.head(20),
        use_container_width=True,
        height=400,
    )

    st.download_button(
        "⬇️ Download Cleaned CSV",
        data=cleaned.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )


# ============================================================
# AI ANALYSIS
# ============================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">AI Data Quality Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
        <b>🤖 Evidence-based AI analysis</b>
        <p style="color:#667085;">
        Gemini receives calculated DataGuard AI metrics and is instructed
        not to invent unsupported relationships or causes.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
    )

    if st.button(
        "🤖 Run AI Analysis",
        use_container_width=True,
    ):

        if not api_key:

            st.warning(
                "Please enter a Gemini API key."
            )

        else:

            try:

                import google.generativeai as genai

                genai.configure(
                    api_key=api_key
                )

                model = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )

                sales_summary = {}

                if "Sales" in df.columns:

                    sales = pd.to_numeric(
                        df["Sales"],
                        errors="coerce",
                    ).dropna()

                    if not sales.empty:

                        sales_summary = {
                            "total": round(
                                float(
                                    sales.sum()
                                ),
                                2,
                            ),
                            "average": round(
                                float(
                                    sales.mean()
                                ),
                                2,
                            ),
                            "median": round(
                                float(
                                    sales.median()
                                ),
                                2,
                            ),
                            "maximum": round(
                                float(
                                    sales.max()
                                ),
                                2,
                            ),
                        }

                prompt = f"""
You are a professional data quality analyst.

Use ONLY the facts supplied below.

Do NOT invent relationships.
Do NOT invent causes.
If you suggest a possible cause, label it as a HYPOTHESIS.

Distinguish:

1. Confirmed data-quality issues
2. Statistical outliers
3. Machine-learning screening flags

Isolation Forest anomaly flags are NOT proof that a record is incorrect.

Dataset:
Rows: {rows}
Columns: {columns}

Missing cells: {missing}
Duplicate rows: {duplicates}
Invalid values: {invalid_count}

City issue rows: {cities["rows"]}
City issue labels: {cities["labels"]}

IQR outlier rows: {outlier_count}

ML anomaly rows: {anomaly_count}
ML anomaly rate: {anomaly_rate:.2f}%

Rule-based quality score: {score}/100

Sales summary:
{sales_summary}

Return:

1. Executive summary
2. Confirmed issues
3. Statistical observations
4. ML anomaly interpretation
5. Possible causes (HYPOTHESES)
6. Recommended actions
7. Potential business impact
8. Priority: Low / Medium / High / Critical

Do not claim the dataset is unusable only because ML anomalies exist.
"""

                response = model.generate_content(
                    prompt
                )

                st.markdown(
                    response.text
                )

            except Exception as e:

                st.error(
                    f"AI analysis failed: {e}"
                )


# ============================================================
# POWER BI
# ============================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">Power BI-Ready Export</div>',
        unsafe_allow_html=True,
    )

    cleaned = clean_dataset(
        df
    )

    dictionary = data_dictionary(
        cleaned
    )

    quality = pd.DataFrame({
        "Metric": [
            "Rows",
            "Columns",
            "Missing Cells",
            "Duplicate Rows",
            "Invalid Values",
            "City Issue Rows",
            "IQR Outlier Rows",
            "ML Anomaly Rows",
            "ML Anomaly Rate %",
            "Quality Score",
        ],
        "Value": [
            rows,
            columns,
            missing,
            duplicates,
            invalid_count,
            cities["rows"],
            outlier_count,
            anomaly_count,
            round(
                anomaly_rate,
                2,
            ),
            score,
        ],
    })

    numeric = numeric_summary(
        cleaned
    )

    missing_clean = missing_summary(
        cleaned
    )

    files = {
        "cleaned_data.csv":
            cleaned.to_csv(
                index=False
            ),

        "data_dictionary.csv":
            dictionary.to_csv(
                index=False
            ),

        "quality_summary.csv":
            quality.to_csv(
                index=False
            ),

        "numeric_summary.csv":
            numeric.to_csv(
                index=False
            ),

        "missing_summary.csv":
            missing_clean.to_csv(
                index=False
            ),
    }

    st.markdown(
        """
        <div class="status-good">
            <div class="status-title">
                ✓ Power BI package ready
            </div>
            <div class="status-text">
                Your cleaned dataset and supporting quality reports
                are ready for Power BI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    for filename in files:

        st.write(
            f"📄 {filename}"
        )

    st.download_button(
        "⬇️ Download Cleaned Data",
        data=files[
            "cleaned_data.csv"
        ].encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Download Quality Summary",
        data=files[
            "quality_summary.csv"
        ].encode("utf-8"),
        file_name="quality_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "📦 Download Complete Power BI Package",
        data=zip_package(
            files
        ),
        file_name="dataguard_powerbi_package.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ============================================================
# FINAL REPORT
# ============================================================

with tabs[7]:

    st.markdown(
        '<div class="section-title">Final Data Quality Report</div>',
        unsafe_allow_html=True,
    )

    report = pd.DataFrame({
        "Metric": [
            "Dataset",
            "Generated At",
            "Rows",
            "Columns",
            "Missing Cells",
            "Duplicate Rows",
            "Invalid Values",
            "City Issue Rows",
            "IQR Outlier Rows",
            "ML Anomaly Rows",
            "ML Anomaly Rate",
            "Quality Score",
            "Quality Status",
        ],
        "Value": [
            uploaded_file.name,
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            rows,
            columns,
            missing,
            duplicates,
            invalid_count,
            cities["rows"],
            outlier_count,
            anomaly_count,
            f"{anomaly_rate:.2f}%",
            score,
            score_text,
        ],
    })

    st.dataframe(
        report,
        use_container_width=True,
    )

    report_text = f"""
DATAGUARD AI
DATA QUALITY REPORT
===================

Dataset:
{uploaded_file.name}

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DATASET
-------
Rows: {rows}
Columns: {columns}

DATA QUALITY
------------
Missing Cells: {missing}
Duplicate Rows: {duplicates}
Invalid Values: {invalid_count}
City Issue Rows: {cities["rows"]}
IQR Outlier Rows: {outlier_count}

MACHINE LEARNING
----------------
Sensitivity: {anomaly_sensitivity}%
ML Anomaly Rows: {anomaly_count}
ML Anomaly Rate: {anomaly_rate:.2f}%

QUALITY
-------
Score: {score}/100
Status: {score_text}

CLEANING
--------
Rows Before: {len(df)}
Rows After: {len(cleaned)}

Missing Before: {before_missing if 'before_missing' in locals() else missing}
Missing After: {after_missing if 'after_missing' in locals() else 'N/A'}

NOTE
----
Isolation Forest anomaly flags are screening indicators.
They should be investigated before being treated as confirmed
data-quality errors.
"""

    st.download_button(
        "⬇️ Download Final Report",
        data=report_text.encode(
            "utf-8"
        ),
        file_name="dataguard_final_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🛡️ DataGuard AI • Data Quality • Anomaly Detection • Power BI Ready'
    '</div>',
    unsafe_allow_html=True,
)
