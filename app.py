import io
import os
import zipfile

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

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""

if "uploaded_file_type" not in st.session_state:
    st.session_state.uploaded_file_type = ""

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_applied" not in st.session_state:
    st.session_state.cleaning_applied = False

if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = 5


# ============================================================
# PREMIUM DARK UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(99,102,241,0.08),
                transparent 28%
            ),
            #070b14;
        color: #e5e7eb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p, label, span, div {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background: #0b1020 !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 23px !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.1rem !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }

    [data-testid="stSidebar"] p strong {
        color: #64748b !important;
        font-size: 11px !important;
        letter-spacing: 1.3px !important;
    }

    /* FIX SIDEBAR BUTTON TEXT */

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        color: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        min-height: 42px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        text-align: left !important;
        padding: 8px 12px !important;
        margin: 3px 0 !important;
        transition: all 0.18s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.09) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 14px 0 !important;
    }

    /* Sidebar status */

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(34,197,94,0.07) !important;
        border: 1px solid rgba(34,197,94,0.18) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #86efac !important;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    .dg-card {
        background:
            linear-gradient(
                145deg,
                rgba(20,27,45,0.96),
                rgba(12,17,30,0.96)
            );
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 20px;
        box-shadow:
            0 12px 35px rgba(0,0,0,0.18);
    }

    .dg-card-title {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dg-card-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* ======================================================
       KPI
       ====================================================== */

    .kpi {
        background:
            linear-gradient(
                145deg,
                rgba(18,25,42,0.98),
                rgba(11,16,29,0.98)
            );
        border: 1px solid rgba(148,163,184,0.11);
        border-radius: 15px;
        padding: 18px;
        min-height: 115px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 750;
        line-height: 1.1;
    }

    .kpi-note {
        color: #64748b;
        font-size: 11px;
        margin-top: 8px;
    }


    /* ======================================================
       UPLOAD
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(15,23,42,0.45);
        border: 1px dashed rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 12px;
        overflow: hidden;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .dg-footer {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 40px;
    }


    /* ======================================================
       BADGES
       ====================================================== */

    .badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.20);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read CSV/XLSX/XLS into a pandas DataFrame."""

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(include=["object", "category", "string"]).columns.tolist()


def datetime_columns(df):
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def invalid_numeric_values(df):
    """
    Detect values that appear to be numeric columns
    but contain non-numeric values.
    """

    invalid = 0

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        non_null = series.dropna()

        if len(non_null) == 0:
            continue

        converted = pd.to_numeric(
            non_null,
            errors="coerce"
        )

        numeric_ratio = converted.notna().mean()

        if numeric_ratio >= 0.80:
            invalid += int(converted.isna().sum())

    return invalid


def quality_metrics(df):

    rows = len(df)
    columns = len(df.columns)

    missing = int(df.isna().sum().sum())

    duplicates = int(
        df.duplicated().sum()
    )

    invalid = invalid_numeric_values(df)

    if rows == 0 or columns == 0:
        score = 0
    else:

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
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def iqr_outliers(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return 0

    total = 0

    for column in numeric.columns:

        series = numeric[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total += int(
            ((series < lower) | (series > upper)).sum()
        )

    return total


def isolation_forest(df, sensitivity=5):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return 0, 0, pd.Series(dtype=bool)

    working = numeric.copy()

    working = working.replace(
        [np.inf, -np.inf],
        np.nan
    )

    working = working.fillna(
        working.median(numeric_only=True)
    )

    working = working.fillna(0)

    contamination = max(
        0.005,
        min(0.20, sensitivity / 100)
    )

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(working)

    anomaly_mask = predictions == -1

    anomalies = int(
        anomaly_mask.sum()
    )

    normal = int(
        (~anomaly_mask).sum()
    )

    return normal, anomalies, pd.Series(
        anomaly_mask,
        index=df.index
    )


def find_column(df, possible_names):

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        key = name.lower()

        if key in normalized:
            return normalized[key]

    return None


def clean_dataset(df):

    cleaned = df.copy()

    rows_before = len(cleaned)

    duplicates_removed = int(
        cleaned.duplicated().sum()
    )

    cleaned = cleaned.drop_duplicates()

    numeric_cols = numeric_columns(cleaned)

    filled_values = 0

    for column in numeric_cols:

        missing_count = int(
            cleaned[column].isna().sum()
        )

        if missing_count > 0:

            median_value = cleaned[column].median()

            cleaned[column] = cleaned[column].fillna(
                median_value
            )

            filled_values += missing_count

    categorical_cols = categorical_columns(cleaned)

    for column in categorical_cols:

        missing_count = int(
            cleaned[column].isna().sum()
        )

        if missing_count > 0:

            mode = cleaned[column].mode()

            if not mode.empty:

                cleaned[column] = cleaned[column].fillna(
                    mode.iloc[0]
                )

    city_col = find_column(
        cleaned,
        ["City", "city"]
    )

    city_standardized = 0

    if city_col:

        before = cleaned[city_col].copy()

        cleaned[city_col] = (
            cleaned[city_col]
            .astype(str)
            .str.strip()
            .str.title()
        )

        city_standardized = int(
            (before.astype(str) != cleaned[city_col]).sum()
        )

    rows_after = len(cleaned)

    return cleaned, {
        "rows_removed": rows_before - rows_after,
        "duplicates_removed": duplicates_removed,
        "values_filled": filled_values,
        "city_standardized": city_standardized,
    }


def business_metrics(df):

    metrics = {}

    sales_col = find_column(
        df,
        ["Sales", "sales", "Revenue", "revenue"]
    )

    profit_col = find_column(
        df,
        ["Profit", "profit"]
    )

    quantity_col = find_column(
        df,
        ["Quantity", "quantity"]
    )

    if sales_col:

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        )

        metrics["Total Sales"] = round(
            sales.sum(),
            2
        )

        metrics["Average Sale"] = round(
            sales.mean(),
            2
        )

        metrics["Highest Sale"] = round(
            sales.max(),
            2
        )

    if profit_col and sales_col:

        profit = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        )

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        )

        if sales.sum() != 0:

            metrics["Profit Margin"] = round(
                (profit.sum() / sales.sum()) * 100,
                2
            )

    if quantity_col:

        quantity = pd.to_numeric(
            df[quantity_col],
            errors="coerce"
        )

        metrics["Total Quantity"] = round(
            quantity.sum(),
            2
        )

    return metrics


def create_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Data"
        )

    output.seek(0)

    return output


def create_report_excel(df):

    quality = quality_metrics(df)

    outliers = iqr_outliers(df)

    normal, anomalies, _ = isolation_forest(
        df,
        st.session_state.sensitivity
    )

    business = business_metrics(df)

    report = pd.DataFrame(
        {
            "Metric": [
                "Dataset Rows",
                "Dataset Columns",
                "Quality Score",
                "Missing Values",
                "Duplicate Rows",
                "Invalid Values",
                "IQR Outliers",
                "Normal Records",
                "ML Anomalies",
                "Anomaly Rate",
            ],
            "Value": [
                quality["rows"],
                quality["columns"],
                quality["score"],
                quality["missing"],
                quality["duplicates"],
                quality["invalid"],
                outliers,
                normal,
                anomalies,
                round(
                    anomalies / len(df) * 100,
                    2
                ) if len(df) else 0,
            ],
        }
    )

    business_df = pd.DataFrame(
        list(business.items()),
        columns=["Metric", "Value"]
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        report.to_excel(
            writer,
            index=False,
            sheet_name="Quality Report"
        )

        business_df.to_excel(
            writer,
            index=False,
            sheet_name="Business Metrics"
        )

        df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset"
        )

    output.seek(0)

    return output


def create_zip(df):

    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:

        cleaned = st.session_state.cleaned_df

        if cleaned is not None:

            cleaned_csv = cleaned.to_csv(
                index=False
            )

            z.writestr(
                "cleaned_dataset.csv",
                cleaned_csv
            )

        report = create_report_excel(df)

        z.writestr(
            "data_quality_report.xlsx",
            report.getvalue()
        )

    output.seek(0)

    return output


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
            "Please add GEMINI_API_KEY to Streamlit Secrets."
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

DATASET
Dataset name: {dataset_name}
Rows: {len(df)}
Columns: {len(df.columns)}

DATA QUALITY
Quality score: {quality["score"]}%
Missing values: {quality["missing"]}
Duplicate rows: {quality["duplicates"]}
Invalid values: {quality["invalid"]}

STATISTICAL ANALYSIS
IQR outliers: {outliers}

MACHINE LEARNING ANOMALIES
Normal records: {normal}
Anomalies: {anomalies}
Sensitivity: {sensitivity}%

BUSINESS METRICS
{business}

IMPORTANT RULES

1. Use only the numbers supplied above.
2. Do not invent numbers.
3. Do not change any measured number.
4. IQR outliers are statistical signals.
5. Isolation Forest anomalies are screening signals.
6. An anomaly does NOT automatically mean an incorrect record.
7. Clearly distinguish confirmed data-quality problems from statistical signals.
8. Give practical recommendations.
9. Do not claim that a value is wrong unless the supplied data confirms it.

Use exactly these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Recommendations
7. Business Impact
8. Power BI Recommendations
"""

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return interaction.output_text

    except Exception as error:

        return (
            "Gemini analysis failed.\n\n"
            f"Error: {error}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ DataGuard AI"
    )

    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    # Dashboard
    if st.button(
        "⌂  Dashboard",
        key="nav_dashboard",
        use_container_width=True
    ):

        st.session_state.page = "Dashboard"
        st.rerun()

    st.markdown(
        "**WORKSPACE**"
    )

    navigation = [
        ("01", "Upload Data"),
        ("02", "Data Profile"),
        ("03", "Quality Checks"),
        ("04", "Anomaly Detection"),
        ("05", "Data Cleaning"),
        ("06", "Analytics"),
        ("07", "Power BI"),
        ("08", "AI Analysis"),
    ]

    for number, page_name in navigation:

        label = f"{number}  {page_name}"

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True
        ):

            st.session_state.page = page_name
            st.rerun()

    st.divider()

    st.markdown(
        "**SYSTEM**"
    )

    if st.button(
        "⚙️  Settings",
        key="settings",
        use_container_width=True
    ):

        st.session_state.page = "Settings"
        st.rerun()

    if st.button(
        "❓  Help",
        key="help",
        use_container_width=True
    ):

        st.session_state.page = "Help"
        st.rerun()

    st.divider()

    st.success(
        "● All systems operational"
    )


# ============================================================
# SAMPLE DATA
# ============================================================

if (
    st.session_state.df is None
    and os.path.exists("sales_data_raw.csv")
):

    try:

        st.session_state.df = pd.read_csv(
            "sales_data_raw.csv"
        )

        st.session_state.dataset_name = (
            "sales_data_raw.csv"
        )

        st.session_state.uploaded_file_type = "CSV"

    except Exception:
        pass


# ============================================================
# WORKFLOW
# ============================================================

workflow = [
    "01 Upload",
    "02 Profile",
    "03 Quality",
    "04 Anomalies",
    "05 Clean",
    "06 Analytics",
    "07 Power BI",
    "08 AI",
]

page_to_step = {
    "Upload Data": 0,
    "Data Profile": 1,
    "Quality Checks": 2,
    "Anomaly Detection": 3,
    "Data Cleaning": 4,
    "Analytics": 5,
    "Power BI": 6,
    "AI Analysis": 7,
}

current_step = page_to_step.get(
    st.session_state.page,
    0
)

if st.session_state.page != "Dashboard":

    cols = st.columns(8)

    for i, step in enumerate(workflow):

        with cols[i]:

            if i < current_step:

                st.success(
                    f"✓ {step}"
                )

            elif i == current_step:

                st.info(
                    f"● {step}"
                )

            else:

                st.caption(
                    step
                )


# ============================================================
# DATA
# ============================================================

df = st.session_state.df


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.title(
        "Dashboard",
        anchor=False
    )

    if df is None:

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Welcome to DataGuard AI
            </div>

            <div class="dg-card-subtitle">
            Upload a dataset to begin automated data quality,
            anomaly detection and analytics.
            </div>

            <span class="badge">
            AI DATA QUALITY PLATFORM
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        quality = quality_metrics(df)

        normal, anomalies, _ = isolation_forest(
            df,
            st.session_state.sensitivity
        )

        st.caption(
            f"Dataset: {st.session_state.dataset_name}"
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        metrics = [
            (
                c1,
                "Data Quality Score",
                f'{quality["score"]:.2f}%',
                "Overall quality"
            ),
            (
                c2,
                "Total Records",
                f'{quality["rows"]:,}',
                "Rows"
            ),
            (
                c3,
                "Columns",
                f'{quality["columns"]:,}',
                "Features"
            ),
            (
                c4,
                "Missing Values",
                f'{quality["missing"]:,}',
                "Requires review"
            ),
            (
                c5,
                "Duplicates",
                f'{quality["duplicates"]:,}',
                "Duplicate rows"
            ),
            (
                c6,
                "Anomalies",
                f'{anomalies:,}',
                "ML screening"
            ),
        ]

        for col, label, value, note in metrics:

            with col:

                st.markdown(
                    f"""
                    <div class="kpi">

                    <div class="kpi-label">
                    {label}
                    </div>

                    <div class="kpi-value">
                    {value}
                    </div>

                    <div class="kpi-note">
                    {note}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        left, right = st.columns(2)

        with left:

            st.markdown(
                """
                <div class="dg-card">

                <div class="dg-card-title">
                Dataset Health
                </div>

                <div class="dg-card-subtitle">
                Current quality indicators
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            health_df = pd.DataFrame(
                {
                    "Dimension": [
                        "Completeness",
                        "Consistency",
                        "Uniqueness",
                        "Validity",
                    ],
                    "Score": [
                        max(
                            0,
                            round(
                                100
                                - (
                                    quality["missing"]
                                    / max(1, quality["rows"] * quality["columns"])
                                )
                                * 100,
                                2
                            )
                        ),
                        max(
                            0,
                            round(
                                100
                                - (
                                    quality["invalid"]
                                    / max(1, quality["rows"])
                                )
                                * 100,
                                2
                            )
                        ),
                        max(
                            0,
                            round(
                                100
                                - (
                                    quality["duplicates"]
                                    / max(1, quality["rows"])
                                )
                                * 100,
                                2
                            )
                        ),
                        max(
                            0,
                            round(
                                100
                                - (
                                    quality["invalid"]
                                    / max(1, quality["rows"])
                                )
                                * 100,
                                2
                            )
                        ),
                    ],
                }
            )

            st.bar_chart(
                health_df.set_index("Dimension")
            )

        with right:

            st.markdown(
                """
                <div class="dg-card">

                <div class="dg-card-title">
                Anomaly Detection
                </div>

                <div class="dg-card-subtitle">
                Isolation Forest screening
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)

            with a:
                st.metric(
                    "Normal",
                    f"{normal:,}"
                )

            with b:
                st.metric(
                    "Anomalies",
                    f"{anomalies:,}"
                )

            with c:
                rate = (
                    anomalies / len(df) * 100
                    if len(df)
                    else 0
                )

                st.metric(
                    "Rate",
                    f"{rate:.2f}%"
                )

            st.caption(
                "Anomalies are screening signals and should "
                "be reviewed before being treated as incorrect data."
            )


# ============================================================
# UPLOAD DATA
# ============================================================

elif st.session_state.page == "Upload Data":

    st.title(
        "Upload Data",
        anchor=False
    )

    st.caption(
        "Upload CSV, XLSX or XLS files for data quality analysis."
    )

    st.write("")

    st.markdown(
        """
        <div class="dg-card">

        <div class="dg-card-title">
        Choose your dataset
        </div>

        <div class="dg-card-subtitle">
        Supported formats: CSV, XLSX, XLS • Maximum size: 50 MB
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop your dataset here or browse files",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        key="dataset_uploader"
    )

    if uploaded_file is not None:

        try:

            loaded_df = read_uploaded_file(
                uploaded_file
            )

            st.session_state.df = loaded_df
            st.session_state.dataset_name = (
                uploaded_file.name
            )
            st.session_state.uploaded_file_type = (
                uploaded_file.name.split(".")[-1].upper()
            )

            st.session_state.cleaned_df = None
            st.session_state.cleaning_applied = False

            st.success(
                f"Successfully loaded {uploaded_file.name}"
            )

        except Exception as error:

            st.error(
                f"Could not read the file: {error}"
            )

    df = st.session_state.df

    if df is not None:

        st.write("")

        st.markdown(
            f"""
            <div class="dg-card">

            <div class="dg-card-title">
            {st.session_state.dataset_name}
            </div>

            <div class="dg-card-subtitle">
            Dataset successfully loaded and ready for analysis.
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
                "File Type",
                st.session_state.uploaded_file_type
            )

        with c4:
            st.metric(
                "Memory",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            )

        st.write("")

        # IMPORTANT:
        # This is the only intentional dataframe on Upload page.
        # The previous stray dataframe was likely coming from
        # an accidental st.write(df.iloc[...]) or st.dataframe(...).

        with st.expander(
            "Preview dataset",
            expanded=False
        ):

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Dataset is ready. Continue to Data Profile to inspect "
            "columns, data types and distributions."
        )


# ============================================================
# DATA PROFILE
# ============================================================

elif st.session_state.page == "Data Profile":

    st.title(
        "Data Profile",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        profile = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(df[column].dtype)
                    for column in df.columns
                ],
                "Non-Null": [
                    int(df[column].notna().sum())
                    for column in df.columns
                ],
                "Missing": [
                    int(df[column].isna().sum())
                    for column in df.columns
                ],
                "Unique": [
                    int(df[column].nunique())
                    for column in df.columns
                ],
            }
        )

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

elif st.session_state.page == "Quality Checks":

    st.title(
        "Quality Checks",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        quality = quality_metrics(df)

        c1, c2, c3, c4 = st.columns(4)

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

        st.write("")

        quality_table = pd.DataFrame(
            {
                "Check": [
                    "Completeness",
                    "Duplicates",
                    "Invalid Values",
                    "IQR Outliers",
                ],
                "Result": [
                    quality["missing"],
                    quality["duplicates"],
                    quality["invalid"],
                    iqr_outliers(df),
                ],
            }
        )

        st.dataframe(
            quality_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif st.session_state.page == "Anomaly Detection":

    st.title(
        "Anomaly Detection",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        sensitivity = st.slider(
            "Isolation Forest sensitivity",
            min_value=1,
            max_value=20,
            value=st.session_state.sensitivity
        )

        st.session_state.sensitivity = sensitivity

        normal, anomalies, mask = isolation_forest(
            df,
            sensitivity
        )

        rate = (
            anomalies / len(df) * 100
            if len(df)
            else 0
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Normal",
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

        st.info(
            "Isolation Forest identifies unusual patterns. "
            "These are screening signals, not automatic proof of bad data."
        )

        anomaly_df = df.loc[mask]

        st.subheader(
            "Detected Anomalies"
        )

        st.dataframe(
            anomaly_df.head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DATA CLEANING
# ============================================================

elif st.session_state.page == "Data Cleaning":

    st.title(
        "Data Cleaning",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        cleaned, stats = clean_dataset(
            df
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Rows Removed",
                f'{stats["rows_removed"]:,}'
            )

        with c2:
            st.metric(
                "Values Filled",
                f'{stats["values_filled"]:,}'
            )

        with c3:
            st.metric(
                "Duplicates Removed",
                f'{stats["duplicates_removed"]:,}'
            )

        with c4:
            st.metric(
                "City Standardized",
                f'{stats["city_standardized"]:,}'
            )

        st.write("")

        if st.button(
            "Apply Cleaning",
            type="primary"
        ):

            st.session_state.cleaned_df = cleaned
            st.session_state.cleaning_applied = True

            st.success(
                "Cleaning applied successfully."
            )

        if st.session_state.cleaned_df is not None:

            st.subheader(
                "Cleaned Dataset Preview"
            )

            st.dataframe(
                st.session_state.cleaned_df.head(20),
                use_container_width=True,
                hide_index=True
            )

            cleaned_file = create_excel(
                st.session_state.cleaned_df
            )

            st.download_button(
                "Download Cleaned Excel",
                data=cleaned_file,
                file_name="dataguard_cleaned_data.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


# ============================================================
# ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":

    st.title(
        "Analytics",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        metrics = business_metrics(df)

        if metrics:

            columns = st.columns(
                min(len(metrics), 4)
            )

            for col, (label, value) in zip(
                columns,
                metrics.items()
            ):

                with col:

                    if isinstance(value, (int, float)):

                        if "Margin" in label:

                            st.metric(
                                label,
                                f"{value:.2f}%"
                            )

                        else:

                            st.metric(
                                label,
                                f"{value:,.2f}"
                            )

                    else:

                        st.metric(
                            label,
                            value
                        )

        st.write("")

        numeric = numeric_columns(df)

        if numeric:

            selected_column = st.selectbox(
                "Select numeric column",
                numeric
            )

            st.subheader(
                f"{selected_column} Distribution"
            )

            chart_df = df[
                [selected_column]
            ].dropna()

            st.bar_chart(
                chart_df
                .head(100)
                .reset_index(drop=True)
            )

        else:

            st.info(
                "No numeric columns available for analytics."
            )


# ============================================================
# POWER BI
# ============================================================

elif st.session_state.page == "Power BI":

    st.title(
        "Power BI",
        anchor=False
    )

    st.markdown(
        """
        <div class="dg-card">

        <div class="dg-card-title">
        Power BI Integration
        </div>

        <div class="dg-card-subtitle">
        Prepare your cleaned dataset for Power BI reporting.
        </div>

        <span class="badge">
        POWER BI READY
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    if df is not None:

        export_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        csv_data = export_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Power BI Dataset",
            data=csv_data,
            file_name="dataguard_powerbi_dataset.csv",
            mime="text/csv"
        )

        st.info(
            "Import this CSV into Power BI Desktop and build "
            "your final business dashboard."
        )


# ============================================================
# AI ANALYSIS
# ============================================================

elif st.session_state.page == "AI Analysis":

    st.title(
        "AI Analysis",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        quality = quality_metrics(
            df
        )

        outliers = iqr_outliers(
            df
        )

        normal, anomalies, _ = isolation_forest(
            df,
            st.session_state.sensitivity
        )

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Gemini Data Intelligence
            </div>

            <div class="dg-card-subtitle">
            AI-generated interpretation of measured data quality
            and business signals.
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
                "Gemini is analyzing your dataset..."
            ):

                analysis = generate_gemini_analysis(
                    st.session_state.dataset_name,
                    df,
                    quality,
                    outliers,
                    normal,
                    anomalies,
                    st.session_state.sensitivity
                )

            st.markdown(
                analysis
            )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.title(
        "Settings",
        anchor=False
    )

    st.subheader(
        "Detection Settings"
    )

    sensitivity = st.slider(
        "Isolation Forest sensitivity",
        1,
        20,
        st.session_state.sensitivity
    )

    st.session_state.sensitivity = sensitivity

    st.caption(
        "Higher sensitivity identifies more records as potential anomalies."
    )

    st.subheader(
        "Dataset"
    )

    if df is not None:

        st.write(
            f"Current dataset: **{st.session_state.dataset_name}**"
        )

    else:

        st.write(
            "No dataset loaded."
        )


# ============================================================
# HELP
# ============================================================

elif st.session_state.page == "Help":

    st.title(
        "Help",
        anchor=False
    )

    st.markdown(
        """
        ### How DataGuard AI works

        **01 Upload**
        Upload your CSV or Excel dataset.

        **02 Profile**
        Inspect columns, data types, missing values and uniqueness.

        **03 Quality**
        Identify missing values, duplicates, invalid values and outliers.

        **04 Anomalies**
        Use Isolation Forest to identify unusual records.

        **05 Clean**
        Remove duplicates and fill missing values.

        **06 Analytics**
        Explore important business metrics.

        **07 Power BI**
        Export the prepared dataset for Power BI.

        **08 AI**
        Generate an AI-assisted data quality and business analysis.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dg-footer">
    🛡️ DataGuard AI
    &nbsp;•&nbsp;
    Portfolio Edition
    &nbsp;•&nbsp;
    Python
    &nbsp;•&nbsp;
    Pandas
    &nbsp;•&nbsp;
    Scikit-learn
    &nbsp;•&nbsp;
    Streamlit
    &nbsp;•&nbsp;
    Gemini
    </div>
    """,
    unsafe_allow_html=True
)
