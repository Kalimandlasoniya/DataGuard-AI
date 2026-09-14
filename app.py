import io
import os
import zipfile
import html

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

    /* =========================
       GLOBAL
       ========================= */

    .stApp {
        background: #080b12;
        color: #e5e7eb;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background: #0b0f17;
        border-right: 1px solid #1f2937;
    }

    .brand {
        padding: 8px 4px 24px 4px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #2563eb;
        font-size: 23px;
        margin-bottom: 10px;
    }

    .brand-title {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 800;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 12px;
        margin-top: 3px;
    }

    .sidebar-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .system-box {
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 12px;
        margin-top: 18px;
        background: #0f141d;
    }

    .system-line {
        color: #cbd5e1;
        font-size: 12px;
    }

    .green-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 7px;
    }

    /* =========================
       HEADER
       ========================= */

    .eyebrow {
        color: #64748b;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 31px;
        font-weight: 850;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 7px;
    }

    .dataset-bar {
        border: 1px solid #1e293b;
        background: #0d121b;
        border-radius: 14px;
        padding: 13px 16px;
        margin: 18px 0;
    }

    .dataset-text {
        color: #cbd5e1;
        font-size: 13px;
        font-weight: 650;
    }

    /* =========================
       WORKFLOW
       ========================= */

    .workflow {
        display: flex;
        gap: 7px;
        align-items: center;
        overflow-x: auto;
        border: 1px solid #1e293b;
        background: #0d121b;
        border-radius: 14px;
        padding: 10px;
        margin-bottom: 22px;
    }

    .workflow-step {
        min-width: 95px;
        padding: 8px;
        text-align: center;
        border-radius: 9px;
        color: #64748b;
        font-size: 10px;
    }

    .workflow-step.active {
        background: #172554;
        color: #93c5fd;
        border: 1px solid #2563eb;
    }

    .workflow-number {
        display: block;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .workflow-name {
        font-weight: 700;
    }

    .workflow-arrow {
        color: #334155;
    }

    /* =========================
       CARDS
       ========================= */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #0d121b;
        border-color: #1e293b;
        border-radius: 15px;
    }

    /* =========================
       KPI
       ========================= */

    div[data-testid="stMetric"] {
        background: #0d121b;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* =========================
       BUTTONS
       ========================= */

    .stButton > button {
        min-height: 40px;
        border-radius: 9px;
        border: 1px solid #334155;
        background: #111827;
        color: #e2e8f0;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #3b82f6;
        color: #93c5fd;
    }

    /* =========================
       TABLE
       ========================= */

    .stDataFrame {
        border-radius: 12px;
        border: 1px solid #1e293b;
    }

    /* =========================
       UPLOAD
       ========================= */

    .upload-box {
        text-align: center;
        border: 1px dashed #334155;
        background: #0d121b;
        border-radius: 16px;
        padding: 35px;
        margin-bottom: 15px;
    }

    .upload-icon {
        font-size: 40px;
    }

    .upload-title {
        color: #f8fafc;
        font-size: 19px;
        font-weight: 800;
        margin-top: 8px;
    }

    .upload-text {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* =========================
       FOOTER
       ========================= */

    .footer {
        text-align: center;
        color: #475569;
        font-size: 11px;
        border-top: 1px solid #1e293b;
        padding-top: 18px;
        margin-top: 40px;
    }

    /* =========================
       MOBILE
       ========================= */

    @media(max-width: 700px) {

        .hero-title {
            font-size: 24px;
        }

        .workflow-step {
            min-width: 75px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
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
    "chart_df": None,
    "file_name": None,
    "analysis": None,
    "analysis_complete": False,
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
# DATA FUNCTIONS
# ============================================================

def safe_to_datetime(series):

    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            format="mixed"
        )

    except Exception:

        try:
            return pd.to_datetime(
                series,
                errors="coerce"
            )

        except Exception:

            return pd.Series(
                pd.NaT,
                index=series.index
            )


def detect_datetime_columns(df):

    detected = []

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

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[col]
        ):
            detected.append(col)
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        name = str(col).lower()

        parsed = safe_to_datetime(
            df[col]
        )

        ratio = parsed.notna().mean()

        if any(
            token in name
            for token in tokens
        ):

            if ratio >= 0.50:
                detected.append(col)

        elif ratio >= 0.95:

            detected.append(col)

    return detected


def detect_identifier_columns(
    df,
    datetime_columns=None
):

    datetime_columns = (
        datetime_columns or []
    )

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

    identifiers = []

    for col in df.columns:

        if col in datetime_columns:
            continue

        name = str(col).lower()

        if any(
            token in name
            for token in tokens
        ):

            identifiers.append(col)
            continue

        if (
            pd.api.types.is_integer_dtype(
                df[col]
            )
            or
            pd.api.types.is_object_dtype(
                df[col]
            )
            or
            pd.api.types.is_string_dtype(
                df[col]
            )
        ):

            unique_ratio = (
                df[col].nunique(
                    dropna=True
                )
                / max(len(df), 1)
            )

            if unique_ratio >= 0.98:
                identifiers.append(col)

    return identifiers


def profile_dataset(df):

    rows = []

    for col in df.columns:

        rows.append(
            {
                "Column": col,
                "Data Type": str(
                    df[col].dtype
                ),
                "Non-Null Count": int(
                    df[col].notna().sum()
                ),
                "Missing": int(
                    df[col].isna().sum()
                ),
                "Unique Values": int(
                    df[col].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def detect_invalid_values(df):

    invalid_count = 0
    details = []

    tokens = [
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

    for col in df.columns:

        name = str(col).lower()

        if not any(
            token in name
            for token in tokens
        ):
            continue

        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        conversion_invalid = int(
            (
                df[col].notna()
                & numeric.isna()
            ).sum()
        )

        negative_invalid = 0

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

            negative_invalid = int(
                (numeric < 0).sum()
            )

        total = (
            conversion_invalid
            + negative_invalid
        )

        if total > 0:

            invalid_count += total

            details.append(
                {
                    "Column": col,
                    "Invalid Values": total,
                }
            )

    return (
        invalid_count,
        pd.DataFrame(details)
    )


def standardize_city_column(df):

    result = df.copy()
    total_changes = 0

    city_columns = [
        col
        for col in result.columns
        if "city" in str(col).lower()
    ]

    for col in city_columns:

        original = (
            result[col]
            .astype(str)
        )

        def normalize(value):

            if pd.isna(value):
                return value

            value = str(
                value
            ).strip()

            if not value:
                return value

            key = value.lower()

            if key in CITY_MAPPING:
                return CITY_MAPPING[key]

            return value.title()

        result[col] = (
            result[col]
            .apply(normalize)
        )

        changed = (
            original
            != result[col].astype(str)
        ).sum()

        total_changes += int(
            changed
        )

    return (
        result,
        total_changes
    )


def detect_iqr_outliers(df):

    total = 0
    details = []

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    for col in numeric_columns:

        series = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
            .dropna()
        )

        if len(series) < 4:
            continue

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (series < lower)
                | (series > upper)
            ).sum()
        )

        if count > 0:

            total += count

            details.append(
                {
                    "Column": col,
                    "Q1": round(
                        q1,
                        2
                    ),
                    "Q3": round(
                        q3,
                        2
                    ),
                    "Lower Bound": round(
                        lower,
                        2
                    ),
                    "Upper Bound": round(
                        upper,
                        2
                    ),
                    "Outliers": count,
                }
            )

    return (
        total,
        pd.DataFrame(details)
    )


def detect_ml_anomalies(
    df,
    contamination_pct
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
        .copy()
    )

    if (
        numeric_df.empty
        or len(df) < 10
    ):

        return (
            pd.Series(
                False,
                index=df.index
            ),
            0
        )

    numeric_df = (
        numeric_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    numeric_df = (
        numeric_df
        .fillna(
            numeric_df.median()
        )
        .fillna(0)
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=(
            contamination_pct / 100
        ),
        random_state=42,
    )

    predictions = (
        model.fit_predict(
            numeric_df
        )
    )

    mask = (
        predictions == -1
    )

    return (
        pd.Series(
            mask,
            index=df.index
        ),
        int(mask.sum())
    )


def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count
):

    if total_cells <= 0:
        return 0

    missing_penalty = (
        missing_count
        / total_cells
        * 100
        * 0.50
    )

    duplicate_penalty = (
        duplicate_count
        / max(total_cells, 1)
        * 100
        * 0.30
    )

    invalid_penalty = (
        invalid_count
        / max(total_cells, 1)
        * 100
        * 0.20
    )

    score = (
        100
        - missing_penalty
        - duplicate_penalty
        - invalid_penalty
    )

    return round(
        max(
            0,
            min(
                100,
                score
            )
        ),
        2
    )


def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(
        cleaned
    )

    cleaned = (
        cleaned
        .drop_duplicates()
    )

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

    for col in numeric_columns:

        cleaned[col] = (
            pd.to_numeric(
                cleaned[col],
                errors="coerce"
            )
            .astype("float64")
        )

        missing = int(
            cleaned[col].isna().sum()
        )

        if missing > 0:

            median = (
                cleaned[col]
                .median()
            )

            if pd.notna(median):

                cleaned[col] = (
                    cleaned[col]
                    .fillna(median)
                )

                values_filled += missing

    categorical_columns = (
        cleaned
        .select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        )
        .columns
    )

    for col in categorical_columns:

        missing = int(
            cleaned[col]
            .isna()
            .sum()
        )

        if missing > 0:

            mode = (
                cleaned[col]
                .mode()
            )

            if not mode.empty:

                cleaned[col] = (
                    cleaned[col]
                    .fillna(
                        mode.iloc[0]
                    )
                )

                values_filled += missing

    return (
        cleaned,
        rows_removed,
        values_filled,
        city_changes
    )


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

def find_sales_column(df):

    preferred = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    for name in preferred:

        for col in df.columns:

            if (
                str(col).lower()
                == name
            ):

                numeric = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                if numeric.notna().sum() > 0:
                    return col

    for col in df.columns:

        name = str(
            col
        ).lower()

        if any(
            token in name
            for token in [
                "sales",
                "revenue",
                "amount"
            ]
        ):

            numeric = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if numeric.notna().sum() > 0:
                return col

    return None


def prepare_business_data(df):

    sales_column = (
        find_sales_column(df)
    )

    if sales_column is None:

        return (
            pd.DataFrame(),
            None,
            pd.DataFrame(),
            None
        )

    business_df = df.copy()

    business_df[
        sales_column
    ] = pd.to_numeric(
        business_df[
            sales_column
        ],
        errors="coerce"
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

    chart_df = business_df.copy()

    upper_bound = None

    series = chart_df[
        sales_column
    ]

    if len(series) >= 4:

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        upper_bound = (
            q3 + 1.5 * iqr
        )

        chart_df = (
            chart_df[
                chart_df[
                    sales_column
                ] <= upper_bound
            ]
        )

    return (
        business_df,
        sales_column,
        chart_df,
        upper_bound
    )


# ============================================================
# POWER BI EXPORTS
# ============================================================

def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column
):

    exports = {}

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = (
        cleaned_df
        .to_csv(index=False)
        .encode()
    )

    exports[
        "DataGuard_Data_Profile.csv"
    ] = (
        profile_dataset(
            cleaned_df
        )
        .to_csv(index=False)
        .encode()
    )

    if (
        not business_df.empty
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
            .encode()
        )

        city_columns = [
            col
            for col in business_df.columns
            if "city" in str(col).lower()
        ]

        if city_columns:

            city_col = (
                city_columns[0]
            )

            city_summary = (
                business_df
                .groupby(
                    city_col
                )[sales_column]
                .agg(
                    Total_Sales="sum",
                    Record_Count="count"
                )
                .reset_index()
            )

            exports[
                "DataGuard_City_Summary.csv"
            ] = (
                city_summary
                .to_csv(index=False)
                .encode()
            )

        category_columns = [
            col
            for col in business_df.columns
            if "category"
            in str(col).lower()
        ]

        if category_columns:

            category_col = (
                category_columns[0]
            )

            category_summary = (
                business_df
                .groupby(
                    category_col
                )[sales_column]
                .agg(
                    Total_Sales="sum",
                    Record_Count="count"
                )
                .reset_index()
            )

            exports[
                "DataGuard_Category_Summary.csv"
            ] = (
                category_summary
                .to_csv(index=False)
                .encode()
            )

        date_columns = (
            detect_datetime_columns(
                business_df
            )
        )

        if date_columns:

            date_col = (
                date_columns[0]
            )

            temp = business_df.copy()

            temp[date_col] = (
                safe_to_datetime(
                    temp[date_col]
                )
            )

            monthly = (
                temp
                .dropna(
                    subset=[date_col]
                )
                .assign(
                    Month=lambda x:
                    x[date_col]
                    .dt
                    .to_period("M")
                    .astype(str)
                )
                .groupby(
                    "Month"
                )[sales_column]
                .sum()
                .reset_index(
                    name="Total_Sales"
                )
            )

            exports[
                "DataGuard_Monthly_Sales.csv"
            ] = (
                monthly
                .to_csv(index=False)
                .encode()
            )

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for name, content in exports.items():

            z.writestr(
                name,
                content
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI
# ============================================================

def run_gemini(report):

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
            "Gemini API key is not configured."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are an AI data quality analyst.

Analyze ONLY the measured DataGuard AI report below.

Do not invent facts.

IQR outliers are statistical signals,
not automatically confirmed errors.

Isolation Forest anomalies are screening signals,
not automatically confirmed errors.

Cleaning has already been performed.

Use these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

REPORT:

{report}
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return getattr(
            response,
            "text",
            str(response)
        )

    except Exception as exc:

        return (
            f"Gemini analysis failed: {exc}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-title">
                DataGuard AI
            </div>
            <div class="brand-subtitle">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
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
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-label">DETECTION SETTINGS</div>',
        unsafe_allow_html=True
    )

    contamination = st.slider(
        "Isolation Forest sensitivity",
        1,
        20,
        st.session_state.contamination_pct,
        help=(
            "Higher sensitivity flags more "
            "records as unusual."
        )
    )

    st.session_state.contamination_pct = (
        contamination
    )

    st.caption(
        "Higher sensitivity flags more "
        "records as unusual."
    )

    st.markdown(
        '<div class="sidebar-label">SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="system-box">
            <div class="system-line">
                <span class="green-dot"></span>
                All systems operational
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="eyebrow">DATA INTELLIGENCE WORKSPACE</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="hero-title">{page}</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-subtitle">
        AI-powered data quality, anomaly detection
        and business analytics.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD PAGE
# ============================================================

if page == "Upload Data":

    st.markdown(
        """
        <div class="upload-box">
            <div class="upload-icon">📊</div>
            <div class="upload-title">
                Upload your dataset
            </div>
            <div class="upload-text">
                CSV, XLSX or XLS • Maximum 50 MB
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        label_visibility="collapsed"
    )

else:

    uploaded_file = None


# ============================================================
# LOAD UPLOADED DATA
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

                loaded_df = pd.read_csv(
                    uploaded_file
                )

            else:

                loaded_df = pd.read_excel(
                    uploaded_file
                )

            st.session_state.df = (
                loaded_df
            )

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.chart_df = None
            st.session_state.analysis = None
            st.session_state.gemini_analysis = None
            st.session_state.analysis_complete = False

            st.success(
                f"Dataset loaded successfully: "
                f"{uploaded_file.name}"
            )

            st.rerun()

        except Exception as exc:

            st.error(
                f"Unable to load dataset: {exc}"
            )


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.df


# ============================================================
# NO DATA
# ============================================================

if df is None:

    st.info(
        "Upload a CSV or Excel dataset to begin."
    )

    st.markdown(
        """
        ### What DataGuard AI can do

        - 📋 Profile your dataset
        - 🔎 Detect missing and invalid values
        - 📊 Measure data quality
        - 📈 Detect IQR outliers
        - 🤖 Detect ML anomalies
        - 🧹 Clean the dataset
        - 💼 Analyze business metrics
        - 📊 Prepare Power BI exports
        - 🧠 Generate Gemini AI insights
        """,
    )

    st.markdown(
        """
        <div class="footer">
            🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
            <br>
            Built with Python • Pandas • Scikit-learn • Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# ANALYSIS
# ============================================================

try:

    total_rows = len(df)
    total_columns = len(df.columns)
    total_cells = (
        total_rows
        * total_columns
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
            invalid_count
        )
    )

    datetime_columns = (
        detect_datetime_columns(df)
    )

    identifier_columns = (
        detect_identifier_columns(
            df,
            datetime_columns
        )
    )

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    iqr_count, iqr_details = (
        detect_iqr_outliers(df)
    )

    anomaly_mask, anomaly_count = (
        detect_ml_anomalies(
            df,
            contamination
        )
    )

    normal_count = (
        total_rows
        - anomaly_count
    )

    (
        cleaned_df,
        rows_removed,
        values_filled,
        city_changes
    ) = clean_dataset(df)

    (
        business_df,
        sales_column,
        chart_df,
        upper_bound
    ) = prepare_business_data(
        df
    )

    st.session_state.cleaned_df = (
        cleaned_df
    )

    st.session_state.business_df = (
        business_df
    )

    st.session_state.chart_df = (
        chart_df
    )

    st.session_state.analysis = {

        "rows": total_rows,

        "columns": total_columns,

        "cells": total_cells,

        "missing": missing_count,

        "duplicates": duplicate_count,

        "invalid": invalid_count,

        "quality": quality_score,

        "datetime_columns":
            datetime_columns,

        "identifier_columns":
            identifier_columns,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "iqr_count":
            iqr_count,

        "iqr_details":
            iqr_details,

        "anomaly_mask":
            anomaly_mask,

        "anomaly_count":
            anomaly_count,

        "normal_count":
            normal_count,

        "cleaned_rows":
            len(cleaned_df),

        "rows_removed":
            rows_removed,

        "values_filled":
            values_filled,

        "city_changes":
            city_changes,

        "invalid_details":
            invalid_details,

        "business_df":
            business_df,

        "chart_df":
            chart_df,

        "sales_column":
            sales_column,

        "upper_bound":
            upper_bound,
    }

    st.session_state.analysis_complete = True

except Exception as exc:

    st.error(
        f"Analysis failed: {exc}"
    )

    st.stop()


a = st.session_state.analysis


# ============================================================
# DATASET STATUS
# ============================================================

st.markdown(
    f"""
    <div class="dataset-bar">
        <span class="dataset-text">
            📄 {html.escape(str(st.session_state.file_name))}
        </span>
        &nbsp;&nbsp;
        {a['rows']:,} rows
        •
        {a['columns']} columns
        •
        <b style="color:#86efac;">
            ✓ Ready
        </b>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WORKFLOW
# ============================================================

workflow = [
    ("01", "Upload"),
    ("02", "Profile"),
    ("03", "Quality"),
    ("04", "Anomalies"),
    ("05", "Clean"),
    ("06", "Analytics"),
    ("07", "Power BI"),
    ("08", "AI"),
]

workflow_html = ""

for number, name in workflow:

    workflow_html += f"""
        <div class="workflow-step">
            <span class="workflow-number">
                {number}
            </span>
            <span class="workflow-name">
                {name}
            </span>
        </div>
    """

    if number != "08":

        workflow_html += (
            '<span class="workflow-arrow">→</span>'
        )

st.markdown(
    f'<div class="workflow">{workflow_html}</div>',
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.subheader(
        "Dashboard"
    )

    st.caption(
        "Monitor dataset health and quality signals."
    )

    # KPI
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Data Quality Score",
        f"{a['quality']:.2f}/100"
    )

    c2.metric(
        "Missing Cells",
        f"{a['missing']:,}"
    )

    c3.metric(
        "Duplicate Records",
        f"{a['duplicates']:,}"
    )

    c4.metric(
        "ML Anomalies",
        f"{a['anomaly_count']:,}"
    )

    st.write("")

    # PROFILE
    st.subheader(
        "Dataset Profile"
    )

    p1, p2, p3, p4, p5 = (
        st.columns(5)
    )

    p1.metric(
        "Rows",
        f"{a['rows']:,}"
    )

    p2.metric(
        "Columns",
        a["columns"]
    )

    p3.metric(
        "Numeric",
        len(a["numeric_columns"])
    )

    p4.metric(
        "Categorical",
        len(a["categorical_columns"])
    )

    p5.metric(
        "Date / Time",
        len(a["datetime_columns"])
    )

    st.write("")

    # QUALITY
    left, right = st.columns(
        [1, 1.3]
    )

    with left:

        with st.container(border=True):

            st.subheader(
                "Quality Overview"
            )

            score = a["quality"]

            if score >= 95:
                status = "Excellent"

            elif score >= 85:
                status = "Good"

            elif score >= 70:
                status = "Needs Attention"

            else:
                status = "Critical"

            st.metric(
                "Overall Quality",
                f"{score:.2f}/100",
                status
            )

            st.progress(
                min(
                    score / 100,
                    1.0
                )
            )

            completeness = (
                1
                - a["missing"]
                / max(a["cells"], 1)
            ) * 100

            uniqueness = (
                1
                - a["duplicates"]
                / max(a["rows"], 1)
            ) * 100

            validity = (
                1
                - a["invalid"]
                / max(a["cells"], 1)
            ) * 100

            q1, q2, q3 = (
                st.columns(3)
            )

            q1.metric(
                "Completeness",
                f"{completeness:.1f}%"
            )

            q2.metric(
                "Uniqueness",
                f"{uniqueness:.1f}%"
            )

            q3.metric(
                "Validity",
                f"{validity:.1f}%"
            )

    with right:

        with st.container(border=True):

            st.subheader(
                "Issues Detected"
            )

            issues = pd.DataFrame(
                {
                    "Issue": [
                        "Missing Values",
                        "Duplicate Records",
                        "Invalid Values",
                        "IQR Outliers",
                        "ML Anomalies"
                    ],
                    "Count": [
                        a["missing"],
                        a["duplicates"],
                        a["invalid"],
                        a["iqr_count"],
                        a["anomaly_count"]
                    ]
                }
            )

            st.dataframe(
                issues,
                use_container_width=True,
                hide_index=True
            )

    st.write("")

    # ANOMALIES
    with st.container(border=True):

        st.subheader(
            "Anomaly Detection"
        )

        st.caption(
            f"Isolation Forest • "
            f"{contamination}% sensitivity"
        )

        n1, n2, n3 = (
            st.columns(3)
        )

        n1.metric(
            "Normal Records",
            f"{a['normal_count']:,}"
        )

        n2.metric(
            "ML Anomalies",
            f"{a['anomaly_count']:,}"
        )

        anomaly_rate = (
            a["anomaly_count"]
            / max(a["rows"], 1)
            * 100
        )

        n3.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )

        st.info(
            "ML anomalies are screening signals, "
            "not confirmed data errors."
        )

    # BUSINESS
    if a["sales_column"]:

        st.write("")

        with st.container(border=True):

            st.subheader(
                "Business Snapshot"
            )

            business_df = (
                a["business_df"]
            )

            sales_column = (
                a["sales_column"]
            )

            total_sales = (
                business_df[
                    sales_column
                ].sum()
            )

            average_sale = (
                business_df[
                    sales_column
                ].mean()
            )

            highest_sale = (
                business_df[
                    sales_column
                ].max()
            )

            b1, b2, b3, b4 = (
                st.columns(4)
            )

            b1.metric(
                "Total Sales",
                f"{total_sales:,.0f}"
            )

            b2.metric(
                "Average Sale",
                f"{average_sale:,.2f}"
            )

            b3.metric(
                "Highest Sale",
                f"{highest_sale:,.2f}"
            )

            b4.metric(
                "Sales Records",
                f"{len(business_df):,}"
            )

    # PREVIEW
    st.write("")

    with st.container(border=True):

        st.subheader(
            "Data Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Showing 10 of {len(df):,} records. "
            "All analysis uses the complete dataset."
        )


# ============================================================
# UPLOAD DATA
# ============================================================

elif page == "Upload Data":

    st.subheader(
        "Upload Data"
    )

    st.write(
        "Upload a CSV or Excel file to begin the "
        "DataGuard AI pipeline."
    )

    st.info(
        f"Current dataset: "
        f"{st.session_state.file_name}"
    )

    st.metric(
        "Current Records",
        f"{len(df):,}"
    )


# ============================================================
# DATA PROFILE
# ============================================================

elif page == "Data Profile":

    st.subheader(
        "Dataset Profile"
    )

    st.caption(
        "Detailed structural profile of the complete dataset."
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Rows",
        f"{a['rows']:,}"
    )

    c2.metric(
        "Columns",
        a["columns"]
    )

    c3.metric(
        "Memory",
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    c4.metric(
        "Duplicate Rows",
        f"{a['duplicates']:,}"
    )

    st.subheader(
        "Column Profile"
    )

    profile = profile_dataset(
        df
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    if a["datetime_columns"]:

        with st.expander(
            "Detected Date / Time Columns"
        ):

            st.write(
                a["datetime_columns"]
            )

    if a["identifier_columns"]:

        with st.expander(
            "Detected Identifier Columns"
        ):

            st.write(
                a["identifier_columns"]
            )


# ============================================================
# QUALITY CHECKS
# ============================================================

elif page == "Quality Checks":

    st.subheader(
        "Quality Checks"
    )

    st.caption(
        "Rule-based validation and statistical quality signals."
    )

    q1, q2, q3, q4 = (
        st.columns(4)
    )

    q1.metric(
        "Quality Score",
        f"{a['quality']:.2f}/100"
    )

    q2.metric(
        "Missing",
        f"{a['missing']:,}"
    )

    q3.metric(
        "Duplicates",
        f"{a['duplicates']:,}"
    )

    q4.metric(
        "Invalid",
        f"{a['invalid']:,}"
    )

    st.write("")

    completeness = (
        1
        - a["missing"]
        / max(a["cells"], 1)
    ) * 100

    uniqueness = (
        1
        - a["duplicates"]
        / max(a["rows"], 1)
    ) * 100

    validity = (
        1
        - a["invalid"]
        / max(a["cells"], 1)
    ) * 100

    cards = [
        (
            "Completeness",
            completeness
        ),
        (
            "Uniqueness",
            uniqueness
        ),
        (
            "Validity",
            validity
        ),
        (
            "Outlier Screening",
            max(
                0,
                100
                - (
                    a["iqr_count"]
                    / max(
                        a["rows"],
                        1
                    )
                    * 100
                )
            )
        )
    ]

    cols = st.columns(4)

    for col, (
        name,
        value
    ) in zip(
        cols,
        cards
    ):

        with col:

            with st.container(
                border=True
            ):

                st.subheader(
                    name
                )

                st.metric(
                    "Score",
                    f"{value:.1f}%"
                )

                if value >= 95:

                    st.success(
                        "Passed"
                    )

                elif value >= 80:

                    st.warning(
                        "Warning"
                    )

                else:

                    st.error(
                        "Needs Attention"
                    )

    st.subheader(
        "Invalid Values"
    )

    if a["invalid_details"].empty:

        st.success(
            "No invalid values detected."
        )

    else:

        st.dataframe(
            a["invalid_details"],
            use_container_width=True,
            hide_index=True
        )

    st.subheader(
        "IQR Outliers"
    )

    if a["iqr_details"].empty:

        st.success(
            "No IQR outliers detected."
        )

    else:

        st.dataframe(
            a["iqr_details"],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif page == "Anomaly Detection":

    st.subheader(
        "Anomaly Detection"
    )

    st.caption(
        "Statistical and machine-learning screening signals."
    )

    st.info(
        "IQR outliers and Isolation Forest anomalies "
        "are screening signals and are not automatically "
        "confirmed data errors."
    )

    c1, c2, c3 = (
        st.columns(3)
    )

    c1.metric(
        "IQR Outliers",
        f"{a['iqr_count']:,}"
    )

    c2.metric(
        "ML Anomalies",
        f"{a['anomaly_count']:,}"
    )

    c3.metric(
        "Normal Records",
        f"{a['normal_count']:,}"
    )

    st.write("")

    st.subheader(
        "IQR Statistical Detection"
    )

    if a["iqr_details"].empty:

        st.success(
            "No statistical outliers detected."
        )

    else:

        st.dataframe(
            a["iqr_details"],
            use_container_width=True,
            hide_index=True
        )

    st.write("")

    st.subheader(
        "Isolation Forest"
    )

    st.caption(
        f"Current sensitivity: {contamination}%. "
        "Change it from the sidebar to rerun the model."
    )

    anomaly_df = (
        df.loc[
            a["anomaly_mask"]
        ]
        .copy()
    )

    if anomaly_df.empty:

        st.success(
            "No ML anomalies detected."
        )

    else:

        st.write(
            f"{len(anomaly_df):,} records flagged."
        )

        st.dataframe(
            anomaly_df.head(500),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "Data Cleaning":

    st.subheader(
        "Automated Cleaning"
    )

    st.caption(
        "Transform raw data into an analysis-ready dataset."
    )

    st.success(
        "Cleaning completed. Duplicate records were removed, "
        "city values standardized and missing values handled."
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Original Rows",
        f"{a['rows']:,}"
    )

    c2.metric(
        "Cleaned Rows",
        f"{a['cleaned_rows']:,}"
    )

    c3.metric(
        "Rows Removed",
        f"{a['rows_removed']:,}"
    )

    c4.metric(
        "Values Filled",
        f"{a['values_filled']:,}"
    )

    st.write("")

    st.subheader(
        "Cleaning Summary"
    )

    s1, s2 = (
        st.columns(2)
    )

    with s1:

        st.metric(
            "City Values Standardized",
            f"{a['city_changes']:,}"
        )

    with s2:

        st.metric(
            "Duplicate Rows Removed",
            f"{a['rows_removed']:,}"
        )

    st.subheader(
        "Cleaned Dataset"
    )

    st.dataframe(
        a["cleaned_df"].head(20),
        use_container_width=True,
        hide_index=True
    )

    cleaned_csv = (
        a["cleaned_df"]
        .to_csv(index=False)
        .encode()
    )

    st.download_button(
        "Download Cleaned Dataset",
        data=cleaned_csv,
        file_name="DataGuard_Cleaned_Data.csv",
        mime="text/csv"
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.subheader(
        "Business Analytics"
    )

    st.caption(
        "Explore trends and commercial patterns "
        "from the dataset."
    )

    if a["sales_column"] is None:

        st.warning(
            "No Sales, Revenue or Amount column "
            "was detected."
        )

    else:

        business_df = (
            a["business_df"]
        )

        chart_df = (
            a["chart_df"]
        )

        sales_column = (
            a["sales_column"]
        )

        upper_bound = (
            a["upper_bound"]
        )

        if upper_bound is not None:

            st.info(
                f"Business charts use an IQR upper bound "
                f"of {upper_bound:,.2f}. "
                "This affects visualization only and "
                "does not delete records."
            )

        total_sales = (
            business_df[
                sales_column
            ].sum()
        )

        average_sale = (
            business_df[
                sales_column
            ].mean()
        )

        highest_sale = (
            business_df[
                sales_column
            ].max()
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Total Sales",
            f"{total_sales:,.0f}"
        )

        c2.metric(
            "Average Sale",
            f"{average_sale:,.2f}"
        )

        c3.metric(
            "Highest Sale",
            f"{highest_sale:,.2f}"
        )

        c4.metric(
            "Records",
            f"{len(business_df):,}"
        )

        st.write("")

        st.subheader(
            "Sales Distribution"
        )

        st.bar_chart(
            chart_df[
                sales_column
            ],
            use_container_width=True
        )

        city_columns = [
            col
            for col in business_df.columns
            if "city"
            in str(col).lower()
        ]

        if city_columns:

            city_col = (
                city_columns[0]
            )

            st.subheader(
                "Sales by City"
            )

            city_sales = (
                business_df
                .groupby(
                    city_col
                )[sales_column]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                city_sales,
                use_container_width=True
            )

        category_columns = [
            col
            for col in business_df.columns
            if "category"
            in str(col).lower()
        ]

        if category_columns:

            category_col = (
                category_columns[0]
            )

            st.subheader(
                "Sales by Category"
            )

            category_sales = (
                business_df
                .groupby(
                    category_col
                )[sales_column]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.bar_chart(
                category_sales,
                use_container_width=True
            )

        date_columns = (
            detect_datetime_columns(
                business_df
            )
        )

        if date_columns:

            date_col = (
                date_columns[0]
            )

            temp = (
                business_df
                .copy()
            )

            temp[date_col] = (
                safe_to_datetime(
                    temp[date_col]
                )
            )

            monthly = (
                temp
                .dropna(
                    subset=[
                        date_col
                    ]
                )
                .assign(
                    Month=lambda x:
                    x[date_col]
                    .dt
                    .to_period("M")
                    .astype(str)
                )
                .groupby(
                    "Month"
                )[sales_column]
                .sum()
            )

            st.subheader(
                "Monthly Sales Trend"
            )

            st.line_chart(
                monthly,
                use_container_width=True
            )


# ============================================================
# POWER BI
# ============================================================

elif page == "Power BI":

    st.subheader(
        "Power BI Export Center"
    )

    st.caption(
        "Business-ready datasets prepared for Power BI."
    )

    exports = (
        create_powerbi_exports(
            a["cleaned_df"],
            a["business_df"],
            a["sales_column"]
        )
    )

    st.info(
        "Import these CSV files into Power BI Desktop "
        "using Get Data → Text/CSV."
    )

    for filename, content in (
        exports.items()
    ):

        col1, col2 = (
            st.columns(
                [4, 1]
            )
        )

        with col1:

            st.write(
                f"**{filename}**"
            )

            st.caption(
                "Power BI-ready CSV"
            )

        with col2:

            st.download_button(
                "Download",
                data=content,
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}"
            )

    st.write("")

    zip_data = create_zip(
        exports
    )

    st.download_button(
        "Download All Power BI Files",
        data=zip_data,
        file_name="DataGuard_PowerBI_Exports.zip",
        mime="application/zip"
    )

    st.subheader(
        "Recommended Power BI Workflow"
    )

    st.markdown(
        """
        **1.** Import `DataGuard_Cleaned_Data.csv`

        **2.** Validate data types in Power Query

        **3.** Create KPI cards

        **4.** Create Sales by City and Category visuals

        **5.** Create Monthly Sales trend

        **6.** Add Data Quality and Anomaly KPIs

        **7.** Add slicers and filters

        **8.** Build the final dashboard
        """
    )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.subheader(
        "AI Analysis"
    )

    st.caption(
        "Interpret measured DataGuard signals using Gemini."
    )

    report = f"""
DATAGUARD AI ANALYSIS

Dataset:
{st.session_state.file_name}

Rows:
{a['rows']}

Columns:
{a['columns']}

Quality Score:
{a['quality']}/100

Missing:
{a['missing']}

Duplicates:
{a['duplicates']}

Invalid:
{a['invalid']}

IQR Outliers:
{a['iqr_count']}

Isolation Forest Sensitivity:
{contamination}%

ML Anomalies:
{a['anomaly_count']}

Normal Records:
{a['normal_count']}

Rows Removed:
{a['rows_removed']}

Values Filled:
{a['values_filled']}

City Standardization Changes:
{a['city_changes']}

Sales Column:
{a['sales_column']}

Business Visualization Upper Bound:
{a['upper_bound']}
"""

    with st.expander(
        "View DataGuard AI Report"
    ):

        st.code(
            report,
            language="text"
        )

    if st.button(
        "Run Gemini AI Analysis",
        type="primary"
    ):

        with st.spinner(
            "Gemini is analyzing the measured data..."
        ):

            result = run_gemini(
                report
            )

            st.session_state.gemini_analysis = (
                result
            )

    if st.session_state.gemini_analysis:

        st.subheader(
            "Gemini Findings"
        )

        st.markdown(
            st.session_state.gemini_analysis
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.subheader(
        "Data Quality Report"
    )

    st.caption(
        "Complete analysis summary for documentation "
        "and portfolio presentation."
    )

    report = f"""
DATAGUARD AI
DATA QUALITY REPORT
==================================================

Dataset:
{st.session_state.file_name}

Rows:
{a['rows']:,}

Columns:
{a['columns']}

Quality Score:
{a['quality']:.2f}/100

Missing Cells:
{a['missing']:,}

Duplicate Records:
{a['duplicates']:,}

Invalid Values:
{a['invalid']:,}

IQR Outliers:
{a['iqr_count']:,}

Isolation Forest Sensitivity:
{contamination}%

ML Anomalies:
{a['anomaly_count']:,}

Normal Records:
{a['normal_count']:,}

Rows Removed:
{a['rows_removed']:,}

Values Filled:
{a['values_filled']:,}

City Values Standardized:
{a['city_changes']:,}

Sales Column:
{a['sales_column']}

Business Visualization Upper Bound:
{a['upper_bound']}

NOTE:
IQR outliers and Isolation Forest anomalies are
screening signals and are not automatically confirmed
data errors.
"""

    st.text_area(
        "Report Preview",
        report,
        height=480
    )

    st.download_button(
        "Download Report",
        data=report,
        file_name="DataGuard_AI_Report.txt",
        mime="text/plain"
    )

    if st.session_state.gemini_analysis:

        st.subheader(
            "AI Analysis"
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
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br>
        Built with Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
