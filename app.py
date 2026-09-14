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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    /* ---------- BRAND ---------- */

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 20px 0;
    }

    .brand-icon {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 23px;
    }

    .brand-title {
        font-size: 21px;
        font-weight: 700;
        color: white;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 2px;
    }

    /* ---------- HEADER ---------- */

    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .page-title {
        font-size: 31px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 3px;
    }

    .page-subtitle {
        color: #6b7280;
        font-size: 14px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 13px;
        border-radius: 20px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        font-size: 13px;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
    }

    /* ---------- CARDS ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 19px;
        min-height: 125px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #111827;
        font-size: 28px;
        font-weight: 750;
        line-height: 1.1;
    }

    .metric-helper {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 9px;
    }

    .quality-card {
        background: linear-gradient(135deg, #111827, #1f2937);
        color: white;
        border-radius: 16px;
        padding: 23px;
        min-height: 170px;
    }

    .quality-title {
        color: #d1d5db;
        font-size: 13px;
    }

    .quality-score {
        font-size: 43px;
        font-weight: 800;
        margin-top: 12px;
    }

    .quality-status {
        font-size: 13px;
        color: #86efac;
        margin-top: 5px;
    }

    /* ---------- SECTION ---------- */

    .section-heading {
        font-size: 19px;
        font-weight: 700;
        color: #111827;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-description {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 14px;
    }

    /* ---------- INFO ---------- */

    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 13px 16px;
        border-radius: 10px;
        font-size: 13px;
        margin: 10px 0;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 13px 16px;
        border-radius: 10px;
        font-size: 13px;
        margin: 10px 0;
    }

    .success-box {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 13px 16px;
        border-radius: 10px;
        font-size: 13px;
        margin: 10px 0;
    }

    /* ---------- UPLOAD ---------- */

    .upload-card {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
    }

    /* ---------- TABLE ---------- */

    .table-caption {
        color: #6b7280;
        font-size: 12px;
        margin-bottom: 6px;
    }

    /* ---------- NAVIGATION ---------- */

    .nav-label {
        color: #9ca3af !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        margin-top: 18px;
        margin-bottom: 7px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        padding: 30px 0 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "business_df" not in st.session_state:
    st.session_state.business_df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "gemini_analysis" not in st.session_state:
    st.session_state.gemini_analysis = None


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
        elif ratio >= 0.95:
            datetime_columns.append(column)

    return list(dict.fromkeys(datetime_columns))


def detect_identifier_columns(df, datetime_columns=None):

    if datetime_columns is None:
        datetime_columns = []

    identifier_columns = []

    for column in df.columns:

        if column in datetime_columns:
            continue

        name = str(column).lower()

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


def create_profile(df):

    rows = []

    for column in df.columns:

        rows.append(
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Non-Null Count": int(df[column].notna().sum()),
                "Missing": int(df[column].isna().sum()),
                "Unique Values": int(df[column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def detect_invalid_values(df):

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

    invalid_count = 0
    details = []

    for column in df.columns:

        name = str(column).lower()

        if not any(keyword in name for keyword in keywords):
            continue

        series = df[column]

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_empty = series.notna()

        conversion_failures = (
            numeric.isna() & non_empty
        )

        count = int(conversion_failures.sum())

        if count > 0:

            invalid_count += count

            details.append(
                {
                    "Column": column,
                    "Issue": "Non-numeric value",
                    "Count": count,
                }
            )

        if "age" in name:

            negative = int((numeric < 0).sum())

            if negative > 0:

                invalid_count += negative

                details.append(
                    {
                        "Column": column,
                        "Issue": "Negative age",
                        "Count": negative,
                    }
                )

        elif any(
            keyword in name
            for keyword in [
                "quantity",
                "qty",
                "sales",
                "revenue",
                "amount",
                "price",
            ]
        ):

            negative = int((numeric < 0).sum())

            if negative > 0:

                invalid_count += negative

                details.append(
                    {
                        "Column": column,
                        "Issue": "Negative numeric value",
                        "Count": negative,
                    }
                )

    return invalid_count, pd.DataFrame(details)


def standardize_city_column(df):

    cleaned = df.copy()
    total_changes = 0
    city_columns = []

    for column in cleaned.columns:

        if "city" not in str(column).lower():
            continue

        city_columns.append(column)

        original = cleaned[column].copy()

        def normalize_city(value):

            if pd.isna(value):
                return value

            value_str = str(value).strip().lower()

            if value_str in CITY_MAPPING:
                return CITY_MAPPING[value_str]

            return value_str.title()

        cleaned[column] = cleaned[column].apply(normalize_city)

        total_changes += int(
            (original.astype(str) != cleaned[column].astype(str)).sum()
        )

    return cleaned, total_changes


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
                    "Outliers": count,
                    "Lower Bound": round(lower, 2),
                    "Upper Bound": round(upper, 2),
                }
            )

    return total_outliers, pd.DataFrame(details)


def detect_ml_anomalies(df, contamination_pct):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:
        return pd.Series(False, index=df.index), 0

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_df = numeric_df.fillna(
        numeric_df.median(numeric_only=True)
    )

    numeric_df = numeric_df.fillna(0)

    if len(numeric_df) < 10:
        return pd.Series(False, index=df.index), 0

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

    cleaned, city_changes = (
        standardize_city_column(cleaned)
    )

    values_filled = 0

    numeric_columns = cleaned.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        try:

            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            ).astype("float64")

        except Exception:
            continue

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            median_value = (
                cleaned[column].median()
            )

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column].fillna(
                        median_value
                    )
                )

                values_filled += (
                    missing_before
                )

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

            mode_values = (
                cleaned[column].mode(
                    dropna=True
                )
            )

            if len(mode_values) > 0:

                cleaned[column] = (
                    cleaned[column].fillna(
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


def detect_sales_column(df):

    exact_names = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    for name in exact_names:

        for column in df.columns:

            if str(column).lower() == name:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    return column

    for column in df.columns:

        name = str(column).lower()

        if any(
            keyword in name
            for keyword in [
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


def prepare_business_sales(df):

    sales_column = detect_sales_column(df)

    if sales_column is None:
        return df.copy(), None, None

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
        return business_df, sales_column, None

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

    profile = create_profile(
        cleaned_df
    )

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = cleaned_df.to_csv(
        index=False
    ).encode("utf-8")

    exports[
        "DataGuard_Data_Profile.csv"
    ] = profile.to_csv(
        index=False
    ).encode("utf-8")

    if sales_column is not None:

        sales_summary = (
            business_df
            .groupby(sales_column)
            .size()
            .reset_index(
                name="Record_Count"
            )
        )

        exports[
            "DataGuard_Sales_Summary.csv"
        ] = sales_summary.to_csv(
            index=False
        ).encode("utf-8")

    city_columns = [
        column
        for column in cleaned_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns:

        city_column = city_columns[0]

        city_summary = (
            cleaned_df[
                city_column
            ]
            .value_counts()
            .reset_index()
        )

        city_summary.columns = [
            "City",
            "Record_Count",
        ]

        exports[
            "DataGuard_City_Summary.csv"
        ] = city_summary.to_csv(
            index=False
        ).encode("utf-8")

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

        monthly = cleaned_df.copy()

        monthly[date_column] = (
            safe_to_datetime(
                monthly[date_column]
            )
        )

        monthly[sales_column] = pd.to_numeric(
            monthly[sales_column],
            errors="coerce",
        )

        monthly = monthly.dropna(
            subset=[
                date_column,
                sales_column,
            ]
        )

        monthly[
            "Month"
        ] = monthly[
            date_column
        ].dt.to_period(
            "M"
        ).astype(str)

        monthly_sales = (
            monthly
            .groupby("Month")[
                sales_column
            ]
            .sum()
            .reset_index()
        )

        monthly_sales.columns = [
            "Month",
            "Sales",
        ]

        exports[
            "DataGuard_Monthly_Sales.csv"
        ] = monthly_sales.to_csv(
            index=False
        ).encode("utf-8")

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
            "The DataGuard AI core pipeline works "
            "independently of Gemini."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are analyzing a data-quality report generated by DataGuard AI.

Analyze ONLY the supplied report.

Do NOT invent:
- columns
- errors
- causes
- business events
- fraud
- system failures
- ETL failures
- retry behavior
- manual entry problems

Possible causes must be labeled as hypotheses.

IQR outliers and Isolation Forest anomalies
are NOT automatically data errors.

Cleaning has already been performed.
Do not recommend repeating the same cleaning.

If sales values were excluded from charts,
understand that this is visualization-only.
The underlying analytical records were not deleted.

Produce a concise portfolio-friendly analysis with
these exact sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

Report:

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

    except Exception as e:

        return (
            f"Gemini analysis could not be generated.\n\n"
            f"Reason: {str(e)}\n\n"
            "The core DataGuard AI pipeline remains "
            "fully functional without Gemini."
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
                <div class="brand-title">DataGuard AI</div>
                <div class="brand-subtitle">
                    AI Data Quality Platform
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="nav-label">WORKSPACE</div>',
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
        '<div class="nav-label">CONFIGURATION</div>',
        unsafe_allow_html=True,
    )

    contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.caption(
        "Higher sensitivity flags more records "
        "as unusual."
    )

    st.markdown("---")

    st.caption(
        "ML anomalies are screening signals, "
        "not confirmed errors."
    )

    st.caption(
        "DataGuard AI • Portfolio Edition"
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    """
    <div class="page-header">
        <div>
            <div class="page-title">Data Quality Command Center</div>
            <div class="page-subtitle">
                Profile, detect, clean and monitor your data in one workspace.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="upload-card">',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help="Supported formats: CSV, XLSX and XLS",
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

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

        # Automatically prepare analysis
        datetime_columns = (
            detect_datetime_columns(df)
        )

        identifier_columns = (
            detect_identifier_columns(
                df,
                datetime_columns,
            )
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

        total_cells = (
            df.shape[0] * df.shape[1]
        )

        quality_score = build_quality_score(
            total_cells,
            missing_count,
            duplicate_count,
            invalid_count,
        )

        iqr_count, iqr_details = (
            detect_iqr_outliers(df)
        )

        anomaly_mask, anomaly_count = (
            detect_ml_anomalies(
                df,
                contamination_pct,
            )
        )

        cleaned_df, rows_removed, values_filled, city_changes = (
            clean_dataset(df)
        )

        business_df, sales_column, upper_bound = (
            prepare_business_sales(
                cleaned_df
            )
        )

        st.session_state.cleaned_df = (
            cleaned_df
        )

        st.session_state.business_df = (
            business_df
        )

        st.session_state.analysis = {
            "datetime_columns": datetime_columns,
            "identifier_columns": identifier_columns,
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "invalid_count": invalid_count,
            "invalid_details": invalid_details,
            "quality_score": quality_score,
            "iqr_count": iqr_count,
            "iqr_details": iqr_details,
            "anomaly_mask": anomaly_mask,
            "anomaly_count": anomaly_count,
            "cleaned_df": cleaned_df,
            "rows_removed": rows_removed,
            "values_filled": values_filled,
            "city_changes": city_changes,
            "business_df": business_df,
            "sales_column": sales_column,
            "upper_bound": upper_bound,
        }

    except Exception as e:

        st.error(
            f"Unable to process the file: {e}"
        )

        st.stop()


# ============================================================
# EMPTY STATE
# ============================================================

if st.session_state.df is None:

    st.markdown(
        """
        <div style="
            background:white;
            border:1px solid #e5e7eb;
            border-radius:18px;
            padding:60px 30px;
            text-align:center;
            margin-top:20px;
        ">
            <div style="font-size:48px;">📂</div>
            <h2>Start with your dataset</h2>
            <p style="color:#6b7280;">
                Upload a CSV or Excel file to begin
                automated data-quality analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# GET ANALYSIS
# ============================================================

df = st.session_state.df
analysis = st.session_state.analysis

cleaned_df = analysis["cleaned_df"]
business_df = analysis["business_df"]

quality_score = analysis["quality_score"]

missing_count = analysis["missing_count"]
duplicate_count = analysis["duplicate_count"]
invalid_count = analysis["invalid_count"]

iqr_count = analysis["iqr_count"]
anomaly_count = analysis["anomaly_count"]

rows_removed = analysis["rows_removed"]
values_filled = analysis["values_filled"]
city_changes = analysis["city_changes"]

datetime_columns = analysis[
    "datetime_columns"
]

identifier_columns = analysis[
    "identifier_columns"
]

sales_column = analysis[
    "sales_column"
]

upper_bound = analysis[
    "upper_bound"
]


# ============================================================
# DATASET STATUS
# ============================================================

st.markdown(
    f"""
    <div class="info-box">
        <strong>Dataset:</strong> {st.session_state.file_name}
        &nbsp; • &nbsp;
        <strong>{len(df):,}</strong> rows
        &nbsp; • &nbsp;
        <strong>{len(df.columns):,}</strong> columns
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================
# ============================================================
# DASHBOARD — PRODUCTION STYLE
# ============================================================

if page == "Dashboard":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    score = quality_score

    if score >= 95:
        score_status = "Excellent"
        score_icon = "●"
    elif score >= 85:
        score_status = "Good"
        score_icon = "●"
    elif score >= 70:
        score_status = "Needs Attention"
        score_icon = "●"
    else:
        score_status = "Critical"
        score_icon = "●"

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            margin-bottom:24px;
        ">

            <div>
                <div style="
                    font-size:30px;
                    font-weight:800;
                    color:#111827;
                    letter-spacing:-0.8px;
                ">
                    Data Quality Overview
                </div>

                <div style="
                    color:#6b7280;
                    font-size:14px;
                    margin-top:5px;
                ">
                    Monitor the health, reliability and
                    anomaly status of your dataset.
                </div>
            </div>

            <div style="
                background:#ecfdf5;
                border:1px solid #a7f3d0;
                color:#047857;
                border-radius:24px;
                padding:8px 15px;
                font-size:13px;
                font-weight:700;
            ">
                ● Analysis Complete
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # DATASET INFORMATION BAR
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div style="
            background:white;
            border:1px solid #e5e7eb;
            border-radius:14px;
            padding:15px 18px;
            margin-bottom:20px;
            display:flex;
            justify-content:space-between;
            align-items:center;
        ">

            <div>
                <div style="
                    font-size:12px;
                    color:#9ca3af;
                    text-transform:uppercase;
                    letter-spacing:.05em;
                    font-weight:700;
                ">
                    Active Dataset
                </div>

                <div style="
                    font-size:15px;
                    font-weight:700;
                    color:#111827;
                    margin-top:4px;
                ">
                    📄 {st.session_state.file_name}
                </div>
            </div>

            <div style="
                color:#6b7280;
                font-size:13px;
            ">
                Last analyzed:
                <strong style="color:#111827;">
                    {datetime.now().strftime("%d %b %Y, %I:%M %p")}
                </strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOP KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:16px;
                padding:20px;
                min-height:145px;
                box-shadow:0 2px 5px rgba(0,0,0,.025);
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div style="
                        font-size:12px;
                        font-weight:700;
                        color:#6b7280;
                    ">
                        TOTAL RECORDS
                    </div>

                    <div style="
                        font-size:21px;
                    ">
                        📊
                    </div>

                </div>

                <div style="
                    font-size:31px;
                    font-weight:800;
                    color:#111827;
                    margin-top:13px;
                ">
                    {len(df):,}
                </div>

                <div style="
                    font-size:12px;
                    color:#9ca3af;
                    margin-top:5px;
                ">
                    Records analyzed
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:16px;
                padding:20px;
                min-height:145px;
                box-shadow:0 2px 5px rgba(0,0,0,.025);
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div style="
                        font-size:12px;
                        font-weight:700;
                        color:#6b7280;
                    ">
                        MISSING CELLS
                    </div>

                    <div style="font-size:21px;">
                        ⚠️
                    </div>

                </div>

                <div style="
                    font-size:31px;
                    font-weight:800;
                    color:#111827;
                    margin-top:13px;
                ">
                    {missing_count:,}
                </div>

                <div style="
                    font-size:12px;
                    color:#9ca3af;
                    margin-top:5px;
                ">
                    Missing values detected
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:16px;
                padding:20px;
                min-height:145px;
                box-shadow:0 2px 5px rgba(0,0,0,.025);
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div style="
                        font-size:12px;
                        font-weight:700;
                        color:#6b7280;
                    ">
                        DUPLICATES
                    </div>

                    <div style="font-size:21px;">
                        🔁
                    </div>

                </div>

                <div style="
                    font-size:31px;
                    font-weight:800;
                    color:#111827;
                    margin-top:13px;
                ">
                    {duplicate_count:,}
                </div>

                <div style="
                    font-size:12px;
                    color:#9ca3af;
                    margin-top:5px;
                ">
                    Duplicate records
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(
                    135deg,
                    #111827,
                    #1f2937
                );
                border-radius:16px;
                padding:20px;
                min-height:145px;
                box-shadow:0 3px 8px rgba(0,0,0,.08);
            ">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div style="
                        font-size:12px;
                        font-weight:700;
                        color:#9ca3af;
                    ">
                        QUALITY SCORE
                    </div>

                    <div style="font-size:21px;">
                        🛡️
                    </div>

                </div>

                <div style="
                    font-size:31px;
                    font-weight:800;
                    color:white;
                    margin-top:13px;
                ">
                    {quality_score:.2f}
                    <span style="
                        font-size:14px;
                        color:#9ca3af;
                    ">
                        /100
                    </span>
                </div>

                <div style="
                    font-size:12px;
                    color:#86efac;
                    margin-top:5px;
                    font-weight:600;
                ">
                    ● {score_status}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # QUALITY HEALTH + ISSUE SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Quality Health</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.35, 1],
        gap="large",
    )

    with left:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:16px;
                padding:23px;
            ">

                <div style="
                    font-size:16px;
                    font-weight:750;
                    color:#111827;
                ">
                    Dataset Health
                </div>

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    margin-top:4px;
                    margin-bottom:18px;
                ">
                    Overall quality assessment
                </div>

                <div style="
                    height:13px;
                    background:#e5e7eb;
                    border-radius:20px;
                    overflow:hidden;
                ">

                    <div style="
                        width:{quality_score}%;
                        height:100%;
                        background:linear-gradient(
                            90deg,
                            #10b981,
                            #22c55e
                        );
                        border-radius:20px;
                    ">
                    </div>

                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-top:9px;
                    font-size:12px;
                    color:#6b7280;
                ">

                    <span>0</span>

                    <strong style="color:#111827;">
                        {quality_score:.2f}% healthy
                    </strong>

                    <span>100</span>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        total_issues = (
            missing_count
            + duplicate_count
            + invalid_count
        )

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:16px;
                padding:23px;
            ">

                <div style="
                    font-size:16px;
                    font-weight:750;
                    color:#111827;
                ">
                    Issues Requiring Attention
                </div>

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    margin-top:4px;
                    margin-bottom:17px;
                ">
                    Confirmed quality issues
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:10px;
                ">
                    <span style="font-size:13px;">
                        Missing values
                    </span>

                    <strong>
                        {missing_count:,}
                    </strong>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-bottom:10px;
                ">
                    <span style="font-size:13px;">
                        Duplicate records
                    </span>

                    <strong>
                        {duplicate_count:,}
                    </strong>
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                ">
                    <span style="font-size:13px;">
                        Invalid values
                    </span>

                    <strong>
                        {invalid_count:,}
                    </strong>
                </div>

                <div style="
                    border-top:1px solid #e5e7eb;
                    margin-top:15px;
                    padding-top:13px;
                    display:flex;
                    justify-content:space-between;
                    font-size:13px;
                ">

                    <strong>
                        Total confirmed issues
                    </strong>

                    <strong>
                        {total_issues:,}
                    </strong>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # ANOMALY MONITORING
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Anomaly Monitoring</div>',
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:15px;
                padding:20px;
            ">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    IQR OUTLIERS
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {iqr_count:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Statistically unusual observations
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:15px;
                padding:20px;
            ">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    ML ANOMALIES
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {anomaly_count:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Isolation Forest screening signals
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with a3:

        normal_records = (
            len(df) - anomaly_count
        )

        st.markdown(
            f"""
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:15px;
                padding:20px;
            ">

                <div style="
                    color:#6b7280;
                    font-size:12px;
                    font-weight:700;
                ">
                    NORMAL RECORDS
                </div>

                <div style="
                    font-size:28px;
                    font-weight:800;
                    margin-top:9px;
                    color:#111827;
                ">
                    {normal_records:,}
                </div>

                <div style="
                    color:#9ca3af;
                    font-size:12px;
                    margin-top:4px;
                ">
                    Not flagged by Isolation Forest
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="
            background:#fffbeb;
            border:1px solid #fde68a;
            border-radius:11px;
            padding:12px 15px;
            margin-top:13px;
            font-size:12px;
            color:#92400e;
        ">
            <strong>Interpretation:</strong>
            IQR outliers and ML anomalies are screening
            signals. They are not automatically confirmed
            data errors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # CLEANING SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Cleaning Summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Rows Removed",
            f"{rows_removed:,}",
        )

    with c2:

        st.metric(
            "Values Filled",
            f"{values_filled:,}",
        )

    with c3:

        st.metric(
            "Cities Standardized",
            f"{city_changes:,}",
        )

    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Dataset Preview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom:8px;
        ">

            <div style="
                color:#6b7280;
                font-size:12px;
            ">
                Showing <strong>10</strong> of
                <strong>{len(df):,}</strong> records
            </div>

            <div style="
                color:#9ca3af;
                font-size:12px;
            ">
                Preview only — full dataset analyzed
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-heading">Pipeline Status</div>',
        unsafe_allow_html=True,
    )

    pipeline_items = [
        ("01", "Profile"),
        ("02", "Quality"),
        ("03", "Anomalies"),
        ("04", "Cleaning"),
        ("05", "Analytics"),
        ("06", "Exports"),
    ]

    pipeline_cols = st.columns(6)

    for col, (number, name) in zip(
        pipeline_cols,
        pipeline_items,
    ):

        with col:

            st.markdown(
                f"""
                <div style="
                    background:white;
                    border:1px solid #e5e7eb;
                    border-radius:12px;
                    padding:13px;
                    text-align:center;
                ">

                    <div style="
                        font-size:11px;
                        color:#9ca3af;
                        font-weight:700;
                    ">
                        STEP {number}
                    </div>

                    <div style="
                        color:#047857;
                        font-size:13px;
                        font-weight:700;
                        margin-top:5px;
                    ">
                        ✓ {name}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.markdown(
        '<div class="section-heading">Data Quality</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Identify missing data, duplicate records,
            invalid values and structural characteristics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

    c2.metric(
        "Duplicates",
        f"{duplicate_count:,}",
    )

    c3.metric(
        "Invalid Values",
        f"{invalid_count:,}",
    )

    c4.metric(
        "Quality Score",
        f"{quality_score:.2f}/100",
    )

    st.markdown(
        '<div class="section-heading">Dataset Structure</div>',
        unsafe_allow_html=True,
    )

    structure_cols = st.columns(4)

    structure_cols[0].metric(
        "Numeric Columns",
        len(
            df.select_dtypes(
                include=np.number
            ).columns
        ),
    )

    structure_cols[1].metric(
        "Categorical Columns",
        len(
            df.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category",
                ]
            ).columns
        ),
    )

    structure_cols[2].metric(
        "Date / Time",
        len(datetime_columns),
    )

    structure_cols[3].metric(
        "Identifiers",
        len(identifier_columns),
    )

    with st.expander(
        "📋 View complete data profile",
        expanded=True,
    ):

        profile = create_profile(df)

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander(
        "⚠️ Invalid value details"
    ):

        if invalid_count > 0:

            st.dataframe(
                analysis["invalid_details"],
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No invalid values detected."
            )

    with st.expander(
        "📅 Detected date/time columns"
    ):

        if datetime_columns:
            st.write(
                datetime_columns
            )
        else:
            st.info(
                "No date/time columns detected."
            )

    with st.expander(
        "🆔 Detected identifier columns"
    ):

        if identifier_columns:
            st.write(
                identifier_columns
            )
        else:
            st.info(
                "No identifier columns detected."
            )


# ============================================================
# ANOMALIES
# ============================================================

elif page == "Anomalies":

    st.markdown(
        '<div class="section-heading">Anomaly Monitoring</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Important:</strong>
            Statistical outliers and ML anomalies are
            unusual observations. They are not automatically
            data errors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "IQR Outlier Observations",
        f"{iqr_count:,}",
    )

    c2.metric(
        "Isolation Forest Anomalies",
        f"{anomaly_count:,}",
    )

    c3.metric(
        "Normal Records",
        f"{len(df) - anomaly_count:,}",
    )

    st.markdown(
        '<div class="section-heading">IQR Detection</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "IQR detects observations outside "
        "1.5 × the interquartile range."
    )

    if len(analysis["iqr_details"]) > 0:

        st.dataframe(
            analysis["iqr_details"],
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No IQR outliers detected."
        )

    st.markdown(
        '<div class="section-heading">Isolation Forest</div>',
        unsafe_allow_html=True,
    )

    st.info(
        f"Current sensitivity: {contamination_pct}%. "
        "Higher sensitivity flags more records as unusual."
    )

    anomaly_display = df.copy()

    anomaly_display[
        "ML_Anomaly"
    ] = np.where(
        analysis["anomaly_mask"],
        "Anomaly",
        "Normal",
    )

    anomaly_only = anomaly_display[
        anomaly_display["ML_Anomaly"]
        == "Anomaly"
    ]

    st.caption(
        f"Showing {min(100, len(anomaly_only)):,} "
        f"of {len(anomaly_only):,} detected anomalies"
    )

    st.dataframe(
        anomaly_only.head(100),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CLEANING
# ============================================================

elif page == "Cleaning":

    st.markdown(
        '<div class="section-heading">Automated Cleaning</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            DataGuard AI applies safe automated transformations
            to create a cleaner analytical dataset.
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
        f"{rows_removed:,}",
    )

    c4.metric(
        "Values Filled",
        f"{values_filled:,}",
    )

    st.markdown(
        '<div class="section-heading">Cleaning Actions</div>',
        unsafe_allow_html=True,
    )

    actions = [
        (
            "Duplicate removal",
            f"{rows_removed:,} duplicate rows removed",
        ),
        (
            "Missing-value treatment",
            f"{values_filled:,} values filled",
        ),
        (
            "City standardization",
            f"{city_changes:,} city values standardized",
        ),
    ]

    for title, description in actions:

        st.success(
            f"✓ **{title}** — {description}"
        )

    st.markdown(
        '<div class="section-heading">Cleaned Dataset</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Showing 10 of {len(cleaned_df):,} records"
    )

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":

    st.markdown(
        '<div class="section-heading">Business Analytics</div>',
        unsafe_allow_html=True,
    )

    if sales_column is None:

        st.warning(
            "No Sales, Revenue or Amount column "
            "was detected. Business sales charts "
            "are therefore unavailable."
        )

    else:

        st.success(
            f"Detected sales column: **{sales_column}**"
        )

        if upper_bound is not None:

            st.caption(
                f"Visualization upper bound: "
                f"{upper_bound:,.2f}"
            )

        chart_df = business_df.copy()

        st.markdown(
            '<div class="section-heading">Sales Distribution</div>',
            unsafe_allow_html=True,
        )

        hist_values, bin_edges = np.histogram(
            chart_df[sales_column],
            bins=10,
        )

        histogram_df = pd.DataFrame(
            {
                "Sales Range": [
                    f"{bin_edges[i]:,.0f} - "
                    f"{bin_edges[i + 1]:,.0f}"
                    for i in range(
                        len(bin_edges) - 1
                    )
                ],
                "Records": hist_values,
            }
        )

        st.bar_chart(
            histogram_df.set_index(
                "Sales Range"
            )
        )

        city_columns = [
            column
            for column in chart_df.columns
            if "city" in str(column).lower()
        ]

        product_columns = [
            column
            for column in chart_df.columns
            if "product" in str(column).lower()
        ]

        category_columns = [
            column
            for column in chart_df.columns
            if "category" in str(column).lower()
        ]

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="section-heading">Records by City</div>',
                unsafe_allow_html=True,
            )

            if city_columns:

                city_column = city_columns[0]

                city_chart = (
                    chart_df[
                        city_column
                    ]
                    .value_counts()
                    .head(15)
                )

                st.bar_chart(
                    city_chart
                )

            else:

                st.info(
                    "No city column detected."
                )

        with col2:

            st.markdown(
                '<div class="section-heading">Sales by Product</div>',
                unsafe_allow_html=True,
            )

            if product_columns:

                product_column = (
                    product_columns[0]
                )

                product_sales = (
                    chart_df
                    .groupby(
                        product_column
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(15)
                )

                st.bar_chart(
                    product_sales
                )

            else:

                st.info(
                    "No product column detected."
                )

        col3, col4 = st.columns(2)

        with col3:

            st.markdown(
                '<div class="section-heading">Sales by Category</div>',
                unsafe_allow_html=True,
            )

            if category_columns:

                category_column = (
                    category_columns[0]
                )

                category_sales = (
                    chart_df
                    .groupby(
                        category_column
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.bar_chart(
                    category_sales
                )

            else:

                st.info(
                    "No category column detected."
                )

        with col4:

            st.markdown(
                '<div class="section-heading">Monthly Sales Trend</div>',
                unsafe_allow_html=True,
            )

            datetime_columns_clean = (
                detect_datetime_columns(
                    cleaned_df
                )
            )

            if datetime_columns_clean:

                date_column = (
                    datetime_columns_clean[0]
                )

                monthly = cleaned_df.copy()

                monthly[date_column] = (
                    safe_to_datetime(
                        monthly[date_column]
                    )
                )

                monthly[sales_column] = (
                    pd.to_numeric(
                        monthly[sales_column],
                        errors="coerce",
                    )
                )

                monthly = monthly.dropna(
                    subset=[
                        date_column,
                        sales_column,
                    ]
                )

                monthly["Month"] = (
                    monthly[
                        date_column
                    ]
                    .dt.to_period("M")
                    .astype(str)
                )

                monthly_sales = (
                    monthly
                    .groupby("Month")[
                        sales_column
                    ]
                    .sum()
                )

                st.line_chart(
                    monthly_sales
                )

            else:

                st.info(
                    "No date column available."
                )

        st.markdown(
            """
            <div class="info-box">
                Business charts use the cleaned dataset.
                Sales values above the IQR visualization
                upper bound are excluded from relevant
                visualizations only. These records are
                <strong>not deleted</strong> from the
                analytical dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# POWER BI
# ============================================================

elif page == "Power BI":

    st.markdown(
        '<div class="section-heading">Power BI Workspace</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Export cleaned and summarized datasets for
            Power BI Desktop.
        </div>
        """,
        unsafe_allow_html=True,
    )

    exports = create_powerbi_exports(
        cleaned_df,
        business_df,
        sales_column,
    )

    st.success(
        f"{len(exports)} Power BI-ready files generated."
    )

    st.markdown(
        '<div class="section-heading">Available Exports</div>',
        unsafe_allow_html=True,
    )

    for filename in exports:

        st.write(
            f"📄 **{filename}**"
        )

        st.download_button(
            label=f"Download {filename}",
            data=exports[filename],
            file_name=filename,
            mime="text/csv",
            key=f"download_{filename}",
        )

    zip_data = create_zip(exports)

    st.markdown(
        '<div class="section-heading">Complete Package</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️ Download All Power BI Files (ZIP)",
        data=zip_data,
        file_name="DataGuard_PowerBI_Exports.zip",
        mime="application/zip",
        type="primary",
    )

    st.markdown(
        """
        <div class="info-box">
            <strong>Power BI workflow:</strong><br><br>
            1. Download the ZIP package.<br>
            2. Extract the CSV files.<br>
            3. Open Power BI Desktop.<br>
            4. Select Get Data → Text/CSV.<br>
            5. Import the cleaned dataset and supporting tables.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "AI Analysis":

    st.markdown(
        '<div class="section-heading">Gemini AI Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-description">
            Generate an AI interpretation of the
            DataGuard quality and anomaly results.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_text = f"""
DataGuard AI Data Quality Report

Dataset:
{st.session_state.file_name}

Rows:
{len(df)}

Columns:
{len(df.columns)}

Missing cells:
{missing_count}

Duplicate records:
{duplicate_count}

Invalid values:
{invalid_count}

Quality score:
{quality_score}/100

Date/time columns:
{datetime_columns}

Identifier columns:
{identifier_columns}

IQR outlier observations:
{iqr_count}

Isolation Forest sensitivity:
{contamination_pct}%

Isolation Forest anomalies:
{anomaly_count}

Normal records:
{len(df) - anomaly_count}

Original rows:
{len(df)}

Cleaned rows:
{len(cleaned_df)}

Rows removed:
{rows_removed}

Values filled:
{values_filled}

City values standardized:
{city_changes}

Sales column:
{sales_column}

Business chart IQR upper bound:
{upper_bound}
"""

    if st.button(
        "✨ Generate Gemini AI Analysis",
        type="primary",
    ):

        with st.spinner(
            "Gemini is analyzing the data-quality report..."
        ):

            result = get_gemini_analysis(
                summary_text
            )

            st.session_state.gemini_analysis = (
                result
            )

    if st.session_state.gemini_analysis:

        st.markdown(
            st.session_state.gemini_analysis
        )

    else:

        st.info(
            "Click **Generate Gemini AI Analysis** "
            "to request an AI-generated interpretation."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":

    st.markdown(
        '<div class="section-heading">Final Data Quality Report</div>',
        unsafe_allow_html=True,
    )

    report = f"""
# DataGuard AI — Data Quality Report

## Dataset

File: {st.session_state.file_name}

Rows: {len(df):,}

Columns: {len(df.columns):,}

## Quality

Quality Score: {quality_score:.2f}/100

Missing Cells: {missing_count:,}

Duplicate Records: {duplicate_count:,}

Invalid Values: {invalid_count:,}

## Structure

Numeric Columns: {len(df.select_dtypes(include=np.number).columns)}

Categorical Columns: {len(df.select_dtypes(include=["object", "string", "category"]).columns)}

Date/Time Columns: {len(datetime_columns)}

Identifier Columns: {len(identifier_columns)}

## Anomaly Detection

IQR Outlier Observations: {iqr_count:,}

Isolation Forest Sensitivity: {contamination_pct}%

ML Anomalies: {anomaly_count:,}

Normal Records: {len(df) - anomaly_count:,}

## Cleaning

Original Rows: {len(df):,}

Cleaned Rows: {len(cleaned_df):,}

Rows Removed: {rows_removed:,}

Values Filled: {values_filled:,}

City Values Standardized: {city_changes:,}

## Business Analytics

Sales Column: {sales_column}

Visualization IQR Upper Bound: {upper_bound}

Note: IQR filtering for sales is used only for visualization.
Records are not deleted from the analytical dataset.

## Interpretation

IQR outliers are statistically unusual observations.

Isolation Forest anomalies are screening signals and
are not automatically confirmed data errors.

Possible root causes should be treated as hypotheses
unless supported by additional source-system evidence.
"""

    st.text_area(
        "Report Preview",
        report,
        height=500,
    )

    st.download_button(
        "⬇️ Download Report",
        data=report,
        file_name="DataGuard_AI_Report.txt",
        mime="text/plain",
        type="primary",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI &nbsp;•&nbsp;
        AI-Powered Data Quality & Anomaly Detection
        <br>
        Built with Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
