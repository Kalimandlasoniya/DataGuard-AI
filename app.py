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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        color: white;
    }

    .brand-subtitle {
        font-size: 12px;
        color: #9ca3af !important;
        margin-top: 3px;
    }

    .page-title {
        font-size: 32px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
    }

    .page-subtitle {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 22px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 750;
        color: #111827;
        margin-top: 24px;
        margin-bottom: 10px;
    }

    .section-description {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .card-title {
        font-size: 16px;
        font-weight: 750;
        color: #111827;
    }

    .card-text {
        font-size: 13px;
        color: #6b7280;
        margin-top: 5px;
    }

    .kpi {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        min-height: 120px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    .kpi-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 700;
    }

    .kpi-value {
        color: #111827;
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-caption {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 3px;
    }

    .good {
        color: #166534;
        background: #dcfce7;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .warning {
        color: #92400e;
        background: #fef3c7;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .danger {
        color: #991b1b;
        background: #fee2e2;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    .info {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 13px 15px;
        color: #1e3a8a;
        font-size: 13px;
        margin: 10px 0;
    }

    .success {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 13px 15px;
        color: #166534;
        font-size: 13px;
        margin: 10px 0;
    }

    .warning-box {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 13px 15px;
        color: #92400e;
        font-size: 13px;
        margin: 10px 0;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
    }

    .stButton button {
        border-radius: 9px;
        font-weight: 650;
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
            return pd.Series(
                pd.NaT,
                index=series.index,
            )
    except Exception:
        return pd.Series(
            pd.NaT,
            index=series.index,
        )


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

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            datetime_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        name = str(column).lower()

        parsed = safe_to_datetime(
            df[column]
        )

        ratio = parsed.notna().mean()

        if any(token in name for token in tokens):

            if ratio >= 0.50:
                datetime_columns.append(column)

        else:

            if ratio >= 0.95:
                datetime_columns.append(column)

    return list(
        dict.fromkeys(datetime_columns)
    )


def detect_identifier_columns(
    df,
    datetime_columns=None,
):

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

        if any(
            token in name
            for token in identifier_tokens
        ):
            identifier_columns.append(column)
            continue

        if len(df) > 0:

            unique_ratio = (
                df[column].nunique(
                    dropna=True
                )
                / len(df)
            )

            if (
                unique_ratio >= 0.98
                and (
                    pd.api.types.is_integer_dtype(
                        df[column]
                    )
                    or pd.api.types.is_object_dtype(
                        df[column]
                    )
                    or pd.api.types.is_string_dtype(
                        df[column]
                    )
                )
            ):
                identifier_columns.append(
                    column
                )

    return list(
        dict.fromkeys(
            identifier_columns
        )
    )


def profile_dataset(df):

    return pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non-Null Count": [
                int(
                    df[column]
                    .notna()
                    .sum()
                )
                for column in df.columns
            ],
            "Missing": [
                int(
                    df[column]
                    .isna()
                    .sum()
                )
                for column in df.columns
            ],
            "Unique Values": [
                int(
                    df[column]
                    .nunique(
                        dropna=True
                    )
                )
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

        if not any(
            token in name
            for token in numeric_tokens
        ):
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        original_non_null = (
            df[column].notna()
        )

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

            negative_values = (
                numeric_values < 0
            )

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
        pd.DataFrame(
            invalid_details
        ),
    )


def standardize_city_column(df):

    result = df.copy()

    total_changes = 0

    city_columns = [
        column
        for column in result.columns
        if "city"
        in str(column).lower()
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
            original.astype("string")
            .fillna("")
            != standardized
            .fillna("")
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
    total_outliers = 0

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


def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        )
        .copy()
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
        contamination=(
            contamination_pct / 100
        ),
        random_state=42,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    anomaly_mask = (
        predictions == -1
    )

    return (
        pd.Series(
            anomaly_mask,
            index=df.index,
        ),
        int(
            anomaly_mask.sum()
        ),
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
        missing_count
        / total_cells
    ) * 100

    duplicate_penalty = (
        duplicate_count
        / max(
            1,
            total_cells,
        )
    ) * 100

    invalid_penalty = (
        invalid_count
        / max(
            1,
            total_cells,
        )
    ) * 100

    score = 100 - (
        missing_penalty * 0.50
        + duplicate_penalty * 0.30
        + invalid_penalty * 0.20
    )

    return round(
        max(
            0,
            min(
                100,
                score,
            ),
        ),
        2,
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
            cleaned[column]
            .isna()
            .sum()
        )

        if missing_before > 0:

            median_value = (
                cleaned[column]
                .median()
            )

            if pd.notna(
                median_value
            ):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
                        median_value
                    )
                )

                values_filled += (
                    missing_before
                )

    categorical_columns = (
        cleaned
        .select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
    )

    for column in (
        categorical_columns
    ):

        missing_before = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing_before > 0:

            mode_values = (
                cleaned[column]
                .mode(
                    dropna=True
                )
            )

            if len(
                mode_values
            ) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(
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

    sales_column = (
        find_sales_column(df)
    )

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

    if len(
        business_df
    ) == 0:

        return (
            None,
            sales_column,
            None,
        )

    q1 = (
        business_df[
            sales_column
        ].quantile(0.25)
    )

    q3 = (
        business_df[
            sales_column
        ].quantile(0.75)
    )

    iqr = q3 - q1

    upper_bound = (
        q3 + 1.5 * iqr
    )

    chart_df = (
        business_df[
            business_df[
                sales_column
            ] <= upper_bound
        ]
        .copy()
    )

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

    profile = profile_dataset(
        cleaned_df
    )

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
            .encode("utf-8")
        )

    city_columns = [
        column
        for column in cleaned_df.columns
        if "city"
        in str(column).lower()
    ]

    if city_columns:

        city_column = (
            city_columns[0]
        )

        city_summary = (
            cleaned_df
            .groupby(
                city_column
            )
            .size()
            .reset_index(
                name="Record_Count"
            )
        )

        exports[
            "DataGuard_City_Summary.csv"
        ] = (
            city_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    datetime_columns = (
        detect_datetime_columns(
            cleaned_df
        )
    )

    if (
        datetime_columns
        and sales_column is not None
    ):

        date_column = (
            datetime_columns[0]
        )

        monthly_df = (
            cleaned_df.copy()
        )

        monthly_df[
            date_column
        ] = safe_to_datetime(
            monthly_df[
                date_column
            ]
        )

        monthly_df[
            sales_column
        ] = pd.to_numeric(
            monthly_df[
                sales_column
            ],
            errors="coerce",
        )

        monthly_df = (
            monthly_df
            .dropna(
                subset=[
                    date_column,
                    sales_column,
                ]
            )
        )

        monthly_summary = (
            monthly_df
            .assign(
                Month=monthly_df[
                    date_column
                ]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month")[
                sales_column
            ]
            .sum()
            .reset_index()
        )

        exports[
            "DataGuard_Monthly_Sales.csv"
        ] = (
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

        for filename, data in (
            exports.items()
        ):

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


def get_gemini_analysis(
    summary_text
):

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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:8px 2px 18px 2px;">
            <div style="font-size:30px;">🛡️</div>
            <div class="brand-title">
                DataGuard AI
            </div>
            <div class="brand-subtitle">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Navigation")

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

    st.markdown("---")

    st.markdown("### Anomaly Detection")

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
        "as unusual. ML anomalies are screening "
        "signals, not confirmed errors."
    )

    st.markdown("---")

    st.markdown("### Pipeline")

    st.markdown(
        """
        **01** Upload

        **02** Profile

        **03** Quality Checks

        **04** Anomaly Detection

        **05** Cleaning

        **06** Analytics

        **07** Power BI

        **08** AI Analysis
        """
    )

    st.markdown("---")

    st.caption(
        "DataGuard AI • Portfolio Edition"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-title">DataGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-subtitle">'
    "AI-powered data quality, anomaly detection "
    "and analytics platform"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    """
    <div class="card">
        <div class="card-title">
            Upload your dataset
        </div>
        <div class="card-text">
            Start a new data-quality analysis by uploading
            a CSV or Excel dataset.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)


# ============================================================
# LOAD DATA
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

            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.analysis = None
            st.session_state.analysis_complete = False
            st.session_state.gemini_analysis = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )
            st.stop()


# ============================================================
# ANALYSIS ENGINE
# ============================================================

if st.session_state.df is not None:

    df = st.session_state.df

    if not st.session_state.analysis_complete:

        with st.spinner(
            "Analyzing the complete dataset..."
        ):

            rows = len(df)
            columns = len(df.columns)

            total_cells = (
                rows * columns
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
                    invalid_count,
                )
            )

            datetime_columns = (
                detect_datetime_columns(
                    df
                )
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
                rows
                - ml_anomaly_count
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
# DISPLAY APP
# ============================================================

if (
    st.session_state.df is not None
    and st.session_state.analysis_complete
):

    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df
    business_df = st.session_state.business_df
    analysis = st.session_state.analysis

    contamination_pct = (
        st.session_state.contamination_pct
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.markdown(
            "## Dashboard Overview"
        )

        st.markdown(
            f"""
            <div class="info">
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

        score = analysis[
            "quality_score"
        ]

        if score >= 99:
            status = "Excellent"
        elif score >= 95:
            status = "Good"
        elif score >= 85:
            status = "Needs Attention"
        else:
            status = "Critical"

        k1, k2, k3, k4 = st.columns(4)

        with k1:

            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">
                        DATA QUALITY SCORE
                    </div>
                    <div class="kpi-value">
                        {score}/100
                    </div>
                    <div class="kpi-caption">
                        {status}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:

            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">
                        MISSING CELLS
                    </div>
                    <div class="kpi-value">
                        {analysis["missing_count"]:,}
                    </div>
                    <div class="kpi-caption">
                        Across dataset
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:

            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">
                        DUPLICATES
                    </div>
                    <div class="kpi-value">
                        {analysis["duplicate_count"]:,}
                    </div>
                    <div class="kpi-caption">
                        Duplicate records
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k4:

            st.markdown(
                f"""
                <div class="kpi">
                    <div class="kpi-label">
                        ML ANOMALIES
                    </div>
                    <div class="kpi-value">
                        {analysis["ml_anomaly_count"]:,}
                    </div>
                    <div class="kpi-caption">
                        {contamination_pct}% sensitivity
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            "## Data Profile"
        )

        p1, p2, p3, p4, p5 = st.columns(5)

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
            "IQR Outliers",
            f'{analysis["iqr_outlier_count"]:,}',
        )

        st.markdown(
            "## Quality Monitoring"
        )

        quality_df = pd.DataFrame(
            {
                "Metric": [
                    "Missing",
                    "Duplicates",
                    "Invalid Values",
                    "IQR Outliers",
                    "ML Anomalies",
                ],
                "Count": [
                    analysis[
                        "missing_count"
                    ],
                    analysis[
                        "duplicate_count"
                    ],
                    analysis[
                        "invalid_count"
                    ],
                    analysis[
                        "iqr_outlier_count"
                    ],
                    analysis[
                        "ml_anomaly_count"
                    ],
                ],
            }
        )

        st.bar_chart(
            quality_df.set_index(
                "Metric"
            )
        )

        if business_df is not None:

            st.markdown(
                "## Business Analytics Preview"
            )

            sales_column = analysis[
                "sales_column"
            ]

            left, right = st.columns(2)

            with left:

                st.markdown(
                    f"### Sales Distribution"
                )

                values = (
                    business_df[
                        sales_column
                    ]
                )

                hist, bins = np.histogram(
                    values,
                    bins=10,
                )

                hist_df = pd.DataFrame(
                    {
                        "Sales Range": [
                            f"{bins[i]:,.0f} - "
                            f"{bins[i+1]:,.0f}"
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
                    )
                )

            with right:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_counts = (
                        cleaned_df[
                            city_columns[0]
                        ]
                        .value_counts()
                        .head(10)
                    )

                    st.markdown(
                        "### Records by City"
                    )

                    st.bar_chart(
                        city_counts
                    )

        st.markdown(
            "## Data Preview"
        )

        st.caption(
            f"Showing 10 of {len(df):,} records"
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

        st.markdown(
            "## Data Quality"
        )

        st.caption(
            "Detailed profiling and rule-based quality validation."
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Quality Score",
            f'{analysis["quality_score"]}/100',
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

        st.markdown(
            "### Data Profile"
        )

        st.dataframe(
            profile_dataset(df),
            use_container_width=True,
            height=450,
        )

        st.markdown(
            "### Detected Data Types"
        )

        a, b, c = st.columns(3)

        a.metric(
            "Numeric Columns",
            analysis["numeric_count"],
        )

        b.metric(
            "Categorical Columns",
            analysis["categorical_count"],
        )

        c.metric(
            "Date/Time Columns",
            len(
                analysis[
                    "datetime_columns"
                ]
            ),
        )

        with st.expander(
            "View Date/Time columns"
        ):

            if analysis[
                "datetime_columns"
            ]:

                st.write(
                    analysis[
                        "datetime_columns"
                    ]
                )

            else:

                st.info(
                    "No Date/Time columns detected."
                )

        with st.expander(
            "View Identifier columns"
        ):

            if analysis[
                "identifier_columns"
            ]:

                st.write(
                    analysis[
                        "identifier_columns"
                    ]
                )

            else:

                st.info(
                    "No Identifier columns detected."
                )

        st.markdown(
            "### Invalid Values"
        )

        if analysis[
            "invalid_details"
        ].empty:

            st.success(
                "No invalid values were detected."
            )

        else:

            st.dataframe(
                analysis[
                    "invalid_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            "### City Standardization"
        )

        st.info(
            f'{analysis["city_changes"]:,} city values '
            "can be standardized during cleaning."
        )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        st.markdown(
            "## Anomaly Detection"
        )

        st.caption(
            "Statistical outlier detection and machine-learning anomaly screening."
        )

        a1, a2, a3 = st.columns(3)

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

        st.warning(
            "IQR outliers and ML anomalies are not automatically "
            "data errors. They identify statistically unusual "
            "observations that may require business validation."
        )

        st.markdown(
            "### IQR Statistical Outliers"
        )

        if analysis[
            "iqr_details"
        ].empty:

            st.success(
                "No IQR outliers detected."
            )

        else:

            st.dataframe(
                analysis[
                    "iqr_details"
                ],
                use_container_width=True,
            )

        st.markdown(
            "### Isolation Forest"
        )

        st.info(
            f"Current sensitivity: {contamination_pct}%"
        )

        anomaly_display = df.copy()

        anomaly_display[
            "ML_Anomaly"
        ] = np.where(
            analysis[
                "anomaly_mask"
            ],
            "Anomaly",
            "Normal",
        )

        anomaly_records = (
            anomaly_display[
                anomaly_display[
                    "ML_Anomaly"
                ]
                == "Anomaly"
            ]
        )

        with st.expander(
            "View ML anomaly records"
        ):

            st.caption(
                f"Showing up to 500 of "
                f"{len(anomaly_records):,} detected anomalies."
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

        st.markdown(
            "## Automated Cleaning"
        )

        st.caption(
            "Rule-based cleaning applied to create a Power BI-ready dataset."
        )

        c1, c2, c3, c4 = st.columns(4)

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

        st.success(
            "Cleaning actions: duplicate removal, city "
            "standardization, numeric missing-value "
            "imputation using median values, and categorical "
            "missing-value imputation using mode values."
        )

        if analysis[
            "city_changes"
        ] > 0:

            st.info(
                f'{analysis["city_changes"]:,} city values '
                "were standardized."
            )

        st.markdown(
            "### Cleaned Dataset Preview"
        )

        st.caption(
            f"Showing 10 of {len(cleaned_df):,} cleaned records"
        )

        st.dataframe(
            cleaned_df.head(10),
            use_container_width=True,
            height=350,
        )

        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=(
                cleaned_df
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name=(
                "DataGuard_Cleaned_Data.csv"
            ),
            mime="text/csv",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "Analytics":

        st.markdown(
            "## Business Analytics"
        )

        st.caption(
            "Business-focused views generated from the cleaned dataset."
        )

        if business_df is None:

            st.warning(
                "No Sales, Revenue, or Amount column "
                "was detected. Business sales charts "
                "are therefore unavailable."
            )

        else:

            sales_column = analysis[
                "sales_column"
            ]

            upper_bound = analysis[
                "upper_bound"
            ]

            st.info(
                f"Business charts use the cleaned dataset. "
                f"For visualization only, Sales values above "
                f"the cleaned-data IQR upper bound "
                f"{upper_bound:,.2f} are excluded. "
                "These records are not deleted."
            )

            st.markdown(
                "### Sales Distribution"
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
                        f"{bins[i+1]:,.0f}"
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
                )
            )

            left, right = st.columns(2)

            with left:

                city_columns = [
                    column
                    for column in cleaned_df.columns
                    if "city"
                    in str(column).lower()
                ]

                if city_columns:

                    city_counts = (
                        cleaned_df[
                            city_columns[0]
                        ]
                        .value_counts()
                        .head(10)
                    )

                    st.markdown(
                        "### Records by City"
                    )

                    st.bar_chart(
                        city_counts
                    )

            with right:

                product_columns = [
                    column
                    for column in cleaned_df.columns
                    if "product"
                    in str(column).lower()
                ]

                if product_columns:

                    product_sales = (
                        business_df
                        .groupby(
                            product_columns[0]
                        )[sales_column]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                        .head(10)
                    )

                    st.markdown(
                        "### Sales by Product"
                    )

                    st.bar_chart(
                        product_sales
                    )

            category_columns = [
                column
                for column in cleaned_df.columns
                if "category"
                in str(column).lower()
            ]

            if category_columns:

                category_sales = (
                    business_df
                    .groupby(
                        category_columns[0]
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown(
                    "### Sales by Category"
                )

                st.bar_chart(
                    category_sales
                )

            datetime_columns = (
                analysis[
                    "datetime_columns"
                ]
            )

            if datetime_columns:

                date_column = (
                    datetime_columns[0]
                )

                monthly_df = (
                    cleaned_df.copy()
                )

                monthly_df[
                    date_column
                ] = safe_to_datetime(
                    monthly_df[
                        date_column
                    ]
                )

                monthly_df[
                    sales_column
                ] = pd.to_numeric(
                    monthly_df[
                        sales_column
                    ],
                    errors="coerce",
                )

                monthly_df = (
                    monthly_df
                    .dropna(
                        subset=[
                            date_column,
                            sales_column,
                        ]
                    )
                )

                monthly_sales = (
                    monthly_df
                    .assign(
                        Month=monthly_df[
                            date_column
                        ]
                        .dt.to_period("M")
                        .astype(str)
                    )
                    .groupby(
                        "Month"
                    )[sales_column]
                    .sum()
                )

                st.markdown(
                    "### Monthly Sales Trend"
                )

                st.line_chart(
                    monthly_sales
                )


    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        st.markdown(
            "## Power BI Export Center"
        )

        st.caption(
            "Download cleaned and supporting CSV datasets ready for Power BI Desktop."
        )

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis[
                "sales_column"
            ],
        )

        st.info(
            "Manual Power BI workflow: download the ZIP, "
            "extract the CSV files, open Power BI Desktop, "
            "select Get Data → Text/CSV, and import the "
            "cleaned dataset and supporting tables."
        )

        st.markdown(
            "### Available Exports"
        )

        for filename, data in (
            exports.items()
        ):

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                st.markdown(
                    f"**{filename}**"
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

        st.markdown(
            "### Complete Export Package"
        )

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "📦 Download Power BI ZIP",
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

        st.markdown(
            "## Gemini AI Analysis"
        )

        st.caption(
            "AI-generated interpretation based only on the DataGuard quality report."
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

            st.warning(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY to Streamlit secrets "
                "to enable AI analysis."
            )

        if st.button(
            "✨ Generate Gemini AI Analysis",
            type="primary",
        ):

            with st.spinner(
                "Gemini is analyzing..."
            ):

                result = (
                    get_gemini_analysis(
                        summary_text
                    )
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
                "Click the button above to generate "
                "an AI interpretation of your data-quality report."
            )


    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.markdown(
            "## Data Quality Report"
        )

        st.caption(
            "Portfolio-ready summary of the DataGuard AI analysis."
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

        st.code(
            report_text,
            language="text",
        )

        st.download_button(
            "📄 Download Report",
            data=report_text.encode(
                "utf-8"
            ),
            file_name=(
                "DataGuard_Data_Quality_Report.txt"
            ),
            mime="text/plain",
        )

        if st.session_state.gemini_analysis:

            st.markdown(
                "### Gemini Analysis"
            )

            st.markdown(
                st.session_state.gemini_analysis
            )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.markdown(
        "## Welcome to DataGuard AI"
    )

    st.info(
        "Upload a CSV or Excel dataset above to start "
        "your data-quality analysis."
    )

    st.markdown(
        "### What DataGuard AI does"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            "### 🔍"
        )

        st.markdown(
            "**Profile Data**"
        )

        st.caption(
            "Understand columns, data types, "
            "missing values and identifiers."
        )

    with c2:

        st.markdown(
            "### 🚨"
        )

        st.markdown(
            "**Detect Anomalies**"
        )

        st.caption(
            "Use IQR and Isolation Forest "
            "to identify unusual records."
        )

    with c3:

        st.markdown(
            "### 🧹"
        )

        st.markdown(
            "**Clean Data**"
        )

        st.caption(
            "Remove duplicates and handle "
            "missing values."
        )

    with c4:

        st.markdown(
            "### 📊"
        )

        st.markdown(
            "**Analyze & Export**"
        )

        st.caption(
            "Create business analytics and "
            "Power BI-ready datasets."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br>
        Built with Python, Pandas, Scikit-learn and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
