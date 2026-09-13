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

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background: #ffffff;
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background: #eaf7ea;
        border: 1px solid #b8dfb8;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        background: #fff8e1;
        border: 1px solid #f0d98c;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background: #eef5ff;
        border: 1px solid #bfd6f6;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛡️ DataGuard AI")

st.sidebar.markdown(
    """
    ### Data Quality Pipeline

    1. 📂 Upload Data
    2. 🔍 Profile Dataset
    3. ⚠️ Detect Quality Issues
    4. 📊 Detect Statistical Outliers
    5. 🤖 Detect ML Anomalies
    6. 🧠 Analyze Root Causes
    7. 🧹 Clean Data
    8. 📈 Power BI Export
    9. 📄 Generate Report
    """
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Power BI**

    Manual export is enabled.

    Download the Power BI CSV/ZIP files and import them into Power BI Desktop.
    """
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ DataGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered data quality, anomaly detection, automated cleaning
    and Power BI-ready analytics.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY", ""),
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_to_datetime(series):
    """
    Safely convert a pandas Series to datetime.

    Handles different pandas versions and mixed date formats.
    """

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


# ============================================================
# DATETIME DETECTION
# ============================================================

def detect_datetime_columns(df):

    datetime_columns = []

    date_tokens = [
        "date",
        "time",
        "timestamp",
        "datetime",
        "created",
        "updated",
        "month",
        "year",
    ]

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Do not attempt to interpret numeric ID columns as dates.
        if pd.api.types.is_numeric_dtype(series):
            continue

        column_name = str(column).lower()

        if any(token in column_name for token in date_tokens):

            parsed = safe_to_datetime(series)

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.70:
                datetime_columns.append(column)
                continue

        # Generic date detection only for object/string columns.
        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):

            parsed = safe_to_datetime(series)

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.95:
                datetime_columns.append(column)

    return list(dict.fromkeys(datetime_columns))


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier_columns(df):

    identifier_columns = []

    for column in df.columns:

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

        # High uniqueness detection
        if len(df) > 0:

            unique_ratio = df[column].nunique(
                dropna=True
            ) / len(df)

            if (
                unique_ratio >= 0.98
                and
                (
                    pd.api.types.is_integer_dtype(df[column])
                    or
                    pd.api.types.is_object_dtype(df[column])
                    or
                    pd.api.types.is_string_dtype(df[column])
                )
            ):
                identifier_columns.append(column)

    return list(dict.fromkeys(identifier_columns))


# ============================================================
# DATA PROFILING
# ============================================================

def profile_dataset(df):

    profile = []

    for column in df.columns:

        series = df[column]

        profile.append(
            {
                "Column": column,
                "Data Type": str(series.dtype),
                "Missing": int(series.isna().sum()),
                "Unique": int(series.nunique(dropna=True)),
                "Unique %": round(
                    series.nunique(dropna=True)
                    / max(len(series), 1)
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(profile)


# ============================================================
# INVALID VALUE DETECTION
# ============================================================

def detect_invalid_values(df):

    issues = []

    for column in df.columns:

        name = str(column).lower()

        series_numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        if "age" in name:

            negative_count = int(
                (series_numeric < 0).sum()
            )

            too_high_count = int(
                (series_numeric > 120).sum()
            )

            invalid_count = (
                negative_count
                + too_high_count
            )

            if invalid_count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue": "Invalid age values",
                        "Rows Affected": invalid_count,
                    }
                )

        # ----------------------------------------------------
        # QUANTITY / COUNT
        # ----------------------------------------------------

        elif (
            "quantity" in name
            or "qty" in name
            or "count" in name
        ):

            invalid_count = int(
                (series_numeric < 0).sum()
            )

            if invalid_count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue": "Negative quantity/count",
                        "Rows Affected": invalid_count,
                    }
                )

        # ----------------------------------------------------
        # SALES / AMOUNT / PRICE
        # ----------------------------------------------------

        elif (
            "sales" in name
            or "amount" in name
            or "price" in name
        ):

            invalid_count = int(
                (series_numeric < 0).sum()
            )

            if invalid_count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue": "Negative financial value",
                        "Rows Affected": invalid_count,
                    }
                )

    return pd.DataFrame(
        issues,
        columns=[
            "Column",
            "Issue",
            "Rows Affected",
        ],
    )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

CITY_MAPPING = {

    # Bengaluru
    "blr": "Bengaluru",
    "blr.": "Bengaluru",
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bengalooru": "Bengaluru",

    # Chennai
    "madras": "Chennai",
    "chennai": "Chennai",

    # Mumbai
    "bombay": "Mumbai",
    "mumbai": "Mumbai",

    # Hyderabad
    "hyderabad": "Hyderabad",

    # Delhi
    "delhi": "Delhi",
}


def standardize_city_values(df):

    df = df.copy()

    city_columns = [
        column
        for column in df.columns
        if "city" in str(column).lower()
    ]

    changes = []

    for column in city_columns:

        before = df[column].copy()

        cleaned = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        standardized = cleaned.map(
            lambda value:
                CITY_MAPPING.get(
                    value,
                    value.title()
                    if pd.notna(value)
                    else value,
                )
        )

        df[column] = standardized

        changed = (
            before.fillna("")
            != df[column].fillna("")
        )

        count = int(changed.sum())

        if count > 0:

            changes.append(
                {
                    "Column": column,
                    "Action": "City standardization",
                    "Rows Affected": count,
                }
            )

    return (
        df,
        pd.DataFrame(
            changes,
            columns=[
                "Column",
                "Action",
                "Rows Affected",
            ],
        ),
    )


# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

def detect_iqr_outliers(df):

    results = []
    total_outliers = 0

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (series < lower_bound)
            |
            (series > upper_bound)
        )

        count = int(mask.sum())

        if count > 0:

            total_outliers += count

            results.append(
                {
                    "Column": column,
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "Lower Bound": round(
                        lower_bound,
                        2,
                    ),
                    "Upper Bound": round(
                        upper_bound,
                        2,
                    ),
                    "Outliers": count,
                }
            )

    return (
        pd.DataFrame(results),
        total_outliers,
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.empty:

        return (
            df.copy(),
            0,
            0,
        )

    # Replace infinite values.
    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill missing values with median.
    for column in numeric_df.columns:

        median_value = numeric_df[column].median()

        if pd.isna(median_value):
            median_value = 0

        numeric_df[column] = (
            numeric_df[column]
            .fillna(median_value)
        )

    if len(numeric_df) < 10:

        return (
            df.copy(),
            0,
            len(df),
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_pct / 100,
        random_state=42,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    result_df = df.copy()

    result_df["ML_Anomaly"] = (
        predictions == -1
    )

    anomaly_count = int(
        result_df["ML_Anomaly"].sum()
    )

    normal_count = int(
        (~result_df["ML_Anomaly"]).sum()
    )

    return (
        result_df,
        anomaly_count,
        normal_count,
    )


# ============================================================
# QUALITY SCORE
# ============================================================

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
        * 100
    )

    duplicate_penalty = (
        duplicate_count
        / max(1, total_cells)
        * 100
    )

    invalid_penalty = (
        invalid_count
        / max(1, total_cells)
        * 100
    )

    score = 100 - (
        missing_penalty * 0.50
        +
        duplicate_penalty * 0.30
        +
        invalid_penalty * 0.20
    )

    return round(
        max(0, min(100, score)),
        2,
    )


# ============================================================
# ROOT CAUSE HYPOTHESES
# ============================================================

def build_root_cause_hypotheses(
    df,
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_outliers,
    ml_anomalies,
):

    hypotheses = []

    if missing_count > 0:

        hypotheses.append(
            "Missing values may be caused by incomplete data entry, "
            "optional fields, or upstream extraction gaps."
        )

    if duplicate_count > 0:

        hypotheses.append(
            "Duplicate records may be caused by repeated ingestion, "
            "retry logic, or duplicate source records."
        )

    if invalid_count > 0:

        hypotheses.append(
            "Invalid numeric values may indicate weak validation "
            "rules or inconsistent source-system data."
        )

    if iqr_outliers > 0:

        hypotheses.append(
            "IQR outliers may represent legitimate extreme observations "
            "or unusual transactions and should be business-validated."
        )

    if ml_anomalies > 0:

        hypotheses.append(
            "Isolation Forest anomalies indicate unusual combinations "
            "of numeric features; they are not automatically data errors."
        )

    if not hypotheses:

        hypotheses.append(
            "No major confirmed quality issues were detected "
            "using the implemented checks."
        )

    return hypotheses


# ============================================================
# AUTOMATED CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    actions = []

    # --------------------------------------------------------
    # Standardize city values
    # --------------------------------------------------------

    cleaned, city_changes = (
        standardize_city_values(cleaned)
    )

    if not city_changes.empty:

        actions.extend(
            city_changes.to_dict(
                orient="records"
            )
        )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    before_duplicates = len(cleaned)

    cleaned = cleaned.drop_duplicates()

    duplicate_rows_removed = (
        before_duplicates
        - len(cleaned)
    )

    if duplicate_rows_removed > 0:

        actions.append(
            {
                "Column": "All Columns",
                "Action": "Duplicate row removal",
                "Rows Affected":
                    duplicate_rows_removed,
            }
        )

    # --------------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------------

    numeric_columns = cleaned.select_dtypes(
        include=np.number
    ).columns

    values_filled = 0

    for column in numeric_columns:

        # Convert to float so median values
        # do not create nullable integer errors.
        numeric_series = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        ).astype("float64")

        invalid_mask = pd.Series(
            False,
            index=cleaned.index,
        )

        name = str(column).lower()

        if "age" in name:

            invalid_mask = (
                (numeric_series < 0)
                |
                (numeric_series > 120)
            )

        elif (
            "quantity" in name
            or "qty" in name
            or "count" in name
        ):

            invalid_mask = (
                numeric_series < 0
            )

        elif (
            "sales" in name
            or "amount" in name
            or "price" in name
        ):

            invalid_mask = (
                numeric_series < 0
            )

        numeric_series.loc[
            invalid_mask
        ] = np.nan

        missing_before = int(
            numeric_series.isna().sum()
        )

        if missing_before > 0:

            median_value = (
                numeric_series.median()
            )

            if pd.isna(median_value):
                median_value = 0

            numeric_series = (
                numeric_series.fillna(
                    median_value
                )
            )

            values_filled += missing_before

            actions.append(
                {
                    "Column": column,
                    "Action": "Numeric missing/invalid values filled with median",
                    "Rows Affected":
                        missing_before,
                }
            )

        cleaned[column] = numeric_series

    # --------------------------------------------------------
    # Categorical cleaning
    # --------------------------------------------------------

    categorical_columns = cleaned.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns

    for column in categorical_columns:

        if cleaned[column].isna().sum() == 0:
            continue

        mode_values = (
            cleaned[column]
            .mode(dropna=True)
        )

        if len(mode_values) == 0:
            continue

        mode_value = mode_values.iloc[0]

        missing_before = int(
            cleaned[column].isna().sum()
        )

        cleaned[column] = (
            cleaned[column]
            .fillna(mode_value)
        )

        values_filled += missing_before

        actions.append(
            {
                "Column": column,
                "Action": "Categorical missing values filled with mode",
                "Rows Affected":
                    missing_before,
            }
        )

    cleaned_rows = len(cleaned)

    rows_removed = (
        original_rows
        - cleaned_rows
    )

    action_df = pd.DataFrame(
        actions,
        columns=[
            "Column",
            "Action",
            "Rows Affected",
        ],
    )

    return (
        cleaned,
        action_df,
        original_rows,
        cleaned_rows,
        rows_removed,
        values_filled,
    )


# ============================================================
# FIND SALES COLUMN
# ============================================================

def find_sales_column(df):

    exact_candidates = [
        "Sales",
        "sales",
        "SALES",
        "Revenue",
        "revenue",
        "Amount",
        "amount",
        "Total_Sales",
        "Total Sales",
    ]

    for candidate in exact_candidates:

        if candidate in df.columns:
            return candidate

    for column in df.columns:

        name = str(column).lower()

        if (
            "sales" in name
            or "revenue" in name
            or "amount" in name
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

    return None


# ============================================================
# VALIDATED BUSINESS DATA
# ============================================================

def prepare_business_sales(df):

    sales_column = find_sales_column(df)

    if sales_column is None:
        return None, None, None

    business_df = df.copy()

    sales = pd.to_numeric(
        business_df[sales_column],
        errors="coerce",
    )

    # Keep only non-negative numeric sales.
    valid_sales_mask = (
        sales.notna()
        &
        (sales >= 0)
    )

    business_df = business_df.loc[
        valid_sales_mask
    ].copy()

    sales = sales.loc[
        valid_sales_mask
    ]

    # --------------------------------------------------------
    # IQR bound for business visualization
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # We do NOT delete these values from the actual dataset.
    #
    # This only prevents extreme values from destroying
    # the readability of business charts.
    # --------------------------------------------------------

    if len(sales) >= 5:

        q1 = sales.quantile(0.25)
        q3 = sales.quantile(0.75)

        iqr = q3 - q1

        if iqr > 0:

            upper_bound = (
                q3 + 1.5 * iqr
            )

            chart_mask = (
                sales <= upper_bound
            )

            chart_df = business_df.loc[
                chart_mask
            ].copy()

            return (
                chart_df,
                sales_column,
                upper_bound,
            )

    return (
        business_df,
        sales_column,
        None,
    )


# ============================================================
# POWER BI EXPORTS
# ============================================================

def dataframe_to_csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")


def create_powerbi_exports(
    cleaned_df,
    profile_df,
    outlier_df,
    anomaly_df,
):

    exports = {}

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = dataframe_to_csv_bytes(
        cleaned_df
    )

    exports[
        "DataGuard_Data_Profile.csv"
    ] = dataframe_to_csv_bytes(
        profile_df
    )

    if outlier_df is None:
        outlier_df = pd.DataFrame()

    exports[
        "DataGuard_IQR_Outliers.csv"
    ] = dataframe_to_csv_bytes(
        outlier_df
    )

    if anomaly_df is None:
        anomaly_df = pd.DataFrame()

    exports[
        "DataGuard_ML_Anomalies.csv"
    ] = dataframe_to_csv_bytes(
        anomaly_df
    )

    return exports


def create_zip_file(exports):

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
# GEMINI AI ANALYSIS
# ============================================================

def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:

        return (
            "Gemini AI is not configured.\n\n"
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable AI-generated analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior data quality analyst reviewing a dataset.

IMPORTANT RULES:

1. Use ONLY the information provided in the Dataset Report.
2. Do NOT invent columns, values, errors, or data types.
3. Do NOT assume a column has an incorrect data type unless the report explicitly says so.
4. Do NOT claim fraud, system failure, ETL failure, or business misconduct as a fact.
5. Possible causes must be clearly labelled as hypotheses.
6. Isolation Forest anomalies are NOT automatically data errors.
7. IQR outliers are NOT automatically data errors.
8. Do not criticize the quality score unless the supplied metrics justify it.
9. Do not create unsupported recommendations.
10. Keep the analysis practical and concise.
11. Distinguish between:
   - confirmed data quality issues
   - statistical outliers
   - ML anomalies
   - possible root-cause hypotheses

Provide the following sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Recommended Cleaning Actions
7. Business Impact
8. Power BI Recommendations

Dataset Report:

{summary_text}
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "temperature": 0.2
            },
        )

        return response.output_text

    except Exception as e:

        return (
            "Gemini AI analysis could not be generated.\n\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)


if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel file to start the DataGuard AI pipeline."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        df = pd.read_csv(
            uploaded_file
        )

    else:

        df = pd.read_excel(
            uploaded_file
        )

except Exception as e:

    st.error(
        f"Could not read the uploaded file: {e}"
    )

    st.stop()


# ============================================================
# BASIC INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True,
)

total_rows = len(df)
total_columns = len(df.columns)
total_cells = (
    total_rows
    * total_columns
)

missing_count = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

invalid_df = detect_invalid_values(
    df
)

invalid_count = (
    int(
        invalid_df["Rows Affected"].sum()
    )
    if not invalid_df.empty
    else 0
)

quality_score = build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Rows",
        f"{total_rows:,}",
    )

with col2:

    st.metric(
        "Columns",
        f"{total_columns:,}",
    )

with col3:

    st.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

with col4:

    st.metric(
        "Duplicates",
        f"{duplicate_count:,}",
    )

with col5:

    st.metric(
        "Quality Score",
        f"{quality_score:.2f}/100",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">👀 Data Preview</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    df.head(20),
    use_container_width=True,
)


# ============================================================
# DATA PROFILING
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Data Profiling</div>',
    unsafe_allow_html=True,
)

profile_df = profile_dataset(df)

st.dataframe(
    profile_df,
    use_container_width=True,
)


# ============================================================
# COLUMN TYPES
# ============================================================

datetime_columns = (
    detect_datetime_columns(df)
)

identifier_columns = (
    detect_identifier_columns(df)
)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=[
        "object",
        "string",
        "category",
    ]
).columns.tolist()


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Numeric Columns",
        len(numeric_columns),
    )

with c2:

    st.metric(
        "Categorical Columns",
        len(categorical_columns),
    )

with c3:

    st.metric(
        "Date/Time Columns",
        len(datetime_columns),
    )

with c4:

    st.metric(
        "Identifier Columns",
        len(identifier_columns),
    )


if datetime_columns:

    st.caption(
        "Detected Date/Time Columns: "
        + ", ".join(
            map(str, datetime_columns)
        )
    )


if identifier_columns:

    st.caption(
        "Detected Identifier Columns: "
        + ", ".join(
            map(str, identifier_columns)
        )
    )


# ============================================================
# QUALITY ISSUES
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Data Quality Issues</div>',
    unsafe_allow_html=True,
)


q1, q2, q3 = st.columns(3)

with q1:

    st.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

with q2:

    st.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}",
    )

with q3:

    st.metric(
        "Invalid Values",
        f"{invalid_count:,}",
    )


if missing_count > 0:

    st.warning(
        f"{missing_count:,} missing cells detected."
    )

else:

    st.success(
        "No missing cells detected."
    )


if duplicate_count > 0:

    st.warning(
        f"{duplicate_count:,} duplicate rows detected."
    )

else:

    st.success(
        "No duplicate rows detected."
    )


if not invalid_df.empty:

    st.dataframe(
        invalid_df,
        use_container_width=True,
    )

else:

    st.success(
        "No invalid numeric values detected "
        "by the implemented validation rules."
    )


# ============================================================
# CITY STANDARDIZATION PREVIEW
# ============================================================

standardized_preview_df, city_changes_df = (
    standardize_city_values(df)
)

if not city_changes_df.empty:

    st.markdown(
        '<div class="section-title">🌆 City Standardization</div>',
        unsafe_allow_html=True,
    )

    st.success(
        f"{int(city_changes_df['Rows Affected'].sum()):,} "
        "city values can be standardized."
    )

    st.dataframe(
        city_changes_df,
        use_container_width=True,
    )


# ============================================================
# IQR OUTLIERS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Statistical Outlier Detection</div>',
    unsafe_allow_html=True,
)

iqr_df, total_iqr_outliers = (
    detect_iqr_outliers(df)
)

st.metric(
    "IQR Outlier Observations",
    f"{total_iqr_outliers:,}",
)

if not iqr_df.empty:

    st.dataframe(
        iqr_df,
        use_container_width=True,
    )

    st.info(
        "IQR outliers are statistically unusual observations. "
        "They are not automatically data errors."
    )

else:

    st.success(
        "No IQR outliers detected."
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

st.markdown(
    '<div class="section-title">🤖 ML Anomaly Detection</div>',
    unsafe_allow_html=True,
)

contamination_pct = st.sidebar.slider(
    "Isolation Forest sensitivity",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
)

st.caption(
    f"Current sensitivity: {contamination_pct}%"
)

(
    anomaly_df_full,
    anomaly_count,
    normal_count,
) = detect_ml_anomalies(
    df,
    contamination_pct,
)

st.metric(
    "ML Anomalies",
    f"{anomaly_count:,}",
)

st.metric(
    "Normal Records",
    f"{normal_count:,}",
)

st.info(
    "Isolation Forest identifies unusual patterns across "
    "numeric features. An anomaly is a screening signal, "
    "not proof of an incorrect record."
)


if "ML_Anomaly" in anomaly_df_full.columns:

    anomaly_records = anomaly_df_full[
        anomaly_df_full["ML_Anomaly"] == True
    ].copy()

    if not anomaly_records.empty:

        st.dataframe(
            anomaly_records.head(100),
            use_container_width=True,
        )


# ============================================================
# ROOT CAUSE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Root-Cause Hypotheses</div>',
    unsafe_allow_html=True,
)

hypotheses = build_root_cause_hypotheses(
    df,
    missing_count,
    duplicate_count,
    invalid_count,
    total_iqr_outliers,
    anomaly_count,
)

for hypothesis in hypotheses:

    st.write(
        "• " + hypothesis
    )


# ============================================================
# AUTOMATED CLEANING
# ============================================================

st.markdown(
    '<div class="section-title">🧹 Automated Cleaning</div>',
    unsafe_allow_html=True,
)

(
    cleaned_df,
    cleaning_actions_df,
    original_rows,
    cleaned_rows,
    rows_removed,
    values_filled,
) = clean_dataset(df)


c1, c2, c3, c4 = st.columns(4)

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
        f"{rows_removed:,}",
    )

with c4:

    st.metric(
        "Values Filled",
        f"{values_filled:,}",
    )


if not cleaning_actions_df.empty:

    st.dataframe(
        cleaning_actions_df,
        use_container_width=True,
    )

else:

    st.success(
        "No automated cleaning actions were required."
    )


# ============================================================
# CLEANED DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">✨ Cleaned Data Preview</div>',
    unsafe_allow_html=True,
)

st.dataframe(
    cleaned_df.head(20),
    use_container_width=True,
)


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">📈 Business Analytics</div>',
    unsafe_allow_html=True,
)

(
    business_df,
    sales_column,
    sales_upper_bound,
) = prepare_business_sales(
    cleaned_df
)


if (
    business_df is None
    or sales_column is None
):

    st.warning(
        "No suitable Sales/Revenue/Amount column was found "
        "for business analytics."
    )

else:

    st.caption(
        f"Business charts use validated non-negative "
        f"values from `{sales_column}`."
    )

    if sales_upper_bound is not None:

        st.caption(
            f"For chart readability, extreme values above "
            f"the IQR upper bound ({sales_upper_bound:,.2f}) "
            f"are excluded from these visualizations. "
            f"They remain available in the dataset and "
            f"outlier/anomaly analysis."
        )

    sales_series = pd.to_numeric(
        business_df[sales_column],
        errors="coerce",
    ).dropna()


    # --------------------------------------------------------
    # SALES DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "💰 Sales Distribution"
    )

    if len(sales_series) > 0:

        st.bar_chart(
            sales_series.value_counts(
                bins=10
            ).sort_index()
        )


    # --------------------------------------------------------
    # CITY RECORDS
    # --------------------------------------------------------

    city_columns = [
        column
        for column in business_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns:

        city_column = city_columns[0]

        st.subheader(
            "🌆 Records by City"
        )

        city_data = (
            business_df[city_column]
            .astype("string")
            .value_counts()
        )

        st.bar_chart(
            city_data
        )


    # --------------------------------------------------------
    # SALES BY PRODUCT
    # --------------------------------------------------------

    product_columns = [
        column
        for column in business_df.columns
        if "product" in str(column).lower()
    ]

    if product_columns:

        product_column = product_columns[0]

        st.subheader(
            "📦 Sales by Product"
        )

        product_sales = (
            business_df
            .groupby(product_column)[
                sales_column
            ]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        st.bar_chart(
            product_sales
        )


    # --------------------------------------------------------
    # SALES BY CATEGORY
    # --------------------------------------------------------

    category_columns = [
        column
        for column in business_df.columns
        if "category" in str(column).lower()
    ]

    if category_columns:

        category_column = category_columns[0]

        st.subheader(
            "🗂️ Sales by Category"
        )

        category_sales = (
            business_df
            .groupby(category_column)[
                sales_column
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            category_sales
        )


    # --------------------------------------------------------
    # MONTHLY SALES TREND
    # --------------------------------------------------------

    if datetime_columns:

        date_column = datetime_columns[0]

        business_dates = safe_to_datetime(
            business_df[date_column]
        )

        monthly_df = business_df.copy()

        monthly_df["_Parsed_Date"] = (
            business_dates
        )

        monthly_df = monthly_df.dropna(
            subset=["_Parsed_Date"]
        )

        if not monthly_df.empty:

            monthly_df["Month"] = (
                monthly_df["_Parsed_Date"]
                .dt.to_period("M")
                .astype(str)
            )

            monthly_sales = (
                monthly_df
                .groupby("Month")[
                    sales_column
                ]
                .sum()
                .sort_index()
            )

            st.subheader(
                "📅 Monthly Sales Trend"
            )

            st.line_chart(
                monthly_sales
            )


# ============================================================
# POWER BI EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">📊 Power BI Export</div>',
    unsafe_allow_html=True,
)

# Remove ML helper column from main cleaned dataset.
powerbi_cleaned_df = cleaned_df.copy()

if "ML_Anomaly" in powerbi_cleaned_df.columns:

    powerbi_cleaned_df = (
        powerbi_cleaned_df.drop(
            columns=["ML_Anomaly"]
        )
    )


# IQR export
powerbi_outlier_df = (
    iqr_df.copy()
    if iqr_df is not None
    else pd.DataFrame()
)


# ML anomaly export
powerbi_anomaly_df = (
    anomaly_records.copy()
    if "anomaly_records" in locals()
    else pd.DataFrame()
)


if "ML_Anomaly" in powerbi_anomaly_df.columns:

    powerbi_anomaly_df = (
        powerbi_anomaly_df.drop(
            columns=["ML_Anomaly"]
        )
    )


exports = create_powerbi_exports(
    powerbi_cleaned_df,
    profile_df,
    powerbi_outlier_df,
    powerbi_anomaly_df,
)


zip_bytes = create_zip_file(
    exports
)


st.success(
    "Power BI-ready files generated successfully."
)

st.info(
    """
    **Power BI workflow**

    1. Download the ZIP.
    2. Extract the CSV files.
    3. Open Power BI Desktop.
    4. Select **Get Data → Text/CSV**.
    5. Import the cleaned dataset and supporting tables.
    """
)


st.download_button(
    "📦 Download Power BI Export ZIP",
    data=zip_bytes,
    file_name="DataGuard_AI_PowerBI_Export.zip",
    mime="application/zip",
    use_container_width=True,
)


st.subheader(
    "Individual Power BI Files"
)

for filename, data in exports.items():

    st.download_button(
        f"⬇️ {filename}",
        data=data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# GEMINI AI REPORT
# ============================================================

st.markdown(
    '<div class="section-title">✨ Gemini AI Analysis</div>',
    unsafe_allow_html=True,
)


summary_text = f"""
Dataset name:
{uploaded_file.name}

Dataset dimensions:
Rows: {total_rows}
Columns: {total_columns}
Total cells: {total_cells}

Missing cells:
{missing_count}

Duplicate rows:
{duplicate_count}

Invalid values:
{invalid_count}

Quality score:
{quality_score}/100

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}

Date/Time columns:
{datetime_columns}

Identifier columns:
{identifier_columns}

IQR outlier observations:
{total_iqr_outliers}

Isolation Forest sensitivity:
{contamination_pct}%

Isolation Forest anomalies:
{anomaly_count}

Normal records:
{normal_count}

Automated cleaning:
Original rows: {original_rows}
Cleaned rows: {cleaned_rows}
Rows removed: {rows_removed}
Values filled: {values_filled}

City standardization actions:
{
    int(city_changes_df["Rows Affected"].sum())
    if not city_changes_df.empty
    else 0
}

Detected invalid-value details:
{
    invalid_df.to_dict(orient="records")
    if not invalid_df.empty
    else "None"
}

Detected IQR outlier details:
{
    iqr_df.to_dict(orient="records")
    if iqr_df is not None and not iqr_df.empty
    else "None"
}
"""


with st.expander(
    "View report sent to Gemini"
):

    st.text(
        summary_text
    )


if st.button(
    "🤖 Generate Gemini AI Analysis",
    use_container_width=True,
):

    with st.spinner(
        "Gemini is analyzing the dataset..."
    ):

        gemini_result = (
            get_gemini_analysis(
                summary_text
            )
        )

    st.markdown(
        gemini_result
    )


# ============================================================
# FINAL REPORT
# ============================================================

st.markdown(
    '<div class="section-title">📄 Final Data Quality Report</div>',
    unsafe_allow_html=True,
)

report_lines = []

report_lines.append(
    "# DataGuard AI - Data Quality Report"
)

report_lines.append(
    ""
)

report_lines.append(
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

report_lines.append(
    ""
)

report_lines.append(
    "## Dataset Overview"
)

report_lines.append(
    f"- Rows: {total_rows:,}"
)

report_lines.append(
    f"- Columns: {total_columns:,}"
)

report_lines.append(
    f"- Missing cells: {missing_count:,}"
)

report_lines.append(
    f"- Duplicate rows: {duplicate_count:,}"
)

report_lines.append(
    f"- Invalid values: {invalid_count:,}"
)

report_lines.append(
    f"- Quality score: {quality_score:.2f}/100"
)

report_lines.append(
    ""
)

report_lines.append(
    "## Statistical Analysis"
)

report_lines.append(
    f"- IQR outlier observations: {total_iqr_outliers:,}"
)

report_lines.append(
    ""
)

report_lines.append(
    "## Machine Learning Analysis"
)

report_lines.append(
    f"- Isolation Forest sensitivity: {contamination_pct}%"
)

report_lines.append(
    f"- ML anomalies: {anomaly_count:,}"
)

report_lines.append(
    f"- Normal records: {normal_count:,}"
)

report_lines.append(
    ""
)

report_lines.append(
    "## Automated Cleaning"
)

report_lines.append(
    f"- Original rows: {original_rows:,}"
)

report_lines.append(
    f"- Cleaned rows: {cleaned_rows:,}"
)

report_lines.append(
    f"- Rows removed: {rows_removed:,}"
)

report_lines.append(
    f"- Values filled: {values_filled:,}"
)

report_lines.append(
    ""
)

report_lines.append(
    "## Important Interpretation"
)

report_lines.append(
    "- The quality score is based on confirmed missing, duplicate, and invalid-value checks."
)

report_lines.append(
    "- IQR outliers are statistically unusual observations and are not automatically errors."
)

report_lines.append(
    "- Isolation Forest anomalies are screening signals and require business validation."
)

report_lines.append(
    "- Business charts use validated non-negative sales values and exclude extreme IQR values only for chart readability."
)

final_report = "\n".join(
    report_lines
)

st.download_button(
    "📄 Download Final Report",
    data=final_report,
    file_name="DataGuard_AI_Final_Report.md",
    mime="text/markdown",
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DataGuard AI • Data Quality • Anomaly Detection • "
    "Automated Cleaning • Power BI Analytics"
)
