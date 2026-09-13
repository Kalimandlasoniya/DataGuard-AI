import io
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIGURATION
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
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        color: #777;
    }

    .success-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(40, 167, 69, 0.08);
        border: 1px solid rgba(40, 167, 69, 0.25);
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(0, 123, 255, 0.08);
        border: 1px solid rgba(0, 123, 255, 0.20);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ DataGuard AI")

    st.markdown(
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

    st.divider()

    st.markdown("### Power BI")

    st.info(
        "Manual export is enabled.\n\n"
        "Download the Power BI CSV/ZIP files and "
        "import them into Power BI Desktop."
    )

    st.divider()

    contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.caption(
        "Higher sensitivity flags more records as unusual. "
        "ML anomalies are screening signals, not confirmed errors."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ DataGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered data quality, anomaly detection, automated cleaning "
    "and Power BI-ready analytics."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


# ============================================================
# SAFE DATETIME CONVERSION
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


# ============================================================
# DATETIME DETECTION
# ============================================================

def detect_datetime_columns(df):

    datetime_columns = []

    name_tokens = [
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

        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):

            datetime_columns.append(column)
            continue

        if pd.api.types.is_numeric_dtype(series):

            continue

        column_name = str(column).lower()

        parsed = safe_to_datetime(series)

        parse_ratio = parsed.notna().mean()

        if any(token in column_name for token in name_tokens):

            if parse_ratio >= 0.50:
                datetime_columns.append(column)

        elif parse_ratio >= 0.95:

            datetime_columns.append(column)

    return list(dict.fromkeys(datetime_columns))


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier_columns(
    df,
    datetime_columns=None,
):

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

        if any(
            token in name
            for token in identifier_tokens
        ):

            identifier_columns.append(column)
            continue

        if len(df) > 0:

            unique_ratio = (
                df[column].nunique(dropna=True)
                / len(df)
            )

            if (
                unique_ratio >= 0.98
                and
                (
                    pd.api.types.is_integer_dtype(
                        df[column]
                    )
                    or
                    pd.api.types.is_object_dtype(
                        df[column]
                    )
                    or
                    pd.api.types.is_string_dtype(
                        df[column]
                    )
                )
            ):

                identifier_columns.append(column)

    return list(
        dict.fromkeys(identifier_columns)
    )


# ============================================================
# DATA PROFILING
# ============================================================

def build_profile(df):

    profile = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non-Null Count": [
                df[column].notna().sum()
                for column in df.columns
            ],
            "Missing": [
                df[column].isna().sum()
                for column in df.columns
            ],
            "Unique Values": [
                df[column].nunique(
                    dropna=True
                )
                for column in df.columns
            ],
        }
    )

    return profile


# ============================================================
# INVALID VALUE DETECTION
# ============================================================

def detect_invalid_values(df):

    invalid_details = []
    invalid_count = 0

    for column in df.columns:

        name = str(column).lower()

        numeric_keywords = [
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

        if not any(
            keyword in name
            for keyword in numeric_keywords
        ):
            continue

        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        original_non_null = df[column].notna()

        conversion_failed = (
            numeric_series.isna()
            & original_non_null
        )

        if conversion_failed.any():

            count = int(
                conversion_failed.sum()
            )

            invalid_count += count

            invalid_details.append(
                {
                    "Column": column,
                    "Issue": "Non-numeric value",
                    "Count": count,
                }
            )

        lower_name = name

        if (
            "age" in lower_name
            or "quantity" in lower_name
            or "qty" in lower_name
        ):

            negative_mask = (
                numeric_series < 0
            ).fillna(False)

            if negative_mask.any():

                count = int(
                    negative_mask.sum()
                )

                invalid_count += count

                invalid_details.append(
                    {
                        "Column": column,
                        "Issue": "Negative value",
                        "Count": count,
                    }
                )

        if any(
            keyword in lower_name
            for keyword in [
                "sales",
                "revenue",
                "amount",
                "price",
            ]
        ):

            negative_mask = (
                numeric_series < 0
            ).fillna(False)

            if negative_mask.any():

                count = int(
                    negative_mask.sum()
                )

                invalid_count += count

                invalid_details.append(
                    {
                        "Column": column,
                        "Issue": "Negative business value",
                        "Count": count,
                    }
                )

    return (
        invalid_count,
        pd.DataFrame(invalid_details),
    )


# ============================================================
# CITY STANDARDIZATION
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


def standardize_city_column(df):

    result = df.copy()

    city_columns = [
        column
        for column in result.columns
        if "city" in str(column).lower()
    ]

    standardized_count = 0

    for column in city_columns:

        original = result[column].copy()

        normalized = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        mapped = normalized.map(
            CITY_MAPPING
        )

        final_values = mapped.fillna(
            normalized.str.title()
        )

        changed = (
            original.astype("string")
            != final_values.astype("string")
        )

        changed = changed.fillna(False)

        standardized_count += int(
            changed.sum()
        )

        result[column] = final_values

    return result, standardized_count


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

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        count = int(mask.sum())

        if count > 0:

            total_outliers += count

            results.append(
                {
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "Lower Bound": lower_bound,
                    "Upper Bound": upper_bound,
                    "Outliers": count,
                }
            )

    return (
        total_outliers,
        pd.DataFrame(results),
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
        contamination=contamination_pct / 100,
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
        int(anomaly_mask.sum()),
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
    ) * 100

    duplicate_penalty = (
        duplicate_count
        / max(1, total_cells)
    ) * 100

    invalid_penalty = (
        invalid_count
        / max(1, total_cells)
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


# ============================================================
# ROOT-CAUSE HYPOTHESES
# ============================================================

def generate_root_cause_hypotheses(
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_count,
    ml_anomaly_count,
):

    hypotheses = []

    if missing_count > 0:

        hypotheses.append(
            "Missing values may be caused by incomplete "
            "data entry, optional fields, or upstream "
            "extraction gaps."
        )

    if duplicate_count > 0:

        hypotheses.append(
            "Duplicate records may be caused by repeated "
            "ingestion, retry logic, or duplicate source records."
        )

    if invalid_count > 0:

        hypotheses.append(
            "Invalid numeric values may indicate weak "
            "validation rules or inconsistent source-system data."
        )

    if iqr_count > 0:

        hypotheses.append(
            "IQR outliers may represent legitimate extreme "
            "observations or unusual transactions and should "
            "be business-validated."
        )

    if ml_anomaly_count > 0:

        hypotheses.append(
            "Isolation Forest anomalies indicate unusual "
            "combinations of numeric features; they are not "
            "automatically data errors."
        )

    return hypotheses


# ============================================================
# AUTOMATED CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    cleaned = cleaned.drop_duplicates()

    rows_removed = (
        original_rows
        - len(cleaned)
    )

    # --------------------------------------------------------
    # Standardize cities
    # --------------------------------------------------------

    cleaned, city_changes = (
        standardize_city_column(cleaned)
    )

    # --------------------------------------------------------
    # Numeric conversion and missing-value filling
    # --------------------------------------------------------

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
                    cleaned[column]
                    .fillna(median_value)
                )

                values_filled += (
                    missing_before
                )

    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    categorical_columns = cleaned.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for column in categorical_columns:

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            mode_values = (
                cleaned[column]
                .mode(dropna=True)
            )

            if len(mode_values) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode_values.iloc[0])
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


# ============================================================
# SALES COLUMN DETECTION
# ============================================================

def detect_sales_column(df):

    preferred_names = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue",
    ]

    lower_columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in preferred_names:

        if name in lower_columns:

            return lower_columns[name]

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
# BUSINESS SALES DATA
# ============================================================

def prepare_business_sales(df):

    sales_column = detect_sales_column(df)

    if sales_column is None:

        return (
            pd.DataFrame(),
            None,
            None,
        )

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

        return (
            pd.DataFrame(),
            sales_column,
            None,
        )

    # --------------------------------------------------------
    # IQR filtering ONLY for visualization
    # --------------------------------------------------------

    q1 = business_df[
        sales_column
    ].quantile(0.25)

    q3 = business_df[
        sales_column
    ].quantile(0.75)

    iqr = q3 - q1

    upper_bound = q3 + (
        1.5 * iqr
    )

    chart_df = business_df[
        business_df[sales_column]
        <= upper_bound
    ].copy()

    return (
        chart_df,
        sales_column,
        upper_bound,
    )


# ============================================================
# POWER BI EXPORTS
# ============================================================

def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
):

    exports = {}

    exports["DataGuard_Cleaned_Data.csv"] = (
        cleaned_df.to_csv(index=False)
        .encode("utf-8")
    )

    profile = build_profile(
        cleaned_df
    )

    exports["DataGuard_Data_Profile.csv"] = (
        profile.to_csv(index=False)
        .encode("utf-8")
    )

    if sales_column is not None:

        if (
            sales_column in business_df.columns
        ):

            sales_summary = (
                business_df.groupby(
                    sales_column,
                    dropna=False,
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

    # --------------------------------------------------------
    # City summary
    # --------------------------------------------------------

    city_columns = [
        column
        for column in cleaned_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns:

        city_column = city_columns[0]

        city_summary = (
            cleaned_df.groupby(
                city_column,
                dropna=False,
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

    # --------------------------------------------------------
    # Monthly sales summary
    # --------------------------------------------------------

    datetime_columns = detect_datetime_columns(
        cleaned_df
    )

    if (
        sales_column is not None
        and datetime_columns
    ):

        date_column = datetime_columns[0]

        temp = cleaned_df.copy()

        temp[date_column] = safe_to_datetime(
            temp[date_column]
        )

        temp[sales_column] = pd.to_numeric(
            temp[sales_column],
            errors="coerce",
        )

        monthly = temp.dropna(
            subset=[
                date_column,
                sales_column,
            ]
        ).copy()

        if len(monthly) > 0:

            monthly[
                "Month"
            ] = monthly[
                date_column
            ].dt.to_period("M").astype(str)

            monthly_summary = (
                monthly.groupby("Month")[
                    sales_column
                ]
                .sum()
                .reset_index(
                    name="Total_Sales"
                )
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

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for filename, content in exports.items():

            zip_file.writestr(
                filename,
                content,
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:

        return """
⚠️ **Gemini API key is not configured.**

The core DataGuard AI pipeline is still working correctly.

Add `GEMINI_API_KEY` to Streamlit Secrets if you want to enable
the optional Gemini AI analysis.
"""

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are the AI analysis layer of a professional data-quality
application called DataGuard AI.

Analyze ONLY the supplied report.

Do not invent facts, columns, errors, causes, business events,
fraud, system failures, ETL failures, retry behavior, manual
entry, or other unsupported claims.

Possible causes must be clearly labeled as hypotheses.

Important interpretation rules:

1. Confirmed data-quality problems are based only on the supplied
   missing, duplicate, and invalid-value metrics.

2. IQR outliers are statistically unusual observations.
   They are NOT automatically data errors.

3. Isolation Forest anomalies are screening signals.
   They are NOT automatically incorrect records.

4. Do not contradict any supplied metric.

5. Automated cleaning has already been performed.
   Do not recommend repeating the same cleaning operation.

6. Business charts exclude extreme Sales values only for
   visualization. Those records were NOT deleted from the
   analytical dataset.

7. Keep the analysis practical and suitable for a Data Analyst
   portfolio project.

Use exactly these sections:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations

For possible causes, use wording such as:
"Hypothesis: ..."

Keep the answer concise but useful.

DATA QUALITY REPORT:

{summary_text}
"""

        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
            generation_config={
                "temperature": 0.1
            },
        )

        return response.output_text

    except Exception as e:

        error_text = str(e).lower()

        if (
            "quota" in error_text
            or "429" in error_text
            or "resource_exhausted" in error_text
        ):

            return """
⚠️ **Gemini AI quota temporarily exceeded.**

The Gemini API free-tier request limit has been reached.

The **DataGuard AI pipeline is working correctly** and does not
depend on Gemini for its core functionality.

### Features that continue to work

- Data profiling
- Data quality detection
- IQR outlier detection
- Isolation Forest anomaly detection
- Root-cause hypotheses
- Automated cleaning
- Business analytics
- Power BI export
- Final data-quality report

Please try the Gemini analysis again after the API quota resets.
"""

        return f"""
⚠️ **Gemini AI is temporarily unavailable.**

The core DataGuard AI pipeline is still working correctly.

Technical reason:

`{str(e)}`
"""


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

    file_name = uploaded_file.name

    if file_name.lower().endswith(
        ".csv"
    ):

        df = pd.read_csv(
            uploaded_file
        )

    elif file_name.lower().endswith(
        (".xlsx", ".xls")
    ):

        df = pd.read_excel(
            uploaded_file
        )

    else:

        st.error(
            "Unsupported file format."
        )

        st.stop()

except Exception as e:

    st.error(
        f"Unable to read the uploaded file: {e}"
    )

    st.stop()


if df.empty:

    st.warning(
        "The uploaded dataset is empty."
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📊 Dataset Overview"
    "</div>",
    unsafe_allow_html=True,
)

rows = len(df)
columns = len(df.columns)

missing_count = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

total_cells = (
    rows * columns
)

# Detect types

datetime_columns = (
    detect_datetime_columns(df)
)

identifier_columns = (
    detect_identifier_columns(
        df,
        datetime_columns,
    )
)

numeric_columns = (
    df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

categorical_columns = (
    df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns.tolist()
)

invalid_count, invalid_details = (
    detect_invalid_values(df)
)

quality_score = build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
)


# ============================================================
# OVERVIEW METRICS
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Rows",
        f"{rows:,}",
    )

with col2:

    st.metric(
        "Columns",
        f"{columns:,}",
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
        f"{quality_score}/100",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    "👀 Data Preview"
    "</div>",
    unsafe_allow_html=True,
)

st.dataframe(
    df.head(10),
    use_container_width=True,
)


# ============================================================
# DATA PROFILING
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🔍 Data Profiling"
    "</div>",
    unsafe_allow_html=True,
)

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "Numeric Columns",
        len(numeric_columns),
    )

with p2:

    st.metric(
        "Categorical Columns",
        len(categorical_columns),
    )

with p3:

    st.metric(
        "Date/Time Columns",
        len(datetime_columns),
    )

with p4:

    st.metric(
        "Identifier Columns",
        len(identifier_columns),
    )


if datetime_columns:

    st.write(
        "**Detected Date/Time Columns:** "
        + ", ".join(
            map(str, datetime_columns)
        )
    )

else:

    st.write(
        "**Detected Date/Time Columns:** None"
    )


if identifier_columns:

    st.write(
        "**Detected Identifier Columns:** "
        + ", ".join(
            map(str, identifier_columns)
        )
    )

else:

    st.write(
        "**Detected Identifier Columns:** None"
    )


profile_df = build_profile(df)

with st.expander(
    "View detailed column profile"
):

    st.dataframe(
        profile_df,
        use_container_width=True,
    )


# ============================================================
# DATA QUALITY ISSUES
# ============================================================

st.markdown(
    '<div class="section-title">'
    "⚠️ Data Quality Issues"
    "</div>",
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


if invalid_count > 0:

    st.warning(
        f"{invalid_count:,} invalid values detected."
    )

else:

    st.success(
        "No invalid values detected."
    )


if not invalid_details.empty:

    with st.expander(
        "View invalid-value details"
    ):

        st.dataframe(
            invalid_details,
            use_container_width=True,
        )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🌆 City Standardization"
    "</div>",
    unsafe_allow_html=True,
)

city_preview_df, city_change_count = (
    standardize_city_column(df)
)

if city_change_count > 0:

    st.info(
        f"{city_change_count:,} city values can be standardized."
    )

else:

    st.success(
        "No city standardization changes detected."
    )


# ============================================================
# IQR OUTLIERS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📊 Statistical Outlier Detection"
    "</div>",
    unsafe_allow_html=True,
)

iqr_count, iqr_details = (
    detect_iqr_outliers(df)
)

st.metric(
    "IQR Outlier Observations",
    f"{iqr_count:,}",
)

st.caption(
    "IQR outliers are statistically unusual observations. "
    "They are not automatically data errors."
)

if not iqr_details.empty:

    with st.expander(
        "View IQR details"
    ):

        display_iqr = iqr_details.copy()

        for column in [
            "Q1",
            "Q3",
            "IQR",
            "Lower Bound",
            "Upper Bound",
        ]:

            if column in display_iqr.columns:

                display_iqr[column] = (
                    display_iqr[column]
                    .round(2)
                )

        st.dataframe(
            display_iqr,
            use_container_width=True,
        )


# ============================================================
# ML ANOMALY DETECTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🤖 ML Anomaly Detection"
    "</div>",
    unsafe_allow_html=True,
)

ml_anomaly_mask, ml_anomaly_count = (
    detect_ml_anomalies(
        df,
        contamination_pct,
    )
)

normal_count = (
    len(df)
    - ml_anomaly_count
)

m1, m2 = st.columns(2)

with m1:

    st.metric(
        "ML Anomalies",
        f"{ml_anomaly_count:,}",
    )

with m2:

    st.metric(
        "Normal Records",
        f"{normal_count:,}",
    )

st.write(
    f"**Current sensitivity:** "
    f"{contamination_pct}%"
)

st.caption(
    "Isolation Forest identifies unusual patterns across "
    "numeric features. An anomaly is a screening signal, "
    "not proof of an incorrect record."
)


# ============================================================
# ROOT-CAUSE HYPOTHESES
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🧠 Root-Cause Hypotheses"
    "</div>",
    unsafe_allow_html=True,
)

hypotheses = generate_root_cause_hypotheses(
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_count,
    ml_anomaly_count,
)

if hypotheses:

    for hypothesis in hypotheses:

        st.write(
            f"• {hypothesis}"
        )

else:

    st.success(
        "No major quality issues were detected."
    )


# ============================================================
# AUTOMATED CLEANING
# ============================================================

st.markdown(
    '<div class="section-title">'
    "🧹 Automated Cleaning"
    "</div>",
    unsafe_allow_html=True,
)

(
    cleaned_df,
    rows_removed,
    values_filled,
    city_changes,
) = clean_dataset(df)


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Original Rows",
        f"{len(df):,}",
    )

with c2:

    st.metric(
        "Cleaned Rows",
        f"{len(cleaned_df):,}",
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


if city_changes > 0:

    st.caption(
        f"City standardization changed "
        f"{city_changes:,} values."
    )


# ============================================================
# CLEANED DATA PREVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    "✨ Cleaned Data Preview"
    "</div>",
    unsafe_allow_html=True,
)

st.dataframe(
    cleaned_df.head(10),
    use_container_width=True,
)


# ============================================================
# BUSINESS ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📈 Business Analytics"
    "</div>",
    unsafe_allow_html=True,
)

(
    business_df,
    sales_column,
    sales_iqr_upper,
) = prepare_business_sales(
    cleaned_df
)


if sales_column is None:

    st.warning(
        "No Sales/Revenue/Amount column was detected. "
        "Business sales charts are unavailable."
    )

else:

    st.info(
        f"Business charts use validated non-negative values "
        f"from `{sales_column}`."
    )

    if sales_iqr_upper is not None:

        st.caption(
            "Business charts use the cleaned dataset. "
            f"For visualization only, {sales_column} values above "
            f"the cleaned-data IQR upper bound "
            f"({sales_iqr_upper:,.2f}) are excluded. "
            "These records are not deleted from the analytical dataset."
        )


    # --------------------------------------------------------
    # Sales Distribution
    # --------------------------------------------------------

    st.subheader(
        "💰 Sales Distribution"
    )

    if len(business_df) > 0:

        sales_values = (
            business_df[
                sales_column
            ]
            .dropna()
            .values
        )

        if len(sales_values) > 0:

            counts, bin_edges = np.histogram(
                sales_values,
                bins=10,
            )

            labels = []

            for i in range(
                len(bin_edges) - 1
            ):

                labels.append(
                    f"{bin_edges[i]:,.0f}"
                    f"–"
                    f"{bin_edges[i + 1]:,.0f}"
                )

            distribution_df = pd.DataFrame(
                {
                    "Sales Range": labels,
                    "Records": counts,
                }
            )

            distribution_df = (
                distribution_df
                .set_index("Sales Range")
            )

            st.bar_chart(
                distribution_df
            )


    # --------------------------------------------------------
    # Records by City
    # --------------------------------------------------------

    city_columns_cleaned = [
        column
        for column in cleaned_df.columns
        if "city" in str(column).lower()
    ]

    if city_columns_cleaned:

        city_column = city_columns_cleaned[0]

        st.subheader(
            "🌆 Records by City"
        )

        city_chart = (
            business_df
            .groupby(
                city_column,
                dropna=False,
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        if len(city_chart) > 0:

            st.bar_chart(
                city_chart
            )


    # --------------------------------------------------------
    # Sales by Product
    # --------------------------------------------------------

    product_columns = [
        column
        for column in cleaned_df.columns
        if "product" in str(column).lower()
    ]

    if product_columns:

        product_column = product_columns[0]

        st.subheader(
            "📦 Sales by Product"
        )

        product_sales = (
            business_df
            .groupby(
                product_column,
                dropna=False,
            )[sales_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if len(product_sales) > 0:

            st.bar_chart(
                product_sales
            )


    # --------------------------------------------------------
    # Sales by Category
    # --------------------------------------------------------

    category_columns = [
        column
        for column in cleaned_df.columns
        if "category" in str(column).lower()
    ]

    if category_columns:

        category_column = category_columns[0]

        st.subheader(
            "🗂️ Sales by Category"
        )

        category_sales = (
            business_df
            .groupby(
                category_column,
                dropna=False,
            )[sales_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if len(category_sales) > 0:

            st.bar_chart(
                category_sales
            )


    # --------------------------------------------------------
    # Monthly Sales Trend
    # --------------------------------------------------------

    business_dates = (
        detect_datetime_columns(
            cleaned_df
        )
    )

    if business_dates:

        date_column = business_dates[0]

        st.subheader(
            "📅 Monthly Sales Trend"
        )

        monthly_df = cleaned_df.copy()

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

        monthly_df = monthly_df.dropna(
            subset=[
                date_column,
                sales_column,
            ]
        )

        if len(monthly_df) > 0:

            monthly_df[
                "Month"
            ] = monthly_df[
                date_column
            ].dt.to_period(
                "M"
            ).astype(str)

            monthly_sales = (
                monthly_df
                .groupby(
                    "Month"
                )[sales_column]
                .sum()
            )

            st.line_chart(
                monthly_sales
            )


# ============================================================
# POWER BI EXPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📊 Power BI Export"
    "</div>",
    unsafe_allow_html=True,
)

exports = create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column,
)

zip_bytes = create_zip(
    exports
)

st.success(
    "Power BI-ready files generated successfully."
)

st.markdown(
    """
    ### Power BI workflow

    1. Download the ZIP.
    2. Extract the CSV files.
    3. Open Power BI Desktop.
    4. Select **Get Data → Text/CSV**.
    5. Import the cleaned dataset and supporting tables.
    """
)

st.download_button(
    label="⬇️ Download Power BI ZIP",
    data=zip_bytes,
    file_name="DataGuard_PowerBI_Exports.zip",
    mime="application/zip",
    type="primary",
)


# ============================================================
# INDIVIDUAL POWER BI FILES
# ============================================================

st.subheader(
    "📊 Individual Power BI Files"
)

for filename, content in exports.items():

    st.download_button(
        label=f"⬇️ {filename}",
        data=content,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


# ============================================================
# GEMINI SUMMARY TEXT
# ============================================================

summary_text = f"""
Dataset Name: {uploaded_file.name}

Dataset Dimensions:
Rows: {len(df)}
Columns: {len(df.columns)}

Confirmed Data Quality Metrics:
Missing Cells: {missing_count}
Duplicate Rows: {duplicate_count}
Invalid Values: {invalid_count}
Quality Score: {quality_score}/100

Detected Structure:
Numeric Columns: {len(numeric_columns)}
Categorical Columns: {len(categorical_columns)}
Date/Time Columns: {len(datetime_columns)}
Identifier Columns: {len(identifier_columns)}

Date/Time Columns:
{", ".join(map(str, datetime_columns)) if datetime_columns else "None"}

Identifier Columns:
{", ".join(map(str, identifier_columns)) if identifier_columns else "None"}

City Standardization:
{city_changes} values standardized.

IQR Statistical Outliers:
Total observations flagged: {iqr_count}

IQR Details:
{iqr_details.to_string(index=False) if not iqr_details.empty else "None"}

Isolation Forest:
Sensitivity: {contamination_pct}%
ML anomalies: {ml_anomaly_count}
Normal records: {normal_count}

Automated Cleaning:
Original rows: {len(df)}
Cleaned rows: {len(cleaned_df)}
Rows removed: {rows_removed}
Values filled: {values_filled}

Business Analytics:
Sales column: {sales_column if sales_column is not None else "Not detected"}

Visualization Rule:
Sales values above the cleaned-data IQR upper bound are excluded
from business charts for visualization only. They are NOT deleted
from the analytical dataset.

Important:
Cleaning has already been performed by DataGuard AI.
IQR outliers and ML anomalies are not automatically confirmed errors.
"""


# ============================================================
# GEMINI AI ANALYSIS — BUTTON BASED
# ============================================================

st.markdown(
    '<div class="section-title">'
    "✨ Gemini AI Analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Gemini provides optional AI interpretation of the "
    "data-quality results. The core DataGuard AI pipeline "
    "works independently of Gemini."
)


if "gemini_analysis" not in st.session_state:

    st.session_state.gemini_analysis = None


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

    st.session_state.gemini_analysis = result


if st.session_state.gemini_analysis:

    st.markdown(
        st.session_state.gemini_analysis
    )

else:

    st.info(
        "Click **Generate Gemini AI Analysis** "
        "to request an AI-generated interpretation "
        "of the data-quality results."
    )


# ============================================================
# FINAL DATA QUALITY REPORT
# ============================================================

st.markdown(
    '<div class="section-title">'
    "📄 Final Data Quality Report"
    "</div>",
    unsafe_allow_html=True,
)

report_text = f"""
DataGuard AI • Data Quality • Anomaly Detection • Automated Cleaning • Power BI Analytics

============================================================
DATASET OVERVIEW
============================================================

Dataset: {uploaded_file.name}

Rows: {len(df):,}
Columns: {len(df.columns):,}

Missing Cells: {missing_count:,}
Duplicate Rows: {duplicate_count:,}
Invalid Values: {invalid_count:,}

Quality Score: {quality_score}/100


============================================================
DATA STRUCTURE
============================================================

Numeric Columns: {len(numeric_columns)}
Categorical Columns: {len(categorical_columns)}
Date/Time Columns: {len(datetime_columns)}
Identifier Columns: {len(identifier_columns)}

Date/Time Columns:
{", ".join(map(str, datetime_columns)) if datetime_columns else "None"}

Identifier Columns:
{", ".join(map(str, identifier_columns)) if identifier_columns else "None"}


============================================================
STATISTICAL OUTLIERS
============================================================

IQR Outlier Observations: {iqr_count:,}

IQR outliers are statistically unusual observations.
They are not automatically data errors.


============================================================
ML ANOMALIES
============================================================

Isolation Forest Sensitivity: {contamination_pct}%

ML Anomalies: {ml_anomaly_count:,}
Normal Records: {normal_count:,}

Isolation Forest anomalies are screening signals and
are not automatically incorrect records.


============================================================
AUTOMATED CLEANING
============================================================

Original Rows: {len(df):,}
Cleaned Rows: {len(cleaned_df):,}
Rows Removed: {rows_removed:,}
Values Filled: {values_filled:,}

City Values Standardized: {city_changes:,}


============================================================
BUSINESS ANALYTICS
============================================================

Sales Column:
{sales_column if sales_column is not None else "Not detected"}

Business charts use the cleaned dataset.

Extreme Sales values above the cleaned-data IQR upper bound
are excluded from visualizations only.

These records are not deleted from the analytical dataset.


============================================================
POWER BI
============================================================

Power BI-ready CSV files were generated.

Recommended workflow:

1. Download the DataGuard Power BI ZIP.
2. Extract the CSV files.
3. Open Power BI Desktop.
4. Select Get Data → Text/CSV.
5. Import the cleaned dataset and supporting tables.


============================================================
GEMINI AI
============================================================

Gemini analysis is optional.

The core DataGuard AI pipeline operates independently
of the Gemini API.


============================================================
END OF REPORT
============================================================

Generated by DataGuard AI
"""


st.download_button(
    label="📄 Download Final Data Quality Report",
    data=report_text,
    file_name="DataGuard_Final_Data_Quality_Report.txt",
    mime="text/plain",
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

st.markdown("---")

st.success(
    "✅ DataGuard AI pipeline completed successfully."
)

st.caption(
    "DataGuard AI • Data Quality • Anomaly Detection • "
    "Automated Cleaning • Power BI Analytics"
)
