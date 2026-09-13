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

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
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

    Download the Power BI CSV/ZIP files and import them
    into Power BI Desktop.
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
    AI-powered data quality, anomaly detection,
    automated cleaning and Power BI-ready analytics.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GEMINI API
# ============================================================

GEMINI_API_KEY = st.secrets.get(
    "GEMINI_API_KEY",
    os.getenv("GEMINI_API_KEY", ""),
)


# ============================================================
# SAFE DATETIME
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

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            datetime_columns.append(
                column
            )

            continue

        # Never treat numeric columns as dates.
        if pd.api.types.is_numeric_dtype(
            series
        ):

            continue

        column_name = str(
            column
        ).lower()

        if any(
            token in column_name
            for token in date_tokens
        ):

            parsed = safe_to_datetime(
                series
            )

            valid_ratio = (
                parsed.notna().mean()
            )

            if valid_ratio >= 0.70:

                datetime_columns.append(
                    column
                )

                continue

        if (
            pd.api.types.is_object_dtype(
                series
            )
            or
            pd.api.types.is_string_dtype(
                series
            )
        ):

            parsed = safe_to_datetime(
                series
            )

            valid_ratio = (
                parsed.notna().mean()
            )

            if valid_ratio >= 0.95:

                datetime_columns.append(
                    column
                )

    return list(
        dict.fromkeys(
            datetime_columns
        )
    )


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

        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # Date columns must NEVER be identified as ID columns.
        # ----------------------------------------------------

        if column in datetime_columns:

            continue

        name = str(
            column
        ).lower()

        # Explicit ID names
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

            identifier_columns.append(
                column
            )

            continue

        # High uniqueness detection
        if len(df) > 0:

            unique_ratio = (
                df[column]
                .nunique(dropna=True)
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

                identifier_columns.append(
                    column
                )

    return list(
        dict.fromkeys(
            identifier_columns
        )
    )


# ============================================================
# PROFILING
# ============================================================

def profile_dataset(df):

    profile = []

    for column in df.columns:

        series = df[column]

        profile.append(
            {
                "Column": column,
                "Data Type": str(
                    series.dtype
                ),
                "Missing": int(
                    series.isna().sum()
                ),
                "Unique": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "Unique %": round(
                    series.nunique(
                        dropna=True
                    )
                    /
                    max(len(series), 1)
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(
        profile
    )


# ============================================================
# INVALID VALUES
# ============================================================

def detect_invalid_values(df):

    issues = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if "age" in name:

            count = int(
                (
                    (numeric < 0)
                    |
                    (numeric > 120)
                ).sum()
            )

            if count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue":
                            "Invalid age values",
                        "Rows Affected": count,
                    }
                )

        elif (
            "quantity" in name
            or "qty" in name
            or "count" in name
        ):

            count = int(
                (numeric < 0).sum()
            )

            if count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue":
                            "Negative quantity/count",
                        "Rows Affected": count,
                    }
                )

        elif (
            "sales" in name
            or "amount" in name
            or "price" in name
        ):

            count = int(
                (numeric < 0).sum()
            )

            if count > 0:

                issues.append(
                    {
                        "Column": column,
                        "Issue":
                            "Negative financial value",
                        "Rows Affected": count,
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

        count = int(
            changed.sum()
        )

        if count > 0:

            changes.append(
                {
                    "Column": column,
                    "Action":
                        "City standardization",
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
# IQR OUTLIERS
# ============================================================

def detect_iqr_outliers(df):

    results = []

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

        if len(series) < 5:
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

        lower = (
            q1 - 1.5 * iqr
        )

        upper = (
            q3 + 1.5 * iqr
        )

        mask = (
            (series < lower)
            |
            (series > upper)
        )

        count = int(
            mask.sum()
        )

        if count > 0:

            total_outliers += count

            results.append(
                {
                    "Column": column,
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "Lower Bound":
                        round(lower, 2),
                    "Upper Bound":
                        round(upper, 2),
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

    numeric_df = (
        df.select_dtypes(
            include=np.number
        ).copy()
    )

    if numeric_df.empty:

        return (
            df.copy(),
            0,
            len(df),
        )

    numeric_df = (
        numeric_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    for column in numeric_df.columns:

        median = (
            numeric_df[column]
            .median()
        )

        if pd.isna(median):
            median = 0

        numeric_df[column] = (
            numeric_df[column]
            .fillna(median)
        )

    if len(numeric_df) < 10:

        return (
            df.copy(),
            0,
            len(df),
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

    result = df.copy()

    result["ML_Anomaly"] = (
        predictions == -1
    )

    anomaly_count = int(
        result["ML_Anomaly"].sum()
    )

    normal_count = int(
        (~result["ML_Anomaly"]).sum()
    )

    return (
        result,
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
        / max(
            1,
            total_cells,
        )
        * 100
    )

    invalid_penalty = (
        invalid_count
        / max(
            1,
            total_cells,
        )
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
        max(
            0,
            min(
                100,
                score,
            ),
        ),
        2,
    )


# ============================================================
# ROOT CAUSE HYPOTHESES
# ============================================================

def build_root_cause_hypotheses(
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_outliers,
    ml_anomalies,
):

    hypotheses = []

    if missing_count > 0:

        hypotheses.append(
            "Missing values may be caused by incomplete "
            "data entry, optional fields, or upstream extraction gaps."
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

    if iqr_outliers > 0:

        hypotheses.append(
            "IQR outliers may represent legitimate extreme "
            "observations or unusual transactions and should "
            "be business-validated."
        )

    if ml_anomalies > 0:

        hypotheses.append(
            "Isolation Forest anomalies indicate unusual "
            "combinations of numeric features; they are not "
            "automatically data errors."
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

    original_rows = len(
        cleaned
    )

    actions = []

    # --------------------------------------------------------
    # CITY
    # --------------------------------------------------------

    (
        cleaned,
        city_changes,
    ) = standardize_city_values(
        cleaned
    )

    if not city_changes.empty:

        actions.extend(
            city_changes.to_dict(
                orient="records"
            )
        )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    before = len(
        cleaned
    )

    cleaned = (
        cleaned.drop_duplicates()
    )

    duplicate_removed = (
        before
        - len(cleaned)
    )

    if duplicate_removed > 0:

        actions.append(
            {
                "Column":
                    "All Columns",
                "Action":
                    "Duplicate row removal",
                "Rows Affected":
                    duplicate_removed,
            }
        )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_columns = (
        cleaned.select_dtypes(
            include=np.number
        ).columns
    )

    values_filled = 0

    for column in numeric_columns:

        numeric_series = (
            pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )
            .astype("float64")
        )

        name = str(
            column
        ).lower()

        invalid_mask = pd.Series(
            False,
            index=cleaned.index,
        )

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

            median = (
                numeric_series.median()
            )

            if pd.isna(median):

                median = 0

            numeric_series = (
                numeric_series.fillna(
                    median
                )
            )

            values_filled += (
                missing_before
            )

            actions.append(
                {
                    "Column":
                        column,
                    "Action":
                        "Numeric missing/invalid values filled with median",
                    "Rows Affected":
                        missing_before,
                }
            )

        cleaned[column] = (
            numeric_series
        )

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

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

        missing = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing == 0:
            continue

        modes = (
            cleaned[column]
            .mode(
                dropna=True
            )
        )

        if len(modes) == 0:
            continue

        mode_value = (
            modes.iloc[0]
        )

        cleaned[column] = (
            cleaned[column]
            .fillna(mode_value)
        )

        values_filled += missing

        actions.append(
            {
                "Column":
                    column,
                "Action":
                    "Categorical missing values filled with mode",
                "Rows Affected":
                    missing,
            }
        )

    return (
        cleaned,
        pd.DataFrame(
            actions,
            columns=[
                "Column",
                "Action",
                "Rows Affected",
            ],
        ),
        original_rows,
        len(cleaned),
        original_rows
        - len(cleaned),
        values_filled,
    )


# ============================================================
# FIND SALES COLUMN
# ============================================================

def find_sales_column(df):

    candidates = [
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

    for candidate in candidates:

        if candidate in df.columns:

            return candidate

    for column in df.columns:

        name = str(
            column
        ).lower()

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
# BUSINESS DATA
# ============================================================

def prepare_business_sales(df):

    sales_column = (
        find_sales_column(df)
    )

    if sales_column is None:

        return (
            None,
            None,
            None,
        )

    business_df = (
        df.copy()
    )

    sales = pd.to_numeric(
        business_df[
            sales_column
        ],
        errors="coerce",
    )

    valid_mask = (
        sales.notna()
        &
        (sales >= 0)
    )

    business_df = (
        business_df.loc[
            valid_mask
        ].copy()
    )

    sales = sales.loc[
        valid_mask
    ]

    if len(sales) >= 5:

        q1 = sales.quantile(
            0.25
        )

        q3 = sales.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr > 0:

            upper_bound = (
                q3
                +
                1.5 * iqr
            )

            chart_mask = (
                sales <= upper_bound
            )

            chart_df = (
                business_df.loc[
                    chart_mask
                ].copy()
            )

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
# POWER BI EXPORT
# ============================================================

def dataframe_to_csv_bytes(
    df
):

    return (
        df.to_csv(
            index=False
        ).encode(
            "utf-8"
        )
    )


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

    exports[
        "DataGuard_IQR_Outliers.csv"
    ] = dataframe_to_csv_bytes(
        outlier_df
    )

    exports[
        "DataGuard_ML_Anomalies.csv"
    ] = dataframe_to_csv_bytes(
        anomaly_df
    )

    return exports


def create_zip_file(
    exports
):

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
# GEMINI ANALYSIS
# ============================================================

def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:
        return (
            "⚠️ Gemini AI is not configured.\n\n"
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable AI analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior Data Quality Analyst reviewing
the DataGuard AI dataset report below.

IMPORTANT:

You MUST use ONLY the evidence explicitly provided
in the dataset report.

DO NOT invent facts.

DO NOT assume the exact cause of an issue.

DO NOT claim fraud, user error, ETL failure,
system failure, double-click submission, API retry,
manual entry, or business misconduct as a fact.

If you mention a possible cause, label it explicitly
as a "Hypothesis" and keep it general.

DO NOT create new data-quality problems.

DO NOT invent columns.

DO NOT change any numbers.

DO NOT contradict the supplied metrics.

IMPORTANT DISTINCTIONS:

1. Missing values are confirmed data-quality issues.

2. Duplicate rows are confirmed data-quality issues.

3. Invalid Age and Quantity values are confirmed
   because they were detected by implemented rules.

4. City standardization is a confirmed cleaning action.

5. IQR outliers are statistical observations.
   They are NOT automatically errors.

6. Isolation Forest anomalies are screening signals.
   They are NOT automatically errors.

7. The quality score is calculated from the
   implemented confirmed quality checks.

8. Automated cleaning has ALREADY been performed.
   Do not recommend repeating cleaning that has
   already been completed.

9. The cleaned row count, rows removed and values
   filled are final results from the current pipeline.

10. Business charts exclude extreme Sales values
    only for visualization. Those records are NOT
    deleted from the analytical dataset.

11. Power BI recommendations should match the
    actual DataGuard AI workflow.

Write the analysis using exactly these sections:

### 1. Overall Assessment

Briefly summarize the dataset quality.

### 2. Confirmed Data Quality Problems

Only mention confirmed problems from the report.

### 3. Statistical Outlier Findings

Explain the IQR findings without calling them errors.

### 4. ML Anomaly Findings

Explain Isolation Forest results and clearly state
that anomalies require validation.

### 5. Possible Root Causes

Give only general hypotheses.

Every hypothesis MUST start with:

"Hypothesis:"

### 6. Cleaning Results

Describe what DataGuard AI already cleaned.

Do not recommend repeating completed cleaning.

### 7. Business Impact

Explain possible analytical/reporting impact
without inventing business events.

### 8. Power BI Recommendations

Give practical Power BI recommendations based
ONLY on the supplied dataset report.

Dataset Report:

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

        error_text = str(e)

        if (
            "429" in error_text
            or "quota" in error_text.lower()
            or "too_many_requests" in error_text.lower()
        ):

            return (
                "⚠️ **Gemini AI quota temporarily exceeded.**\n\n"
                "The DataGuard AI pipeline is working correctly, "
                "but the Gemini API request quota has been reached.\n\n"
                "The following DataGuard AI features continue "
                "to work without Gemini:\n\n"
                "- Data profiling\n"
                "- Data quality detection\n"
                "- IQR outlier detection\n"
                "- Isolation Forest anomaly detection\n"
                "- Automated cleaning\n"
                "- Business analytics\n"
                "- Power BI export\n"
                "- Final data-quality report\n\n"
                "Please try Gemini again after the API quota resets."
            )

        return (
            "⚠️ Gemini AI analysis could not be generated.\n\n"
            f"Technical reason: {error_text}"
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

    filename = (
        uploaded_file.name.lower()
    )

    if filename.endswith(
        ".csv"
    ):

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
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True,
)

total_rows = len(
    df
)

total_columns = len(
    df.columns
)

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

invalid_df = (
    detect_invalid_values(
        df
    )
)

invalid_count = (
    int(
        invalid_df[
            "Rows Affected"
        ].sum()
    )
    if not invalid_df.empty
    else 0
)

quality_score = (
    build_quality_score(
        total_cells,
        missing_count,
        duplicate_count,
        invalid_count,
    )
)


c1, c2, c3, c4, c5 = (
    st.columns(5)
)

with c1:

    st.metric(
        "Rows",
        f"{total_rows:,}",
    )

with c2:

    st.metric(
        "Columns",
        f"{total_columns:,}",
    )

with c3:

    st.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

with c4:

    st.metric(
        "Duplicates",
        f"{duplicate_count:,}",
    )

with c5:

    st.metric(
        "Quality Score",
        f"{quality_score:.2f}/100",
    )


# ============================================================
# PREVIEW
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
# PROFILING
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Data Profiling</div>',
    unsafe_allow_html=True,
)

profile_df = (
    profile_dataset(df)
)

st.dataframe(
    profile_df,
    use_container_width=True,
)


# ============================================================
# COLUMN TYPES
# ============================================================

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

numeric_columns = (
    df.select_dtypes(
        include=np.number
    )
    .columns
    .tolist()
)

categorical_columns = (
    df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    )
    .columns
    .tolist()
)


c1, c2, c3, c4 = (
    st.columns(4)
)

with c1:

    st.metric(
        "Numeric Columns",
        len(
            numeric_columns
        ),
    )

with c2:

    st.metric(
        "Categorical Columns",
        len(
            categorical_columns
        ),
    )

with c3:

    st.metric(
        "Date/Time Columns",
        len(
            datetime_columns
        ),
    )

with c4:

    st.metric(
        "Identifier Columns",
        len(
            identifier_columns
        ),
    )


if datetime_columns:

    st.caption(
        "Detected Date/Time Columns: "
        +
        ", ".join(
            map(
                str,
                datetime_columns,
            )
        )
    )


if identifier_columns:

    st.caption(
        "Detected Identifier Columns: "
        +
        ", ".join(
            map(
                str,
                identifier_columns,
            )
        )
    )


# ============================================================
# QUALITY ISSUES
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Data Quality Issues</div>',
    unsafe_allow_html=True,
)


q1, q2, q3 = (
    st.columns(3)
)

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
        "No invalid numeric values detected."
    )


# ============================================================
# CITY
# ============================================================

(
    standardized_preview,
    city_changes_df,
) = standardize_city_values(
    df
)

if not city_changes_df.empty:

    st.markdown(
        '<div class="section-title">🌆 City Standardization</div>',
        unsafe_allow_html=True,
    )

    total_city_changes = int(
        city_changes_df[
            "Rows Affected"
        ].sum()
    )

    st.success(
        f"{total_city_changes:,} city values can be standardized."
    )

    st.dataframe(
        city_changes_df,
        use_container_width=True,
    )


# ============================================================
# IQR
# ============================================================

st.markdown(
    '<div class="section-title">📊 Statistical Outlier Detection</div>',
    unsafe_allow_html=True,
)

iqr_df, total_iqr_outliers = (
    detect_iqr_outliers(
        df
    )
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


if (
    "ML_Anomaly"
    in anomaly_df_full.columns
):

    anomaly_records = (
        anomaly_df_full[
            anomaly_df_full[
                "ML_Anomaly"
            ]
            == True
        ]
        .copy()
    )

    if not anomaly_records.empty:

        st.dataframe(
            anomaly_records.head(
                100
            ),
            use_container_width=True,
        )

else:

    anomaly_records = (
        pd.DataFrame()
    )


# ============================================================
# ROOT CAUSE
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Root-Cause Hypotheses</div>',
    unsafe_allow_html=True,
)

hypotheses = (
    build_root_cause_hypotheses(
        missing_count,
        duplicate_count,
        invalid_count,
        total_iqr_outliers,
        anomaly_count,
    )
)

for hypothesis in hypotheses:

    st.write(
        "• "
        +
        hypothesis
    )


# ============================================================
# CLEANING
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
) = clean_dataset(
    df
)


c1, c2, c3, c4 = (
    st.columns(4)
)

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


# ============================================================
# CLEANED DATA
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
        "No suitable Sales/Revenue/Amount column was found."
    )

else:

    st.caption(
        f"Business charts use validated non-negative values from `{sales_column}`."
    )

    if sales_upper_bound is not None:

        st.caption(
            f"Extreme values above the calculated IQR upper bound "
            f"({sales_upper_bound:,.2f}) are excluded only from "
            f"business visualizations. They remain available in "
            f"the dataset and outlier/anomaly analysis."
        )


    # ========================================================
    # SALES DISTRIBUTION
    # ========================================================

    st.subheader(
        "💰 Sales Distribution"
    )

    sales_values = pd.to_numeric(
        business_df[
            sales_column
        ],
        errors="coerce",
    ).dropna()

    if len(sales_values) > 0:

        # FIX:
        # Use a histogram instead of value_counts(bins=10).
        #
        # This produces cleaner sales distribution bins.
        #

        hist_counts, hist_edges = (
            np.histogram(
                sales_values,
                bins=10,
            )
        )

        labels = []

        for i in range(
            len(hist_edges) - 1
        ):

            labels.append(
                f"{hist_edges[i]:,.0f}"
                f"–"
                f"{hist_edges[i + 1]:,.0f}"
            )

        distribution_df = (
            pd.DataFrame(
                {
                    "Sales Range":
                        labels,
                    "Records":
                        hist_counts,
                }
            )
            .set_index(
                "Sales Range"
            )
        )

        st.bar_chart(
            distribution_df
        )


    # ========================================================
    # CITY
    # ========================================================

    city_columns = [
        column
        for column in business_df.columns
        if "city"
        in str(column).lower()
    ]

    if city_columns:

        city_column = (
            city_columns[0]
        )

        st.subheader(
            "🌆 Records by City"
        )

        city_data = (
            business_df[
                city_column
            ]
            .astype("string")
            .value_counts()
        )

        st.bar_chart(
            city_data
        )


    # ========================================================
    # PRODUCT
    # ========================================================

    product_columns = [
        column
        for column in business_df.columns
        if "product"
        in str(column).lower()
    ]

    if product_columns:

        product_column = (
            product_columns[0]
        )

        st.subheader(
            "📦 Sales by Product"
        )

        product_sales = (
            business_df
            .groupby(
                product_column
            )[
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


    # ========================================================
    # CATEGORY
    # ========================================================

    category_columns = [
        column
        for column in business_df.columns
        if "category"
        in str(column).lower()
    ]

    if category_columns:

        category_column = (
            category_columns[0]
        )

        st.subheader(
            "🗂️ Sales by Category"
        )

        category_sales = (
            business_df
            .groupby(
                category_column
            )[
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


    # ========================================================
    # MONTHLY SALES
    # ========================================================

    if datetime_columns:

        date_column = (
            datetime_columns[0]
        )

        monthly_df = (
            business_df.copy()
        )

        monthly_df[
            "_Parsed_Date"
        ] = safe_to_datetime(
            monthly_df[
                date_column
            ]
        )

        monthly_df = (
            monthly_df.dropna(
                subset=[
                    "_Parsed_Date"
                ]
            )
        )

        if not monthly_df.empty:

            monthly_df[
                "Month"
            ] = (
                monthly_df[
                    "_Parsed_Date"
                ]
                .dt.to_period(
                    "M"
                )
                .astype(str)
            )

            monthly_sales = (
                monthly_df
                .groupby(
                    "Month"
                )[
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

powerbi_cleaned_df = (
    cleaned_df.copy()
)

if (
    "ML_Anomaly"
    in powerbi_cleaned_df.columns
):

    powerbi_cleaned_df = (
        powerbi_cleaned_df.drop(
            columns=[
                "ML_Anomaly"
            ]
        )
    )


powerbi_outlier_df = (
    iqr_df.copy()
)

powerbi_anomaly_df = (
    anomaly_records.copy()
)

if (
    "ML_Anomaly"
    in powerbi_anomaly_df.columns
):

    powerbi_anomaly_df = (
        powerbi_anomaly_df.drop(
            columns=[
                "ML_Anomaly"
            ]
        )
    )


exports = (
    create_powerbi_exports(
        powerbi_cleaned_df,
        profile_df,
        powerbi_outlier_df,
        powerbi_anomaly_df,
    )
)

zip_bytes = (
    create_zip_file(
        exports
    )
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
    file_name=(
        "DataGuard_AI_PowerBI_Export.zip"
    ),
    mime="application/zip",
    use_container_width=True,
)


st.subheader(
    "Individual Power BI Files"
)

for filename, data in (
    exports.items()
):

    st.download_button(
        f"⬇️ {filename}",
        data=data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# GEMINI
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

IQR details:
{
    iqr_df.to_dict(orient="records")
    if not iqr_df.empty
    else "None"
}

Isolation Forest sensitivity:
{contamination_pct}%

Isolation Forest anomalies:
{anomaly_count}

Normal records:
{normal_count}

Automated cleaning results:

Original rows:
{original_rows}

Cleaned rows:
{cleaned_rows}

Rows removed:
{rows_removed}

Values filled:
{values_filled}

City standardization actions:
{
    int(
        city_changes_df["Rows Affected"].sum()
    )
    if not city_changes_df.empty
    else 0
}

Invalid-value details:
{
    invalid_df.to_dict(
        orient="records"
    )
    if not invalid_df.empty
    else "None"
}

Business visualization rule:

Business charts use validated non-negative Sales values.

Extreme Sales values above the calculated IQR upper
bound are excluded from business visualizations only.

Those records are NOT deleted from the analytical dataset.

IQR outliers and Isolation Forest anomalies remain
available for further investigation.

Cleaning has already been performed by DataGuard AI.
Gemini must not recommend repeating completed cleaning.
"""

# ============================================================
# FINAL REPORT
# ============================================================

st.markdown(
    '<div class="section-title">📄 Final Data Quality Report</div>',
    unsafe_allow_html=True,
)

report = f"""
# DataGuard AI - Data Quality Report

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview

- Rows: {total_rows:,}
- Columns: {total_columns:,}
- Missing cells: {missing_count:,}
- Duplicate rows: {duplicate_count:,}
- Invalid values: {invalid_count:,}
- Quality score: {quality_score:.2f}/100

## Statistical Analysis

- IQR outlier observations: {total_iqr_outliers:,}

## Machine Learning Analysis

- Isolation Forest sensitivity: {contamination_pct}%
- ML anomalies: {anomaly_count:,}
- Normal records: {normal_count:,}

## Automated Cleaning

- Original rows: {original_rows:,}
- Cleaned rows: {cleaned_rows:,}
- Rows removed: {rows_removed:,}
- Values filled: {values_filled:,}

## Interpretation

- The quality score is based on confirmed missing,
  duplicate and invalid-value checks.
- IQR outliers are statistically unusual observations
  and are not automatically errors.
- Isolation Forest anomalies are screening signals
  and require business validation.
- Business charts use validated sales values.
- Extreme sales values are excluded only from
  business visualizations, not deleted from the
  analytical dataset.
"""

st.download_button(
    "📄 Download Final Report",
    data=report,
    file_name=(
        "DataGuard_AI_Final_Report.md"
    ),
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
