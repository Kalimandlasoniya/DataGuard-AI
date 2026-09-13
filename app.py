import io
import os
import zipfile
import traceback
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
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        min-height: 120px;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .pipeline {
        padding: 18px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        line-height: 2;
    }

    .section-note {
        color: #6b7280;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APP HEADER
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
# SETTINGS
# ============================================================

def get_setting(name: str, default: str = "") -> str:

    value = os.environ.get(name)

    if value:
        return value

    try:
        value = st.secrets.get(name)

        if value is not None:
            return str(value)

    except Exception:
        pass

    return default


GEMINI_API_KEY = get_setting("GEMINI_API_KEY")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(
        "Upload a CSV or Excel dataset to run the "
        "complete DataGuard AI pipeline."
    )

    st.divider()

    st.subheader("🤖 AI Analysis")

    ai_enabled = st.checkbox(
        "Enable Gemini AI analysis",
        value=True,
    )

    st.divider()

    st.subheader("🔎 Anomaly Detection")

    contamination_pct = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "Approximate percentage of records to flag "
            "as statistical anomalies."
        ),
    )

    st.caption(
        f"Current sensitivity: approximately "
        f"{contamination_pct}%"
    )

    st.divider()

    st.subheader("📊 Power BI")

    st.success(
        "Manual Power BI export enabled"
    )

    st.caption(
        "Download the Power BI-ready ZIP and "
        "import the CSV files into Power BI Desktop."
    )


# ============================================================
# PIPELINE
# ============================================================

def show_pipeline():

    st.markdown(
        """
        <div class="pipeline">

        <b>DataGuard AI Pipeline</b><br><br>

        1. Upload Data →
        2. Profile Dataset →
        3. Detect Quality Issues →
        4. Detect Statistical Outliers →
        5. Detect ML Anomalies →
        6. Analyze Root Causes →
        7. Clean Data →
        8. Power BI Export →
        9. Generate Report

        </div>
        """,
        unsafe_allow_html=True,
    )


show_pipeline()

st.write("")


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


# ============================================================
# DATETIME DETECTION
# ============================================================

def detect_datetime_columns(df):

    datetime_columns = []

    date_tokens = [
        "date",
        "time",
        "datetime",
        "timestamp",
        "day",
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

        valid_ratio = parsed.notna().mean()

        if any(
            token in column_name
            for token in date_tokens
        ):

            if valid_ratio >= 0.50:

                datetime_columns.append(column)

        elif valid_ratio >= 0.95:

            datetime_columns.append(column)

    return datetime_columns


# ============================================================
# NUMERIC / CATEGORICAL DETECTION
# ============================================================

def detect_numeric_columns(df):

    return list(
        df.select_dtypes(
            include=np.number
        ).columns
    )


def detect_categorical_columns(df):

    return list(
        df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
    )


def detect_identifier_columns(df):

    identifiers = []

    for column in df.columns:

        name = str(column).lower()

        unique_ratio = (
            df[column].nunique(dropna=True)
            / max(len(df), 1)
        )

        if (
            "id" in name
            or "identifier" in name
            or unique_ratio > 0.98
        ):

            identifiers.append(column)

    return identifiers


# ============================================================
# FILE LOADING
# ============================================================

def load_file(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx"):

        return pd.read_excel(uploaded_file)

    if file_name.endswith(".xls"):

        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file format."
    )


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
                "Non-Null": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Unique": int(series.nunique(dropna=True)),
                "Missing %": round(
                    series.isna().mean() * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(profile)


# ============================================================
# INVALID VALUES
# ============================================================

def detect_invalid_values(df):

    invalid_records = []

    for column in df.columns:

        series = df[column]

        # Negative values for common positive-value fields
        name = str(column).lower()

        if any(
            token in name
            for token in [
                "age",
                "quantity",
                "count",
                "amount",
                "price",
                "sales",
            ]
        ):

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            negative_mask = numeric < 0

            count = int(
                negative_mask.fillna(False).sum()
            )

            if count > 0:

                invalid_records.append(
                    {
                        "Column": column,
                        "Issue": "Negative value",
                        "Count": count,
                    }
                )

        # Age validation
        if "age" in name:

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            invalid_age = (
                (numeric < 0)
                | (numeric > 120)
            )

            count = int(
                invalid_age.fillna(False).sum()
            )

            if count > 0:

                invalid_records.append(
                    {
                        "Column": column,
                        "Issue": "Invalid age",
                        "Count": count,
                    }
                )

    return pd.DataFrame(
        invalid_records,
        columns=[
            "Column",
            "Issue",
            "Count",
        ],
    )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

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

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.title()
        )

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

    return df, pd.DataFrame(
        changes,
        columns=[
            "Column",
            "Action",
            "Rows Affected",
        ],
    )


# ============================================================
# IQR OUTLIERS
# ============================================================

def detect_iqr_outliers(df):

    results = []
    masks = {}

    numeric_columns = detect_numeric_columns(df)

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
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .lt(lower)
            |
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .gt(upper)
        )

        count = int(mask.sum())

        masks[column] = mask

        if count > 0:

            results.append(
                {
                    "Column": column,
                    "Q1": round(q1, 4),
                    "Q3": round(q3, 4),
                    "IQR": round(iqr, 4),
                    "Lower Bound": round(
                        lower,
                        4,
                    ),
                    "Upper Bound": round(
                        upper,
                        4,
                    ),
                    "Outliers": count,
                }
            )

    return (
        pd.DataFrame(
            results,
            columns=[
                "Column",
                "Q1",
                "Q3",
                "IQR",
                "Lower Bound",
                "Upper Bound",
                "Outliers",
            ],
        ),
        masks,
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

def detect_ml_anomalies(
    df,
    contamination_pct,
):

    numeric_columns = detect_numeric_columns(df)

    if len(numeric_columns) < 1:
        return (
            pd.DataFrame(),
            pd.Series(
                1,
                index=df.index,
            ),
        )

    work = df[numeric_columns].copy()

    for column in numeric_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

        median = work[column].median()

        if pd.isna(median):
            median = 0

        work[column] = work[column].fillna(
            median
        )

    contamination = contamination_pct / 100

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(work)

    scores = model.decision_function(work)

    anomaly_mask = predictions == -1

    results = df.copy()

    results["Anomaly"] = np.where(
        anomaly_mask,
        "Anomaly",
        "Normal",
    )

    results["Anomaly_Score"] = scores

    return results, pd.Series(
        predictions,
        index=df.index,
    )


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

def build_root_cause_analysis(
    df,
    missing_count,
    duplicate_count,
    invalid_count,
    iqr_results,
    anomaly_count,
):

    causes = []

    if missing_count > 0:

        causes.append(
            {
                "Issue": "Missing values",
                "Count": missing_count,
                "Possible Cause":
                    "Incomplete data entry or missing source-system values.",
            }
        )

    if duplicate_count > 0:

        causes.append(
            {
                "Issue": "Duplicate records",
                "Count": duplicate_count,
                "Possible Cause":
                    "Repeated records during data collection or data integration.",
            }
        )

    if invalid_count > 0:

        causes.append(
            {
                "Issue": "Invalid values",
                "Count": invalid_count,
                "Possible Cause":
                    "Incorrect input, inconsistent business rules, or data-entry errors.",
            }
        )

    if not iqr_results.empty:

        for _, row in iqr_results.iterrows():

            causes.append(
                {
                    "Issue":
                        f"IQR outliers - {row['Column']}",
                    "Count":
                        int(row["Outliers"]),
                    "Possible Cause":
                        "Extreme observations requiring business validation.",
                }
            )

    if anomaly_count > 0:

        causes.append(
            {
                "Issue": "ML anomalies",
                "Count": anomaly_count,
                "Possible Cause":
                    "Unusual combinations of numeric features detected by Isolation Forest.",
            }
        )

    if not causes:

        causes.append(
            {
                "Issue": "No major issue detected",
                "Count": 0,
                "Possible Cause":
                    "Dataset appears relatively consistent.",
            }
        )

    return pd.DataFrame(causes)


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
        missing_count / total_cells
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
# AUTOMATED CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    cleaning_actions = []

    original_rows = len(cleaned)

    # --------------------------------------------------------
    # Standardize city fields
    # --------------------------------------------------------

    cleaned, city_log = standardize_city_values(
        cleaned
    )

    if not city_log.empty:

        for _, row in city_log.iterrows():

            cleaning_actions.append(
                {
                    "Column": row["Column"],
                    "Action": row["Action"],
                    "Rows Affected":
                        row["Rows Affected"],
                }
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    duplicate_count = int(
        cleaned.duplicated().sum()
    )

    if duplicate_count > 0:

        cleaned = cleaned.drop_duplicates()

        cleaning_actions.append(
            {
                "Column": "All Columns",
                "Action": "Removed duplicate rows",
                "Rows Affected": duplicate_count,
            }
        )

    # --------------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------------

    numeric_columns = detect_numeric_columns(
        cleaned
    )

    for column in numeric_columns:

        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        ).astype("float64")

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            median = cleaned[column].median()

            if not pd.isna(median):

                cleaned[column] = cleaned[
                    column
                ].fillna(median)

                cleaning_actions.append(
                    {
                        "Column": column,
                        "Action":
                            "Filled missing numeric values with median",
                        "Rows Affected":
                            missing_before,
                    }
                )

    # --------------------------------------------------------
    # Categorical cleaning
    # --------------------------------------------------------

    categorical_columns = (
        detect_categorical_columns(
            cleaned
        )
    )

    for column in categorical_columns:

        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
        )

        missing_before = int(
            cleaned[column].isna().sum()
        )

        if missing_before > 0:

            mode = cleaned[column].mode(
                dropna=True
            )

            if len(mode) > 0:

                cleaned[column] = cleaned[
                    column
                ].fillna(mode.iloc[0])

                cleaning_actions.append(
                    {
                        "Column": column,
                        "Action":
                            "Filled missing categorical values with mode",
                        "Rows Affected":
                            missing_before,
                    }
                )

    # --------------------------------------------------------
    # Date cleaning
    # --------------------------------------------------------

    datetime_columns = detect_datetime_columns(
        cleaned
    )

    for column in datetime_columns:

        parsed = safe_to_datetime(
            cleaned[column]
        )

        invalid_dates = int(
            parsed.isna().sum()
            - cleaned[column].isna().sum()
        )

        cleaned[column] = parsed

        if invalid_dates > 0:

            cleaning_actions.append(
                {
                    "Column": column,
                    "Action":
                        "Converted invalid dates to missing",
                    "Rows Affected":
                        invalid_dates,
                }
            )

    cleaned_rows = len(cleaned)

    cleaning_log = pd.DataFrame(
        cleaning_actions,
        columns=[
            "Column",
            "Action",
            "Rows Affected",
        ],
    )

    summary = {
        "Original Rows": original_rows,
        "Cleaned Rows": cleaned_rows,
        "Rows Removed":
            original_rows - cleaned_rows,
        "Values Filled":
            int(
                cleaning_log[
                    "Rows Affected"
                ].sum()
            )
            if not cleaning_log.empty
            else 0,
    }

    return (
        cleaned,
        cleaning_log,
        summary,
    )


# ============================================================
# GEMINI AI
# ============================================================

def get_gemini_analysis(summary_text):

    if not GEMINI_API_KEY:
        return (
            "Gemini AI is not configured. "
            "Add GEMINI_API_KEY to Streamlit Secrets "
            "to enable AI-generated analysis."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
You are a senior data quality analyst.

Analyze the following dataset quality report.

Provide:

1. Overall assessment
2. Most important data-quality problems
3. Possible root causes
4. Recommended cleaning actions
5. Business impact
6. Recommendations for Power BI reporting

Keep the explanation practical, concise, and easy
for a data analyst to understand.

Dataset report:

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
# POWER BI EXPORT
# ============================================================

def dataframe_to_csv_bytes(df):

    return df.to_csv(
        index=False
    ).encode("utf-8")


def create_powerbi_exports(
    quality_summary,
    quality_issues,
    anomaly_results,
    cleaning_log,
    cleaned_df,
):

    files = {}

    files[
        "quality_summary.csv"
    ] = dataframe_to_csv_bytes(
        quality_summary
    )

    files[
        "quality_issues.csv"
    ] = dataframe_to_csv_bytes(
        quality_issues
    )

    files[
        "anomaly_results.csv"
    ] = dataframe_to_csv_bytes(
        anomaly_results
    )

    files[
        "cleaning_log.csv"
    ] = dataframe_to_csv_bytes(
        cleaning_log
    )

    files[
        "cleaned_data.csv"
    ] = dataframe_to_csv_bytes(
        cleaned_df
    )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for filename, data in files.items():

            zip_file.writestr(
                filename,
                data,
            )

    zip_buffer.seek(0)

    return (
        files,
        zip_buffer.getvalue(),
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel dataset to start the DataGuard AI pipeline."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading dataset..."
    ):

        df = load_file(
            uploaded_file
        )

except Exception as e:

    st.error(
        "Unable to read the uploaded file."
    )

    st.exception(e)

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

st.header("📋 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Rows",
        f"{len(df):,}",
    )

with col2:

    st.metric(
        "Columns",
        f"{len(df.columns):,}",
    )

with col3:

    st.metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}",
    )

with col4:

    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}",
    )


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader("👀 Data Preview")

st.dataframe(
    df.head(10),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DATA PROFILE
# ============================================================

st.header("🔍 Data Profiling")

profile = profile_dataset(df)

st.dataframe(
    profile,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# COLUMN CATEGORIES
# ============================================================

numeric_columns = detect_numeric_columns(
    df
)

categorical_columns = (
    detect_categorical_columns(df)
)

datetime_columns = detect_datetime_columns(
    df
)

identifier_columns = detect_identifier_columns(
    df
)


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


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

st.header("🚨 Data Quality Checks")


missing_count = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

invalid_results = detect_invalid_values(
    df
)

if invalid_results.empty:

    invalid_count = 0

else:

    invalid_count = int(
        invalid_results["Count"].sum()
    )


quality_score = build_quality_score(
    total_cells=df.shape[0] * df.shape[1],
    missing_count=missing_count,
    duplicate_count=duplicate_count,
    invalid_count=invalid_count,
)


q1, q2, q3, q4 = st.columns(4)

with q1:

    st.metric(
        "Quality Score",
        f"{quality_score:.2f}/100",
    )

with q2:

    st.metric(
        "Missing Cells",
        f"{missing_count:,}",
    )

with q3:

    st.metric(
        "Duplicate Rows",
        f"{duplicate_count:,}",
    )

with q4:

    st.metric(
        "Invalid Values",
        f"{invalid_count:,}",
    )


# ============================================================
# MISSING VALUES
# ============================================================

st.subheader("Missing Values by Column")

missing_table = (
    df.isna()
    .sum()
    .reset_index()
)

missing_table.columns = [
    "Column",
    "Missing Values",
]

missing_table["Missing %"] = (
    missing_table["Missing Values"]
    / max(len(df), 1)
    * 100
).round(2)


st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True,
)


missing_chart = missing_table[
    missing_table["Missing Values"] > 0
].set_index("Column")[
    "Missing Values"
]

if not missing_chart.empty:

    st.bar_chart(
        missing_chart
    )


# ============================================================
# DUPLICATES
# ============================================================

st.subheader("Duplicate Records")

if duplicate_count > 0:

    st.warning(
        f"{duplicate_count:,} duplicate rows detected."
    )

    duplicate_preview = df[
        df.duplicated(
            keep=False
        )
    ].head(20)

    st.dataframe(
        duplicate_preview,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No duplicate records detected."
    )


# ============================================================
# INVALID VALUES
# ============================================================

st.subheader("Invalid Values")

if invalid_results.empty:

    st.success(
        "No invalid values detected using the configured validation rules."
    )

else:

    st.dataframe(
        invalid_results,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

st.header("🏙️ City Standardization")

city_columns = [
    column
    for column in df.columns
    if "city" in str(column).lower()
]

if city_columns:

    city_issue_count = 0

    for column in city_columns:

        stripped = (
            df[column]
            .astype("string")
            .str.strip()
        )

        standardized = stripped.str.title()

        city_issue_count += int(
            (
                stripped.fillna("")
                != standardized.fillna("")
            ).sum()
        )

    if city_issue_count > 0:

        st.warning(
            f"{city_issue_count:,} city values "
            "can be standardized."
        )

    else:

        st.success(
            "City values already appear standardized."
        )

else:

    st.info(
        "No city column detected."
    )


# ============================================================
# IQR OUTLIERS
# ============================================================

st.header("📈 Statistical Outlier Detection")

iqr_results, iqr_masks = detect_iqr_outliers(
    df
)

if iqr_results.empty:

    st.success(
        "No IQR outliers detected."
    )

else:

    st.dataframe(
        iqr_results,
        use_container_width=True,
        hide_index=True,
    )

    total_iqr_outliers = int(
        iqr_results["Outliers"].sum()
    )

    st.info(
        f"{total_iqr_outliers:,} outlier observations "
        "were detected using the IQR method."
    )


# ============================================================
# SALES ANALYSIS
# ============================================================

st.header("💰 Business Analytics")


if "Sales" in df.columns:

    sales = pd.to_numeric(
        df["Sales"],
        errors="coerce",
    )

    st.subheader("Sales Distribution")

    sales_bins = (
        sales.dropna()
        .value_counts(
            bins=10
        )
        .sort_index()
    )

    if not sales_bins.empty:

        sales_bins.index = (
            sales_bins.index
            .astype(str)
        )

        st.bar_chart(
            sales_bins
        )


# ============================================================
# RECORDS BY CITY
# ============================================================

city_column = None

for column in df.columns:

    if str(column).lower() == "city":

        city_column = column
        break

if city_column is None:

    for column in df.columns:

        if "city" in str(column).lower():

            city_column = column
            break


if city_column is not None:

    st.subheader("Records by City")

    city_counts = (
        df[city_column]
        .astype("string")
        .value_counts()
        .head(15)
    )

    st.bar_chart(
        city_counts
    )


# ============================================================
# SALES BY PRODUCT
# ============================================================

if (
    "Sales" in df.columns
    and "Product" in df.columns
):

    st.subheader("Sales by Product")

    product_sales = (
        df.assign(
            Sales=pd.to_numeric(
                df["Sales"],
                errors="coerce",
            )
        )
        .groupby(
            "Product",
            dropna=False,
        )["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(15)
    )

    st.bar_chart(
        product_sales
    )


# ============================================================
# SALES BY CATEGORY
# ============================================================

if (
    "Sales" in df.columns
    and "Category" in df.columns
):

    st.subheader("Sales by Category")

    category_sales = (
        df.assign(
            Sales=pd.to_numeric(
                df["Sales"],
                errors="coerce",
            )
        )
        .groupby(
            "Category",
            dropna=False,
        )["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_sales
    )


# ============================================================
# MONTHLY SALES TREND
# ============================================================

date_column = None

for column in datetime_columns:

    if (
        "order" in str(column).lower()
        and "date" in str(column).lower()
    ):

        date_column = column
        break

if date_column is None and datetime_columns:

    date_column = datetime_columns[0]


if (
    date_column is not None
    and "Sales" in df.columns
):

    st.subheader("📅 Monthly Sales Trend")

    trend = df.copy()

    trend["_Date"] = safe_to_datetime(
        trend[date_column]
    )

    trend["Sales"] = pd.to_numeric(
        trend["Sales"],
        errors="coerce",
    )

    monthly_sales = (
        trend.dropna(
            subset=["_Date"]
        )
        .assign(
            Month=lambda x:
                x["_Date"]
                .dt.to_period("M")
                .astype(str)
        )
        .groupby("Month")["Sales"]
        .sum()
    )

    if not monthly_sales.empty:

        st.line_chart(
            monthly_sales
        )


# ============================================================
# MACHINE LEARNING ANOMALIES
# ============================================================

st.header("🤖 ML Anomaly Detection")

with st.spinner(
    "Running Isolation Forest..."
):

    anomaly_export, predictions = (
        detect_ml_anomalies(
            df,
            contamination_pct,
        )
    )


if not anomaly_export.empty:

    anomaly_count = int(
        (
            anomaly_export["Anomaly"]
            == "Anomaly"
        ).sum()
    )

    normal_count = int(
        (
            anomaly_export["Anomaly"]
            == "Normal"
        ).sum()
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        st.metric(
            "Sensitivity",
            f"{contamination_pct}%",
        )

    with a2:

        st.metric(
            "Anomalies",
            f"{anomaly_count:,}",
        )

    with a3:

        st.metric(
            "Normal Records",
            f"{normal_count:,}",
        )

    st.info(
        "Isolation Forest identifies unusual records "
        "based on numeric feature patterns. An anomaly "
        "is not automatically a data error and should "
        "be validated using business context."
    )

    anomaly_preview = anomaly_export[
        anomaly_export["Anomaly"]
        == "Anomaly"
    ].copy()

    if not anomaly_preview.empty:

        st.subheader(
            "Detected Anomalies"
        )

        st.dataframe(
            anomaly_preview.head(50),
            use_container_width=True,
            hide_index=True,
        )

else:

    anomaly_count = 0

    st.info(
        "Not enough numeric data for Isolation Forest."
    )


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

st.header("🧠 Root-Cause Analysis")

root_cause = build_root_cause_analysis(
    df=df,
    missing_count=missing_count,
    duplicate_count=duplicate_count,
    invalid_count=invalid_count,
    iqr_results=iqr_results,
    anomaly_count=anomaly_count,
)

st.dataframe(
    root_cause,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# GEMINI ANALYSIS
# ============================================================

st.header("✨ AI Data Quality Analysis")

summary_text = f"""
Dataset rows: {len(df)}
Dataset columns: {len(df.columns)}

Missing cells: {missing_count}
Duplicate rows: {duplicate_count}
Invalid values: {invalid_count}

Quality score: {quality_score}/100

IQR outlier count:
{
    int(iqr_results["Outliers"].sum())
    if not iqr_results.empty
    else 0
}

Isolation Forest anomalies:
{anomaly_count}

Numeric columns:
{numeric_columns}

Categorical columns:
{categorical_columns}

Date columns:
{datetime_columns}
"""


if ai_enabled:

    with st.spinner(
        "Generating AI analysis..."
    ):

        ai_analysis = get_gemini_analysis(
            summary_text
        )

    st.markdown(
        ai_analysis
    )

else:

    st.info(
        "Gemini AI analysis is disabled."
    )


# ============================================================
# AUTOMATED CLEANING
# ============================================================

st.header("🧹 Automated Data Cleaning")

with st.spinner(
    "Cleaning dataset..."
):

    (
        cleaned_df,
        cleaning_log,
        cleaning_summary,
    ) = clean_dataset(df)


c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Original Rows",
        f"{cleaning_summary['Original Rows']:,}",
    )

with c2:

    st.metric(
        "Cleaned Rows",
        f"{cleaning_summary['Cleaned Rows']:,}",
    )

with c3:

    st.metric(
        "Rows Removed",
        f"{cleaning_summary['Rows Removed']:,}",
    )

with c4:

    st.metric(
        "Values Filled",
        f"{cleaning_summary['Values Filled']:,}",
    )


if not cleaning_log.empty:

    st.subheader(
        "Cleaning Actions"
    )

    st.dataframe(
        cleaning_log,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No cleaning actions were required."
    )


st.subheader(
    "Cleaned Dataset Preview"
)

st.dataframe(
    cleaned_df.head(20),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# POWER BI EXPORT
# ============================================================

st.header("1️⃣1️⃣ Power BI Export")

quality_summary = pd.DataFrame(
    {
        "Metric": [
            "Original Rows",
            "Original Columns",
            "Missing Cells",
            "Duplicate Rows",
            "Invalid Values",
            "Quality Score",
            "IQR Outliers",
            "ML Anomalies",
            "Cleaned Rows",
            "Rows Removed",
            "Values Filled",
        ],
        "Value": [
            len(df),
            len(df.columns),
            missing_count,
            duplicate_count,
            invalid_count,
            quality_score,
            (
                int(
                    iqr_results[
                        "Outliers"
                    ].sum()
                )
                if not iqr_results.empty
                else 0
            ),
            anomaly_count,
            cleaning_summary[
                "Cleaned Rows"
            ],
            cleaning_summary[
                "Rows Removed"
            ],
            cleaning_summary[
                "Values Filled"
            ],
        ],
    }
)


quality_issues_list = []

for _, row in missing_table.iterrows():

    if row["Missing Values"] > 0:

        quality_issues_list.append(
            {
                "Column": row["Column"],
                "Issue": "Missing Values",
                "Count": row["Missing Values"],
            }
        )


if not invalid_results.empty:

    for _, row in invalid_results.iterrows():

        quality_issues_list.append(
            {
                "Column": row["Column"],
                "Issue": row["Issue"],
                "Count": row["Count"],
            }
        )


if duplicate_count > 0:

    quality_issues_list.append(
        {
            "Column": "All Columns",
            "Issue": "Duplicate Rows",
            "Count": duplicate_count,
        }
    )


if not iqr_results.empty:

    for _, row in iqr_results.iterrows():

        quality_issues_list.append(
            {
                "Column": row["Column"],
                "Issue": "IQR Outliers",
                "Count": row["Outliers"],
            }
        )


quality_issues = pd.DataFrame(
    quality_issues_list,
    columns=[
        "Column",
        "Issue",
        "Count",
    ],
)


(
    export_files,
    zip_bytes,
) = create_powerbi_exports(
    quality_summary,
    quality_issues,
    anomaly_export,
    cleaning_log,
    cleaned_df,
)


st.write(
    "DataGuard AI creates Power BI-ready CSV files "
    "with stable filenames for easy import into "
    "Power BI Desktop."
)

st.info(
    "Download the ZIP file and import the CSV files "
    "into Power BI Desktop. No Azure or Power BI API "
    "configuration is required."
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
    "Exported Files"
)


export_file_info = pd.DataFrame(
    {
        "File": [
            "quality_summary.csv",
            "quality_issues.csv",
            "anomaly_results.csv",
            "cleaning_log.csv",
            "cleaned_data.csv",
        ],
        "Purpose": [
            "Overall dataset quality metrics",
            "Detected data-quality issues",
            "ML anomaly results",
            "Detailed cleaning actions",
            "Cleaned dataset for analysis",
        ],
    }
)


st.dataframe(
    export_file_info,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# INDIVIDUAL DOWNLOADS
# ============================================================

st.subheader(
    "📥 Individual CSV Downloads"
)


download_columns = st.columns(5)

for index, (
    filename,
    data,
) in enumerate(
    export_files.items()
):

    with download_columns[
        index % 5
    ]:

        st.download_button(
            label=filename,
            data=data,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# FINAL REPORT
# ============================================================

st.header("📄 Final Data Quality Report")


report_text = f"""
DataGuard AI - Data Quality Report

Generated:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

--------------------------------------------------
DATASET
--------------------------------------------------

Rows: {len(df):,}
Columns: {len(df.columns):,}

--------------------------------------------------
DATA QUALITY
--------------------------------------------------

Quality Score: {quality_score}/100
Missing Cells: {missing_count:,}
Duplicate Rows: {duplicate_count:,}
Invalid Values: {invalid_count:,}

--------------------------------------------------
OUTLIER DETECTION
--------------------------------------------------

IQR Outliers:
{
    int(iqr_results["Outliers"].sum())
    if not iqr_results.empty
    else 0
}

Isolation Forest Anomalies:
{anomaly_count:,}

Isolation Forest Sensitivity:
{contamination_pct}%

--------------------------------------------------
CLEANING
--------------------------------------------------

Original Rows:
{cleaning_summary["Original Rows"]:,}

Cleaned Rows:
{cleaning_summary["Cleaned Rows"]:,}

Rows Removed:
{cleaning_summary["Rows Removed"]:,}

Values Filled:
{cleaning_summary["Values Filled"]:,}

--------------------------------------------------
POWER BI
--------------------------------------------------

Power BI-ready CSV exports were generated.

No Azure connection or Power BI API configuration
is required.

Import the exported CSV files into Power BI Desktop.

--------------------------------------------------
END OF REPORT
--------------------------------------------------
"""


st.text_area(
    "Report",
    report_text,
    height=400,
)


st.download_button(
    "📄 Download Final Report",
    data=report_text,
    file_name="DataGuard_AI_Final_Report.txt",
    mime="text/plain",
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ DataGuard AI | AI-powered Data Quality "
    "and Anomaly Detection"
)

st.caption(
    "Built with Python • Pandas • Scikit-learn • "
    "Streamlit • Gemini AI • Power BI"
)
