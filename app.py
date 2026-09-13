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
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Header */
    .main-header {
        padding: 10px 0 5px 0;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    /* KPI cards */
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        min-height: 125px;
        box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
    }

    .metric-label {
        color: #667085;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #98a2b3;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Section cards */
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e4e7ec;
        margin-bottom: 15px;
    }

    /* Quality score */
    .quality-box {
        background: white;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e4e7ec;
        text-align: center;
    }

    .quality-score {
        font-size: 55px;
        font-weight: 800;
    }

    .quality-label {
        color: #667085;
        font-size: 15px;
    }

    /* Info banner */
    .info-box {
        background: #eef4ff;
        border: 1px solid #c7d7fe;
        padding: 14px 18px;
        border-radius: 10px;
        color: #344054;
        margin-bottom: 15px;
    }

    /* Warning */
    .warning-box {
        background: #fffaeb;
        border: 1px solid #fedf89;
        padding: 14px 18px;
        border-radius: 10px;
        color: #7a2e0b;
        margin-bottom: 15px;
    }

    /* Success */
    .success-box {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        padding: 14px 18px;
        border-radius: 10px;
        color: #05603a;
        margin-bottom: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e4e7ec;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ DataGuard AI</div>
    <div class="subtitle">
        AI-Powered Data Quality, Anomaly Detection & Business Intelligence
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_to_datetime(series):
    """
    Safely convert a pandas Series to datetime.
    Prevents deployment failures caused by mixed date formats.
    """

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
    """
    Detect genuine datetime columns.

    Important:
    Numeric columns such as Customer_ID must NOT
    automatically become datetime columns.
    """

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
        "day"
    )

    for column in df.columns:

        series = df[column]

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)
            continue

        # Never automatically treat numeric IDs as dates
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Only inspect text/object columns
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

        column_name = str(column).lower()

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
            .str.contains(r"[-/:]", regex=True)
            .mean()
            >= 0.50
        )

        if (
            any(keyword in column_name for keyword in date_keywords)
            or looks_date_like
        ):
            datetime_columns.append(column)

    return datetime_columns


def load_uploaded_file(uploaded_file):

    if uploaded_file is None:
        return None

    file_name = uploaded_file.name.lower()

    try:

        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)

        elif file_name.endswith(".xlsx"):
            return pd.read_excel(uploaded_file)

        elif file_name.endswith(".xls"):
            return pd.read_excel(uploaded_file)

        else:
            st.error(
                "Unsupported file format. Please upload CSV or Excel."
            )
            return None

    except Exception as e:

        st.error(
            f"Unable to read the file: {e}"
        )

        return None


def calculate_missing_summary(df):

    result = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2)
    })

    return result.sort_values(
        "Missing Count",
        ascending=False
    )


def calculate_numeric_summary(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    return (
        numeric_df
        .describe()
        .T
        .reset_index()
        .rename(columns={"index": "Column"})
    )


def detect_invalid_values(df):

    results = []

    invalid_masks = []

    for column in df.columns:

        column_name = str(column).lower()

        # Age validation
        if column_name == "age" or column_name.endswith("_age"):

            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            mask = (
                numeric.notna()
                & (
                    (numeric < 0)
                    | (numeric > 100)
                )
            )

            count = int(mask.sum())

            if count > 0:

                results.append({
                    "Column": column,
                    "Issue": "Invalid Age",
                    "Count": count
                })

                invalid_masks.append(mask)

        # Quantity validation
        if (
            column_name == "quantity"
            or column_name.endswith("_quantity")
        ):

            numeric = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            mask = (
                numeric.notna()
                & (numeric <= 0)
            )

            count = int(mask.sum())

            if count > 0:

                results.append({
                    "Column": column,
                    "Issue": "Invalid Quantity",
                    "Count": count
                })

                invalid_masks.append(mask)

    if results:

        result_df = pd.DataFrame(results)

    else:

        result_df = pd.DataFrame(
            columns=[
                "Column",
                "Issue",
                "Count"
            ]
        )

    if invalid_masks:

        combined_mask = pd.concat(
            invalid_masks,
            axis=1
        ).any(axis=1)

        invalid_row_count = int(
            combined_mask.sum()
        )

    else:

        invalid_row_count = 0

    return result_df, invalid_row_count


def check_city_inconsistencies(df):

    if "City" not in df.columns:

        return {
            "exists": False,
            "count": 0,
            "rows": 0,
            "values": []
        }

    city_series = (
        df["City"]
        .astype("string")
        .str.strip()
    )

    problematic_mask = city_series.isin(
        CITY_MAPPING.keys()
    )

    values = sorted(
        city_series[
            problematic_mask
        ]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "exists": True,
        "count": len(values),
        "rows": int(problematic_mask.sum()),
        "values": values
    }


def detect_iqr_outliers(df):

    outlier_masks = {}
    summary = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 5:
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
                errors="coerce"
            ).lt(lower)
            |
            pd.to_numeric(
                df[column],
                errors="coerce"
            ).gt(upper)
        )

        count = int(mask.sum())

        outlier_masks[column] = mask

        summary.append({
            "Column": column,
            "Lower Bound": round(lower, 2),
            "Upper Bound": round(upper, 2),
            "Outlier Count": count
        })

    summary_df = pd.DataFrame(summary)

    if outlier_masks:

        combined = pd.concat(
            [
                mask.rename(column)
                for column, mask in outlier_masks.items()
            ],
            axis=1
        )

        combined_outlier_mask = combined.any(axis=1)

        outlier_row_count = int(
            combined_outlier_mask.sum()
        )

    else:

        outlier_row_count = 0

    return summary_df, outlier_row_count


def run_ml_anomaly_detection(
    df,
    contamination=0.05
):

    numeric_df = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric_df.shape[1] == 0:

        return None, 0

    # Remove columns that look like IDs
    id_columns = []

    for column in numeric_df.columns:

        name = str(column).lower()

        if (
            name == "id"
            or name.endswith("_id")
            or name.startswith("id_")
            or "customer_id" in name
            or "order_id" in name
        ):
            id_columns.append(column)

    numeric_df = numeric_df.drop(
        columns=id_columns,
        errors="ignore"
    )

    if numeric_df.shape[1] == 0:

        return None, 0

    # Convert values
    numeric_df = numeric_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Fill missing values using median
    numeric_df = numeric_df.fillna(
        numeric_df.median()
    )

    # Remove completely empty columns
    numeric_df = numeric_df.dropna(
        axis=1,
        how="all"
    )

    if numeric_df.shape[1] == 0:
        return None, 0

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )

    try:

        predictions = model.fit_predict(
            numeric_df
        )

    except Exception:
        return None, 0

    result = df.copy()

    result["ML_Anomaly"] = (
        predictions == -1
    )

    anomaly_count = int(
        result["ML_Anomaly"].sum()
    )

    return result, anomaly_count


def calculate_quality_score(
    df,
    invalid_count=0,
    city_issue_rows=0,
    outlier_count=0
):

    if df.empty:
        return 0

    rows = max(len(df), 1)

    total_cells = max(
        df.shape[0] * df.shape[1],
        1
    )

    missing_rate = (
        df.isna().sum().sum()
        / total_cells
    )

    duplicate_rate = (
        df.duplicated().sum()
        / rows
    )

    invalid_rate = min(
        invalid_count / rows,
        1
    )

    city_rate = min(
        city_issue_rows / rows,
        1
    )

    outlier_rate = min(
        outlier_count / rows,
        1
    )

    score = (
        25 * (1 - missing_rate)
        +
        20 * (1 - duplicate_rate)
        +
        25 * (1 - invalid_rate)
        +
        15 * (1 - city_rate)
        +
        15 * (1 - outlier_rate)
    )

    return round(
        max(0, min(100, score)),
        1
    )


def quality_label(score):

    if score >= 90:
        return "Excellent"

    elif score >= 75:
        return "Good"

    elif score >= 60:
        return "Needs Improvement"

    else:
        return "Poor"


def clean_data(df):

    cleaned = df.copy()

    # --------------------------------------------------------
    # Standardize city
    # --------------------------------------------------------

    if "City" in cleaned.columns:

        cleaned["City"] = (
            cleaned["City"]
            .astype("string")
            .str.strip()
            .replace(CITY_MAPPING)
        )

    # --------------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------------

    for column in cleaned.columns:

        column_name = str(column).lower()

        if (
            column_name == "age"
            or column_name.endswith("_age")
        ):

            numeric = pd.to_numeric(
                cleaned[column],
                errors="coerce"
            ).astype("float64")

            numeric.loc[
                (numeric < 0)
                |
                (numeric > 100)
            ] = np.nan

            cleaned[column] = numeric

            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )

        elif (
            column_name == "quantity"
            or column_name.endswith("_quantity")
        ):

            numeric = pd.to_numeric(
                cleaned[column],
                errors="coerce"
            ).astype("float64")

            numeric.loc[
                numeric <= 0
            ] = np.nan

            cleaned[column] = numeric

            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )

    # --------------------------------------------------------
    # General missing value handling
    # --------------------------------------------------------

    for column in cleaned.columns:

        if cleaned[column].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(
            cleaned[column]
        ):

            cleaned[column] = cleaned[column].fillna(
                cleaned[column].median()
            )

        else:

            mode = cleaned[column].mode(
                dropna=True
            )

            if not mode.empty:

                cleaned[column] = cleaned[column].fillna(
                    mode.iloc[0]
                )

            else:

                cleaned[column] = cleaned[column].fillna(
                    "Unknown"
                )

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    cleaned = cleaned.drop_duplicates()

    return cleaned


def create_data_dictionary(df):

    rows = []

    for column in df.columns:

        rows.append({
            "Column": column,
            "Data Type": str(df[column].dtype),
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
            )
        })

    return pd.DataFrame(rows)


def create_quality_summary(
    df,
    quality_score,
    invalid_count,
    duplicate_count,
    city_issue_rows,
    outlier_count,
    anomaly_count
):

    rows = len(df)

    return pd.DataFrame({
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
            "Quality Score"
        ],
        "Value": [
            rows,
            df.shape[1],
            int(df.isna().sum().sum()),
            duplicate_count,
            invalid_count,
            city_issue_rows,
            outlier_count,
            anomaly_count,
            round(
                anomaly_count / max(rows, 1) * 100,
                2
            ),
            quality_score
        ]
    })


def generate_zip(files):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for filename, data in files.items():

            z.writestr(
                filename,
                data
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ DataGuard AI")

    st.caption(
        "Data quality monitoring and anomaly detection"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    st.divider()

    st.markdown("### 🚨 ML Settings")

    anomaly_contamination = st.slider(
        "Anomaly sensitivity",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        format="%.0f%%",
        help=(
            "Expected fraction of records to flag for review. "
            "This is a screening setting, not proof that records are erroneous."
        )
    )

    st.divider()

    st.markdown("### 📌 Pipeline")

    st.caption(
        "1. Upload\n"
        "2. Profile\n"
        "3. Detect issues\n"
        "4. Analyze anomalies\n"
        "5. Clean data\n"
        "6. Export for Power BI"
    )

    st.divider()

    st.caption(
        "Built with Python • Pandas • Scikit-learn • Streamlit"
    )


# ============================================================
# LANDING PAGE
# ============================================================

if uploaded_file is None:

    st.markdown("""
    <div class="info-box">
        <b>Welcome to DataGuard AI</b><br>
        Upload a CSV or Excel dataset from the sidebar to automatically
        profile data quality, detect anomalies, identify inconsistencies,
        clean the dataset and prepare Power BI-ready exports.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="section-card">
        <h3>🔍 Detect</h3>
        <p>
        Identify missing values, duplicates, invalid values,
        inconsistencies and statistical outliers.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="section-card">
        <h3>🚨 Analyze</h3>
        <p>
        Use Isolation Forest to flag unusual records
        for further investigation.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="section-card">
        <h3>📦 Export</h3>
        <p>
        Clean your data and download Power BI-ready
        CSV reports and a ZIP package.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = load_uploaded_file(
    uploaded_file
)

if df is None:
    st.stop()


# ============================================================
# BASIC METRICS
# ============================================================

rows = len(df)
columns = df.shape[1]

missing_cells = int(
    df.isna().sum().sum()
)

duplicate_count = int(
    df.duplicated().sum()
)

numeric_columns = len(
    df.select_dtypes(
        include=np.number
    ).columns
)

categorical_columns = (
    columns
    -
    numeric_columns
)

datetime_columns = detect_datetime_columns(
    df
)

invalid_df, invalid_count = detect_invalid_values(
    df
)

city_info = check_city_inconsistencies(
    df
)

outlier_summary, outlier_count = detect_iqr_outliers(
    df
)

ml_df, anomaly_count = run_ml_anomaly_detection(
    df,
    contamination=anomaly_contamination
)

quality_score = calculate_quality_score(
    df,
    invalid_count=invalid_count,
    city_issue_rows=city_info["rows"],
    outlier_count=outlier_count
)

quality_status = quality_label(
    quality_score
)

anomaly_rate = (
    anomaly_count / max(rows, 1) * 100
)


# ============================================================
# KPI HEADER
# ============================================================

st.markdown("### 📊 Dataset Overview")

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rows</div>
        <div class="metric-value">{rows:,}</div>
        <div class="metric-description">Records</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Columns</div>
        <div class="metric-value">{columns}</div>
        <div class="metric-description">Features</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Missing</div>
        <div class="metric-value">{missing_cells:,}</div>
        <div class="metric-description">Missing cells</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Duplicates</div>
        <div class="metric-value">{duplicate_count:,}</div>
        <div class="metric-description">Duplicate rows</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Quality</div>
        <div class="metric-value">{quality_score}</div>
        <div class="metric-description">{quality_status}</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ML Flags</div>
        <div class="metric-value">{anomaly_rate:.1f}%</div>
        <div class="metric-description">Screening rate</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")


# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "📊 Overview",
    "🔍 Data Quality",
    "🚨 Anomalies",
    "📈 Business Insights",
    "🧹 Data Cleaning",
    "🤖 AI Analysis",
    "📦 Power BI Export",
    "📄 Final Report"
])


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.markdown("### Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True,
        height=420
    )

    st.markdown("### Dataset Structure")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Numeric Columns",
        numeric_columns
    )

    c2.metric(
        "Categorical Columns",
        categorical_columns
    )

    c3.metric(
        "Datetime Columns",
        len(datetime_columns)
    )

    c4.metric(
        "Invalid Values",
        invalid_count
    )

    st.markdown("### Detected Datetime Columns")

    if datetime_columns:

        st.success(
            ", ".join(
                datetime_columns
            )
        )

    else:

        st.info(
            "No reliable datetime columns detected."
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[1]:

    st.markdown("### 🔍 Data Quality Assessment")

    q1, q2 = st.columns([1, 2])

    with q1:

        st.markdown(
            f"""
            <div class="quality-box">
                <div class="quality-label">
                    Rule-Based Data Quality Score
                </div>
                <div class="quality-score">
                    {quality_score}/100
                </div>
                <div class="quality-label">
                    {quality_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q2:

        missing_rate = (
            missing_cells
            /
            max(rows * columns, 1)
            * 100
        )

        duplicate_rate = (
            duplicate_count
            /
            max(rows, 1)
            * 100
        )

        invalid_rate = (
            invalid_count
            /
            max(rows, 1)
            * 100
        )

        city_rate = (
            city_info["rows"]
            /
            max(rows, 1)
            * 100
        )

        outlier_rate = (
            outlier_count
            /
            max(rows, 1)
            * 100
        )

        quality_breakdown = pd.DataFrame({
            "Metric": [
                "Missing Rate",
                "Duplicate Rate",
                "Invalid Rate",
                "City Inconsistency Rate",
                "IQR Outlier Rate"
            ],
            "Rate %": [
                missing_rate,
                duplicate_rate,
                invalid_rate,
                city_rate,
                outlier_rate
            ]
        }).set_index("Metric")

        st.bar_chart(
            quality_breakdown
        )

    st.markdown("### Missing Values")

    missing_summary = calculate_missing_summary(
        df
    )

    st.dataframe(
        missing_summary,
        use_container_width=True
    )

    st.markdown("### Invalid Values")

    if invalid_df.empty:

        st.success(
            "No rule-based invalid Age or Quantity values detected."
        )

    else:

        st.dataframe(
            invalid_df,
            use_container_width=True
        )

    st.markdown("### Duplicate Rows")

    if duplicate_count > 0:

        st.warning(
            f"{duplicate_count:,} duplicate rows detected."
        )

    else:

        st.success(
            "No duplicate rows detected."
        )

    st.markdown("### City Consistency")

    if city_info["exists"]:

        if city_info["count"] > 0:

            st.warning(
                f"Detected {city_info['count']} inconsistent city labels "
                f"affecting {city_info['rows']:,} rows."
            )

            st.write(
                "Detected values:",
                ", ".join(city_info["values"])
            )

        else:

            st.success(
                "No mapped city inconsistencies detected."
            )

    else:

        st.info(
            "City column not found."
        )

    st.markdown("### IQR Outlier Analysis")

    if outlier_summary.empty:

        st.info(
            "No numeric columns were suitable for IQR analysis."
        )

    else:

        st.dataframe(
            outlier_summary,
            use_container_width=True
        )

        st.caption(
            f"{outlier_count:,} rows contain at least one IQR-based outlier."
        )


# ============================================================
# ANOMALIES
# ============================================================

with tabs[2]:

    st.markdown("### 🚨 Machine Learning Anomaly Detection")

    st.markdown(
        f"""
        <div class="info-box">
        <b>Current sensitivity: {anomaly_contamination:.0%}</b><br>
        Isolation Forest is being used as a screening method to identify
        unusual records. A flagged record is not automatically considered
        incorrect.
        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Records",
        f"{rows:,}"
    )

    a2.metric(
        "ML Flags",
        f"{anomaly_count:,}"
    )

    a3.metric(
        "Flag Rate",
        f"{anomaly_rate:.2f}%"
    )

    if ml_df is not None:

        st.markdown("### Flagged Records")

        flagged = ml_df[
            ml_df["ML_Anomaly"]
        ]

        st.dataframe(
            flagged.head(100),
            use_container_width=True,
            height=400
        )

        st.caption(
            "Showing the first 100 flagged records."
        )

        st.download_button(
            "⬇️ Download ML Anomaly Results",
            data=ml_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="ml_anomaly_results.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "ML anomaly detection could not be performed because "
            "no suitable numeric features were available."
        )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

with tabs[3]:

    st.markdown("### 📈 Business Insights")

    # --------------------------------------------------------
    # Sales analysis
    # --------------------------------------------------------

    if "Sales" in df.columns:

        sales = pd.to_numeric(
            df["Sales"],
            errors="coerce"
        ).dropna()

        if not sales.empty:

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                "Total Sales",
                f"{sales.sum():,.2f}"
            )

            s2.metric(
                "Average Sale",
                f"{sales.mean():,.2f}"
            )

            s3.metric(
                "Median Sale",
                f"{sales.median():,.2f}"
            )

            s4.metric(
                "Maximum Sale",
                f"{sales.max():,.2f}"
            )

            st.markdown("### Sales Distribution")

            hist_counts, bin_edges = np.histogram(
                sales,
                bins=20
            )

            histogram_df = pd.DataFrame({
                "Sales Range": [
                    f"{bin_edges[i]:,.0f} – {bin_edges[i+1]:,.0f}"
                    for i in range(
                        len(bin_edges) - 1
                    )
                ],
                "Transactions": hist_counts
            }).set_index(
                "Sales Range"
            )

            st.bar_chart(
                histogram_df
            )

            # Category analysis
            if "Category" in df.columns:

                category_sales = (
                    df.assign(
                        Sales=pd.to_numeric(
                            df["Sales"],
                            errors="coerce"
                        )
                    )
                    .groupby("Category")[
                        "Sales"
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                st.markdown("### Sales by Category")

                st.bar_chart(
                    category_sales
                )

    else:

        st.info(
            "A Sales column was not detected."
        )

    # --------------------------------------------------------
    # Datetime trend
    # --------------------------------------------------------

    if datetime_columns:

        st.markdown("### 📅 Trend Analysis")

        selected_date = st.selectbox(
            "Select date column",
            datetime_columns
        )

        date_series = safe_to_datetime(
            df[selected_date]
        )

        trend_df = df.copy()

        trend_df["_Date"] = date_series

        if "Sales" in trend_df.columns:

            trend_df["Sales"] = pd.to_numeric(
                trend_df["Sales"],
                errors="coerce"
            )

            trend = (
                trend_df
                .dropna(subset=["_Date"])
                .groupby(
                    trend_df["_Date"]
                    .dt.date
                )["Sales"]
                .sum()
            )

            if not trend.empty:

                st.line_chart(
                    trend
                )

        else:

            trend = (
                trend_df
                .dropna(subset=["_Date"])
                .groupby(
                    trend_df["_Date"]
                    .dt.date
                )
                .size()
            )

            if not trend.empty:

                st.line_chart(
                    trend
                )

    else:

        st.info(
            "No reliable date column was detected, so trend analysis is unavailable."
        )


# ============================================================
# DATA CLEANING
# ============================================================

with tabs[4]:

    st.markdown("### 🧹 Automated Data Cleaning")

    cleaned_df = clean_data(
        df
    )

    before_rows = len(df)
    after_rows = len(cleaned_df)

    before_missing = int(
        df.isna().sum().sum()
    )

    after_missing = int(
        cleaned_df.isna().sum().sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows Before",
        f"{before_rows:,}"
    )

    c2.metric(
        "Rows After",
        f"{after_rows:,}"
    )

    c3.metric(
        "Missing Before",
        f"{before_missing:,}"
    )

    c4.metric(
        "Missing After",
        f"{after_missing:,}"
    )

    st.markdown("### Cleaning Operations")

    operations = [
        "Standardized mapped city names",
        "Handled invalid Age values",
        "Handled invalid Quantity values",
        "Filled missing numeric values using median",
        "Filled missing categorical values using mode",
        "Removed exact duplicate rows"
    ]

    for operation in operations:

        st.write(
            f"✓ {operation}"
        )

    st.markdown("### Cleaned Dataset Preview")

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True,
        height=400
    )

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=cleaned_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )


# ============================================================
# AI ANALYSIS
# ============================================================

with tabs[5]:

    st.markdown("### 🤖 AI Data Quality Assessment")

    st.markdown("""
    <div class="info-box">
    AI analysis is evidence-based: the model is instructed to use only
    metrics calculated by DataGuard AI and to label possible causes as
    hypotheses rather than facts.
    </div>
    """, unsafe_allow_html=True)

    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Optional. Your API key is used only for this analysis."
    )

    if st.button(
        "🤖 Run AI Analysis",
        use_container_width=True
    ):

        if not gemini_api_key:

            st.warning(
                "Enter a Gemini API key to run AI analysis."
            )

        else:

            try:

                import google.generativeai as genai

                genai.configure(
                    api_key=gemini_api_key
                )

                model = genai.GenerativeModel(
                    "gemini-1.5-flash"
                )

                sales_summary = {}

                if "Sales" in df.columns:

                    sales_numeric = pd.to_numeric(
                        df["Sales"],
                        errors="coerce"
                    ).dropna()

                    if not sales_numeric.empty:

                        sales_summary = {
                            "total_sales": round(
                                float(
                                    sales_numeric.sum()
                                ),
                                2
                            ),
                            "average_sales": round(
                                float(
                                    sales_numeric.mean()
                                ),
                                2
                            ),
                            "median_sales": round(
                                float(
                                    sales_numeric.median()
                                ),
                                2
                            ),
                            "maximum_sales": round(
                                float(
                                    sales_numeric.max()
                                ),
                                2
                            )
                        }

                prompt = f"""
You are a data quality analyst reviewing a dataset.

Use ONLY the facts and metrics supplied below.

Do not invent relationships.
Do not invent causes.
Do not claim a relationship unless the supplied metrics demonstrate it.
If you suggest a possible cause, explicitly label it as a hypothesis.

Important distinction:
- Missing values, duplicates, invalid values and mapped inconsistencies
  are data-quality issues.
- IQR outliers are statistical observations.
- Isolation Forest anomalies are ML screening flags and are NOT proof
  that records are incorrect.

Dataset facts:

Rows: {rows}
Columns: {columns}

Missing cells: {missing_cells}
Duplicate rows: {duplicate_count}
Invalid values: {invalid_count}

City inconsistency rows: {city_info["rows"]}
City inconsistency labels: {city_info["values"]}

IQR outlier rows: {outlier_count}

Isolation Forest anomaly rows: {anomaly_count}
Isolation Forest anomaly rate: {anomaly_rate:.2f}%

Rule-based quality score: {quality_score}/100

Sales summary:
{sales_summary}

Provide the following:

1. Executive summary
2. Confirmed data-quality issues
3. Statistical observations
4. ML anomaly interpretation
5. Possible causes — clearly label these as hypotheses
6. Recommended data-cleaning actions
7. Business impact
8. Final priority level: Low, Medium, High or Critical

Do not say the dataset is unusable solely because of ML anomaly flags.
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
# POWER BI EXPORT
# ============================================================

with tabs[6]:

    st.markdown("### 📦 Power BI-Ready Export")

    cleaned_df = clean_data(
        df
    )

    data_dictionary = create_data_dictionary(
        cleaned_df
    )

    quality_summary = create_quality_summary(
        df,
        quality_score,
        invalid_count,
        duplicate_count,
        city_info["rows"],
        outlier_count,
        anomaly_count
    )

    numeric_summary = calculate_numeric_summary(
        cleaned_df
    )

    missing_summary = calculate_missing_summary(
        cleaned_df
    )

    files = {
        "cleaned_data.csv":
            cleaned_df.to_csv(
                index=False
            ),

        "data_dictionary.csv":
            data_dictionary.to_csv(
                index=False
            ),

        "quality_summary.csv":
            quality_summary.to_csv(
                index=False
            ),

        "numeric_summary.csv":
            numeric_summary.to_csv(
                index=False
            ),

        "missing_summary.csv":
            missing_summary.to_csv(
                index=False
            )
    }

    st.markdown("""
    <div class="success-box">
    <b>✓ Power BI package ready</b><br>
    Import the cleaned CSV into Power BI and use the supporting
    summary files for data-quality reporting.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Available Files")

    for filename in files.keys():

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
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Quality Summary",
        data=files[
            "quality_summary.csv"
        ].encode("utf-8"),
        file_name="quality_summary.csv",
        mime="text/csv",
        use_container_width=True
    )

    zip_data = generate_zip(
        files
    )

    st.download_button(
        "📦 Download Complete Power BI Package",
        data=zip_data,
        file_name="dataguard_powerbi_package.zip",
        mime="application/zip",
        use_container_width=True
    )


# ============================================================
# FINAL REPORT
# ============================================================

with tabs[7]:

    st.markdown("### 📄 DataGuard AI Final Report")

    report_data = {
        "Generated At": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Dataset": uploaded_file.name,
        "Rows": rows,
        "Columns": columns,
        "Missing Cells": missing_cells,
        "Duplicate Rows": duplicate_count,
        "Invalid Values": invalid_count,
        "City Issue Rows": city_info["rows"],
        "IQR Outlier Rows": outlier_count,
        "ML Anomaly Rows": anomaly_count,
        "ML Anomaly Rate": f"{anomaly_rate:.2f}%",
        "Quality Score": quality_score,
        "Quality Status": quality_status,
        "Rows After Cleaning": len(
            clean_data(df)
        )
    }

    report_df = pd.DataFrame({
        "Metric": report_data.keys(),
        "Value": report_data.values()
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

    report_text = f"""
DataGuard AI - Data Quality Report
===================================

Dataset:
{uploaded_file.name}

Generated:
{report_data["Generated At"]}

DATASET
-------
Rows: {rows}
Columns: {columns}

DATA QUALITY
------------
Missing Cells: {missing_cells}
Duplicate Rows: {duplicate_count}
Invalid Values: {invalid_count}
City Issue Rows: {city_info["rows"]}
IQR Outlier Rows: {outlier_count}

MACHINE LEARNING
----------------
Isolation Forest Sensitivity: {anomaly_contamination:.0%}
ML Anomaly Rows: {anomaly_count}
ML Anomaly Rate: {anomaly_rate:.2f}%

QUALITY SCORE
-------------
Score: {quality_score}/100
Status: {quality_status}

CLEANING
--------
Rows Before: {rows}
Rows After: {len(clean_data(df))}
Missing Values After Cleaning:
{int(clean_data(df).isna().sum().sum())}

POWER BI EXPORTS
----------------
cleaned_data.csv
data_dictionary.csv
quality_summary.csv
numeric_summary.csv
missing_summary.csv

Note:
ML anomaly flags are screening indicators and should be reviewed
before being treated as confirmed data-quality errors.
"""

    st.download_button(
        "⬇️ Download Final Report",
        data=report_text.encode("utf-8"),
        file_name="dataguard_final_report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.markdown("""
    <div class="success-box">
    <b>✓ DataGuard AI pipeline completed successfully.</b><br>
    Your dataset has been profiled, checked, analyzed, cleaned and
    prepared for downstream BI analysis.
    </div>
    """, unsafe_allow_html=True)
