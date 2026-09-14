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
# SESSION STATE
# ============================================================

defaults = {
    "page": "Dashboard",
    "df": None,
    "cleaned_df": None,
    "dataset_name": "sales_data_raw.csv",
    "sensitivity": 5,
    "gemini_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"] {
    background: #0b1020;
    border-right: 1px solid rgba(255,255,255,0.10);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* DataGuard AI title */
[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 24px !important;
    font-weight: 800 !important;
}

/* Sidebar captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}

/* Section headings */
[data-testid="stSidebar"] p strong,
[data-testid="stSidebar"] strong {
    color: #94a3b8 !important;
    font-size: 12px !important;
    letter-spacing: 1px;
}

/* Navigation buttons */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid transparent !important;
    color: #cbd5e1 !important;
    text-align: left !important;
    border-radius: 10px !important;
    margin: 3px 0 !important;
    min-height: 42px !important;
}

/* Button text */
[data-testid="stSidebar"] button p {
    color: #cbd5e1 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Hover */
[data-testid="stSidebar"] button:hover {
    background: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.10) !important;
}

/* Primary / active button */
[data-testid="stSidebar"] button[kind="primary"] {
    background: rgba(99,102,241,0.18) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
}

/* Primary button text */
[data-testid="stSidebar"] button[kind="primary"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
}

/* Success system status */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(34,197,94,0.10) !important;
    border: 1px solid rgba(34,197,94,0.25) !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #86efac !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read CSV/XLSX/XLS into a DataFrame."""

    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type.")


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


def datetime_columns(df):
    return df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()


def invalid_numeric_values(df):
    """
    Detect invalid values only in columns that appear
    to be numeric-like.

    Normal categorical columns such as City, Category,
    Region etc. are NOT treated as invalid.
    """

    invalid = 0

    for column in df.columns:

        series = df[column]

        # Already numeric
        if pd.api.types.is_numeric_dtype(series):
            continue

        # Already datetime
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

        # Column appears to be numeric-like
        if numeric_ratio >= 0.80:
            invalid += int(converted.isna().sum())

    return invalid


def quality_metrics(df):

    rows = len(df)
    columns = len(df.columns)

    if rows == 0 or columns == 0:
        return {
            "score": 0,
            "missing": 0,
            "duplicates": 0,
            "invalid": 0,
        }

    missing = int(df.isna().sum().sum())

    duplicates = int(df.duplicated().sum())

    invalid = int(invalid_numeric_values(df))

    total_cells = rows * columns

    missing_rate = missing / total_cells
    duplicate_rate = duplicates / rows
    invalid_rate = invalid / rows

    score = (
        100
        - missing_rate * 50
        - duplicate_rate * 30
        - invalid_rate * 20
    )

    score = max(0, min(100, score))

    return {
        "score": round(score, 2),
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def iqr_outliers(df):

    numeric = numeric_columns(df)

    if not numeric:
        return 0

    total = 0

    for column in numeric:

        series = df[column].dropna()

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

    numeric = numeric_columns(df)

    if len(numeric) == 0:
        return df.index[:0], df.index

    work = df[numeric].copy()

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    work = work.fillna(
        work.median(numeric_only=True)
    )

    if len(work) < 10:
        return df.index[:0], df.index

    contamination = max(
        0.001,
        min(0.20, sensitivity / 100)
    )

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=150
    )

    predictions = model.fit_predict(work)

    anomaly_indices = df.index[predictions == -1]
    normal_indices = df.index[predictions == 1]

    return anomaly_indices, normal_indices


def find_column(df, possible_names):

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in possible_names:

        key = name.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def clean_dataset(df):

    cleaned = df.copy()

    before_rows = len(cleaned)

    duplicate_count = int(cleaned.duplicated().sum())

    cleaned = cleaned.drop_duplicates()

    rows_removed = before_rows - len(cleaned)

    numeric_cols = numeric_columns(cleaned)

    values_filled = 0

    for column in numeric_cols:

        missing_before = int(cleaned[column].isna().sum())

        if missing_before > 0:

            median_value = cleaned[column].median()

            cleaned[column] = cleaned[column].fillna(
                median_value
            )

            values_filled += missing_before

    categorical_cols = categorical_columns(cleaned)

    for column in categorical_cols:

        missing_before = int(cleaned[column].isna().sum())

        if missing_before > 0:

            mode = cleaned[column].mode()

            if len(mode) > 0:

                cleaned[column] = cleaned[column].fillna(
                    mode.iloc[0]
                )

                values_filled += missing_before

    city_column = find_column(
        cleaned,
        ["City"]
    )

    city_standardized = 0

    if city_column:

        before = cleaned[city_column].astype(str)

        standardized = (
            before
            .str.strip()
            .str.title()
        )

        city_standardized = int(
            (before != standardized).sum()
        )

        cleaned[city_column] = standardized

    return cleaned, {
        "rows_removed": rows_removed,
        "duplicates_removed": duplicate_count,
        "values_filled": values_filled,
        "city_standardized": city_standardized,
    }


def business_metrics(df):

    sales_col = find_column(
        df,
        [
            "Sales",
            "Sale",
            "Revenue",
            "Amount",
            "Total Sales",
        ]
    )

    profit_col = find_column(
        df,
        [
            "Profit",
            "Net Profit",
        ]
    )

    result = {}

    if sales_col:

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).dropna()

        if len(sales) > 0:

            result["sales_column"] = sales_col
            result["total_sales"] = round(
                float(sales.sum()),
                2
            )
            result["average_sale"] = round(
                float(sales.mean()),
                2
            )
            result["highest_sale"] = round(
                float(sales.max()),
                2
            )

    if profit_col and sales_col:

        sales_values = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        )

        profit_values = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        )

        valid = pd.DataFrame(
            {
                "sales": sales_values,
                "profit": profit_values,
            }
        ).dropna()

        if len(valid) > 0 and valid["sales"].sum() != 0:

            margin = (
                valid["profit"].sum()
                / valid["sales"].sum()
            ) * 100

            result["profit_margin"] = round(
                float(margin),
                2
            )

    return result


def create_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Cleaned Data"
        )

    output.seek(0)

    return output


def create_report_excel(df):

    quality = quality_metrics(df)

    outliers = iqr_outliers(df)

    anomalies, normal = isolation_forest(
        df,
        st.session_state.sensitivity
    )

    business = business_metrics(df)

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

        quality_df = pd.DataFrame(
            [
                {
                    "Quality Score": quality["score"],
                    "Missing Values": quality["missing"],
                    "Duplicates": quality["duplicates"],
                    "Invalid Values": quality["invalid"],
                    "IQR Outliers": outliers,
                    "ML Anomalies": len(anomalies),
                    "Normal Records": len(normal),
                }
            ]
        )

        quality_df.to_excel(
            writer,
            index=False,
            sheet_name="Quality Summary"
        )

        if business:

            business_df = pd.DataFrame(
                [business]
            )

            business_df.to_excel(
                writer,
                index=False,
                sheet_name="Business Metrics"
            )

    output.seek(0)

    return output


def create_zip(df):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        csv_data = df.to_csv(index=False)

        z.writestr(
            "cleaned_data.csv",
            csv_data
        )

        report = create_report_excel(df)

        z.writestr(
            "DataGuard_AI_Report.xlsx",
            report.getvalue()
        )

    zip_buffer.seek(0)

    return zip_buffer


def generate_gemini_analysis(
    dataset_name,
    df,
    quality,
    outliers,
    normal,
    anomalies,
    sensitivity,
):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        return (
            "Gemini API key is not configured.\n\n"
            "Please add GEMINI_API_KEY to "
            "Streamlit Secrets."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        business = business_metrics(df)

        prompt = f"""
You are a senior data quality and business
analytics consultant.

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
6. An anomaly does NOT automatically mean
   an incorrect record.
7. Clearly distinguish confirmed data-quality
   problems from statistical signals.
8. Give practical recommendations.
9. Do not claim that a value is wrong unless
   the supplied data confirms it.

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
            input=prompt,
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
        "# 🛡️ DataGuard AI"
    )

    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    st.markdown(
    "<div style='color:#94a3b8; font-size:11px; "
    "font-weight:700; letter-spacing:1.5px; "
    "margin:8px 0 10px;'>WORKSPACE</div>",
    unsafe_allow_html=True,
    )

    pages = [
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
    ]

    for page in pages:

        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),
        ):

            st.session_state.page = page
            st.rerun()

    st.divider()

    st.markdown(
    "<div style='color:#94a3b8; font-size:11px; "
    "font-weight:700; letter-spacing:1.5px; "
    "margin:8px 0 10px;'>SYSTEM</div>",
    unsafe_allow_html=True,
    )

    if st.button(
        "⚙️ Settings",
        use_container_width=True
    ):

        st.info(
            "DataGuard AI settings are currently "
            "managed through the application configuration."
        )

    if st.button(
        "❓ Help",
        use_container_width=True
    ):

        st.info(
            "Upload CSV/XLSX/XLS data and use the "
            "workflow to profile, validate, detect "
            "anomalies, clean and analyze it."
        )

    st.divider()

    st.success(
        "● All systems operational"
    )


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

if st.session_state.df is None:

    default_file = "sales_data_raw.csv"

    if os.path.exists(default_file):

        try:

            st.session_state.df = pd.read_csv(
                default_file
            )

            st.session_state.dataset_name = (
                default_file
            )

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

current_page = st.session_state.page

page_map = {
    "Upload Data": 0,
    "Data Profile": 1,
    "Quality Checks": 2,
    "Anomaly Detection": 3,
    "Data Cleaning": 4,
    "Analytics": 5,
    "Power BI": 6,
    "AI Analysis": 7,
}

current_step = page_map.get(
    current_page,
    -1
)

cols = st.columns(8)

for i, step in enumerate(workflow):

    with cols[i]:

        if i < current_step:

            st.success(
                step + " ✓"
            )

        elif i == current_step:

            st.info(step)

        else:

            st.caption(step)


st.divider()


# ============================================================
# DATA CHECK
# ============================================================

df = st.session_state.df


# ============================================================
# DASHBOARD
# ============================================================

if current_page == "Dashboard":

    st.title(
        "Dashboard",
        anchor=False
    )

    st.caption(
        f"Dataset: {st.session_state.dataset_name}"
    )

    if df is None:

        st.warning(
            "No dataset loaded. Go to Upload Data."
        )

    else:

        quality = quality_metrics(df)

        anomalies, normal = isolation_forest(
            df,
            st.session_state.sensitivity
        )

        st.subheader(
            "Data Quality Overview",
            anchor=False
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        c1.metric(
            "Data Quality Score",
            f'{quality["score"]:.2f}%'
        )

        c2.metric(
            "Total Records",
            f"{len(df):,}"
        )

        c3.metric(
            "Columns",
            len(df.columns)
        )

        c4.metric(
            "Missing Values",
            f'{quality["missing"]:,}'
        )

        c5.metric(
            "Duplicates",
            f'{quality["duplicates"]:,}'
        )

        c6.metric(
            "Anomalies",
            f"{len(anomalies):,}"
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            st.subheader(
                "Quality Dimensions",
                anchor=False
            )

            total_cells = len(df) * len(df.columns)

            completeness = (
                100
                - (
                    quality["missing"]
                    / total_cells
                    * 100
                )
                if total_cells
                else 0
            )

            uniqueness = (
                100
                - (
                    quality["duplicates"]
                    / len(df)
                    * 100
                )
                if len(df)
                else 0
            )

            validity = (
                100
                - (
                    quality["invalid"]
                    / len(df)
                    * 100
                )
                if len(df)
                else 0
            )

            dimensions = pd.DataFrame(
                {
                    "Dimension": [
                        "Completeness",
                        "Consistency",
                        "Validity",
                        "Uniqueness",
                    ],
                    "Score": [
                        completeness,
                        quality["score"],
                        validity,
                        uniqueness,
                    ],
                }
            )

            st.bar_chart(
                dimensions.set_index(
                    "Dimension"
                )
            )

        with right:

            st.subheader(
                "Anomaly Detection",
                anchor=False
            )

            a1, a2, a3 = st.columns(3)

            a1.metric(
                "Normal",
                f"{len(normal):,}"
            )

            a2.metric(
                "Anomalies",
                f"{len(anomalies):,}"
            )

            anomaly_rate = (
                len(anomalies)
                / len(df)
                * 100
                if len(df)
                else 0
            )

            a3.metric(
                "Rate",
                f"{anomaly_rate:.2f}%"
            )

            st.caption(
                "Isolation Forest screening signals. "
                "Anomalies should be reviewed before "
                "being treated as incorrect data."
            )

        st.divider()

        st.subheader(
            "Dataset Overview",
            anchor=False
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# UPLOAD DATA
# ============================================================

elif current_page == "Upload Data":

    st.title(
        "Upload Data",
        anchor=False
    )

    st.subheader(
        "Upload your dataset",
        anchor=False
    )

    st.caption(
        "Maximum file size: 50 MB"
    )

    uploaded = st.file_uploader(
        "Drag and drop your CSV, XLSX or XLS file",
        type=["csv", "xlsx", "xls"],
        help="Maximum file size: 50 MB",
    )

    if uploaded is not None:

        try:

            new_df = read_uploaded_file(
                uploaded
            )

            st.session_state.df = new_df

            st.session_state.cleaned_df = None

            st.session_state.dataset_name = (
                uploaded.name
            )

            st.session_state.gemini_result = None

            st.success(
                f"Successfully loaded "
                f"{uploaded.name}"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Rows",
                f"{len(new_df):,}"
            )

            c2.metric(
                "Columns",
                len(new_df.columns)
            )

            c3.metric(
                "File Type",
                uploaded.name.split(".")[-1].upper()
            )

            st.dataframe(
                new_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as error:

            st.error(
                f"Unable to read file: {error}"
            )

    elif df is not None:

        st.info(
            f"Current dataset: "
            f"{st.session_state.dataset_name}"
        )


# ============================================================
# DATA PROFILE
# ============================================================

elif current_page == "Data Profile":

    st.title(
        "Data Profile",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        numeric = numeric_columns(df)
        categorical = categorical_columns(df)
        dates = datetime_columns(df)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Rows",
            f"{len(df):,}"
        )

        c2.metric(
            "Columns",
            len(df.columns)
        )

        c3.metric(
            "Numeric",
            len(numeric)
        )

        c4.metric(
            "Categorical",
            len(categorical)
        )

        st.divider()

        st.subheader(
            "Column Profile",
            anchor=False
        )

        profile = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(df[c].dtype)
                    for c in df.columns
                ],
                "Non-Null": [
                    int(df[c].notna().sum())
                    for c in df.columns
                ],
                "Missing": [
                    int(df[c].isna().sum())
                    for c in df.columns
                ],
                "Unique": [
                    int(df[c].nunique(dropna=True))
                    for c in df.columns
                ],
            }
        )

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "Data Preview",
            anchor=False
        )

        st.dataframe(
            df.head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

elif current_page == "Quality Checks":

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

        outliers = iqr_outliers(df)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Quality Score",
            f'{quality["score"]:.2f}%'
        )

        c2.metric(
            "Missing",
            f'{quality["missing"]:,}'
        )

        c3.metric(
            "Duplicates",
            f'{quality["duplicates"]:,}'
        )

        c4.metric(
            "Invalid",
            f'{quality["invalid"]:,}'
        )

        c5.metric(
            "IQR Outliers",
            f"{outliers:,}"
        )

        st.divider()

        checks = pd.DataFrame(
            {
                "Check": [
                    "Completeness",
                    "Consistency",
                    "Validity",
                    "Uniqueness",
                    "Outliers",
                    "Data Types",
                ],
                "Status": [
                    (
                        "Pass"
                        if quality["missing"] == 0
                        else "Review"
                    ),
                    "Review",
                    (
                        "Pass"
                        if quality["invalid"] == 0
                        else "Review"
                    ),
                    (
                        "Pass"
                        if quality["duplicates"] == 0
                        else "Review"
                    ),
                    (
                        "Review"
                        if outliers > 0
                        else "Pass"
                    ),
                    "Pass",
                ],
            }
        )

        st.subheader(
            "Quality Check Results",
            anchor=False
        )

        st.dataframe(
            checks,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif current_page == "Anomaly Detection":

    st.title(
        "Anomaly Detection",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        st.subheader(
            "Isolation Forest",
            anchor=False
        )

        sensitivity = st.slider(
            "Sensitivity",
            min_value=1,
            max_value=20,
            value=st.session_state.sensitivity,
            help=(
                "Higher sensitivity flags more records "
                "as potential anomalies."
            ),
        )

        st.session_state.sensitivity = sensitivity

        anomalies, normal = isolation_forest(
            df,
            sensitivity
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Normal",
            f"{len(normal):,}"
        )

        c2.metric(
            "Anomalies",
            f"{len(anomalies):,}"
        )

        rate = (
            len(anomalies)
            / len(df)
            * 100
            if len(df)
            else 0
        )

        c3.metric(
            "Anomaly Rate",
            f"{rate:.2f}%"
        )

        st.caption(
            "Isolation Forest results are screening "
            "signals, not proof that records are incorrect."
        )

        st.divider()

        numeric = numeric_columns(df)

        if len(numeric) >= 1:

            selected_column = st.selectbox(
                "Select numeric column",
                numeric,
            )

            chart_df = df[
                [selected_column]
            ].copy()

            st.subheader(
                "Distribution",
                anchor=False
            )

            st.line_chart(
                chart_df.reset_index(
                    drop=True
                )
            )

        anomaly_df = df.loc[
            anomalies
        ].copy()

        st.subheader(
            "Detected Anomalies",
            anchor=False
        )

        if len(anomaly_df) > 0:

            st.dataframe(
                anomaly_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No anomalies detected."
            )


# ============================================================
# DATA CLEANING
# ============================================================

elif current_page == "Data Cleaning":

    st.title(
        "Data Cleaning",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        st.subheader(
            "Cleaning Recommendations",
            anchor=False
        )

        quality = quality_metrics(df)

        if quality["missing"] > 0:

            st.info(
                f"{quality['missing']:,} missing values "
                "can be filled using appropriate "
                "statistical or categorical values."
            )

        if quality["duplicates"] > 0:

            st.info(
                f"{quality['duplicates']:,} duplicate rows "
                "can be removed."
            )

        if quality["invalid"] > 0:

            st.info(
                f"{quality['invalid']:,} potentially invalid "
                "numeric-like values require review."
            )

        st.divider()

        if st.button(
            "🧹 Clean Dataset",
            type="primary",
            use_container_width=True,
        ):

            cleaned, cleaning_stats = clean_dataset(
                df
            )

            st.session_state.cleaned_df = cleaned

            st.success(
                "Cleaning completed successfully."
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Rows Removed",
                cleaning_stats["rows_removed"]
            )

            c2.metric(
                "Duplicates Removed",
                cleaning_stats[
                    "duplicates_removed"
                ]
            )

            c3.metric(
                "Values Filled",
                cleaning_stats[
                    "values_filled"
                ]
            )

            c4.metric(
                "City Standardized",
                cleaning_stats[
                    "city_standardized"
                ]
            )

        if st.session_state.cleaned_df is not None:

            cleaned_df = (
                st.session_state.cleaned_df
            )

            st.divider()

            st.subheader(
                "Before → After",
                anchor=False
            )

            c1, c2 = st.columns(2)

            with c1:

                st.caption("Before")

                st.dataframe(
                    df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            with c2:

                st.caption("After")

                st.dataframe(
                    cleaned_df.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "⬇️ Download Cleaned CSV",
                cleaned_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# ANALYTICS
# ============================================================

elif current_page == "Analytics":

    st.title(
        "Analytics",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        analysis_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        metrics = business_metrics(
            analysis_df
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Sales",
            (
                f'{metrics["total_sales"]:,.2f}'
                if "total_sales" in metrics
                else "N/A"
            )
        )

        c2.metric(
            "Average Sale",
            (
                f'{metrics["average_sale"]:,.2f}'
                if "average_sale" in metrics
                else "N/A"
            )
        )

        c3.metric(
            "Highest Sale",
            (
                f'{metrics["highest_sale"]:,.2f}'
                if "highest_sale" in metrics
                else "N/A"
            )
        )

        c4.metric(
            "Profit Margin",
            (
                f'{metrics["profit_margin"]:.2f}%'
                if "profit_margin" in metrics
                else "N/A"
            )
        )

        st.divider()

        sales_col = find_column(
            analysis_df,
            [
                "Sales",
                "Sale",
                "Revenue",
                "Amount",
                "Total Sales",
            ]
        )

        category_col = find_column(
            analysis_df,
            [
                "Category",
                "Product Category",
            ]
        )

        region_col = find_column(
            analysis_df,
            [
                "Region"
            ]
        )

        if sales_col:

            left, right = st.columns(2)

            with left:

                st.subheader(
                    "Sales Distribution",
                    anchor=False
                )

                st.bar_chart(
                    analysis_df[
                        sales_col
                    ].value_counts(
                        bins=10
                    ).sort_index()
                )

            with right:

                if category_col:

                    st.subheader(
                        "Sales by Category",
                        anchor=False
                    )

                    category_sales = (
                        analysis_df
                        .groupby(
                            category_col
                        )[sales_col]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        category_sales
                    )

                elif region_col:

                    st.subheader(
                        "Sales by Region",
                        anchor=False
                    )

                    region_sales = (
                        analysis_df
                        .groupby(
                            region_col
                        )[sales_col]
                        .sum()
                        .sort_values(
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        region_sales
                    )

        st.divider()

        st.subheader(
            "Numeric Correlation",
            anchor=False
        )

        numeric = numeric_columns(
            analysis_df
        )

        if len(numeric) >= 2:

            correlation = (
                analysis_df[numeric]
                .corr()
            )

            st.dataframe(
                correlation.round(2),
                use_container_width=True,
            )

        else:

            st.info(
                "At least two numeric columns "
                "are required for correlation."
            )


# ============================================================
# POWER BI
# ============================================================

elif current_page == "Power BI":

    st.title(
        "Power BI Export Center",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        export_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        st.subheader(
            "Power BI Ready Data",
            anchor=False
        )

        st.write(
            "Use the cleaned dataset as the source "
            "for your Power BI dashboard."
        )

        c1, c2 = st.columns(2)

        with c1:

            excel_file = create_excel(
                export_df
            )

            st.download_button(
                "⬇️ Download Power BI Excel",
                excel_file.getvalue(),
                file_name="DataGuard_AI_PowerBI.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        with c2:

            st.download_button(
                "⬇️ Download Power BI CSV",
                export_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="DataGuard_AI_PowerBI.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.divider()

        st.subheader(
            "Recommended Power BI Pages",
            anchor=False
        )

        recommendations = pd.DataFrame(
            {
                "Page": [
                    "Executive Overview",
                    "Data Quality",
                    "Sales Analysis",
                    "Anomaly Analysis",
                    "Regional Analysis",
                ],
                "Recommended Visuals": [
                    "KPI cards, sales trend, category chart",
                    "Quality score, missing values, duplicates",
                    "Sales, profit, category and monthly trends",
                    "Normal vs anomaly, scatter plot",
                    "Region performance and ranking",
                ],
            }
        )

        st.dataframe(
            recommendations,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# AI ANALYSIS
# ============================================================

elif current_page == "AI Analysis":

    st.title(
        "AI Analysis",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        st.subheader(
            "Gemini AI Analysis",
            anchor=False
        )

        st.write(
            "Generate an AI-assisted interpretation "
            "of your measured data-quality and "
            "business metrics."
        )

        if st.button(
            "✨ Generate AI Analysis",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Gemini is analyzing the dataset..."
            ):

                quality = quality_metrics(
                    df
                )

                outliers = iqr_outliers(
                    df
                )

                anomalies, normal = (
                    isolation_forest(
                        df,
                        st.session_state.sensitivity
                    )
                )

                result = generate_gemini_analysis(
                    st.session_state.dataset_name,
                    df,
                    quality,
                    outliers,
                    len(normal),
                    len(anomalies),
                    st.session_state.sensitivity,
                )

                st.session_state.gemini_result = result

        if st.session_state.gemini_result:

            st.divider()

            st.markdown(
                st.session_state.gemini_result
            )


# ============================================================
# REPORTS
# ============================================================

elif current_page == "Reports":

    st.title(
        "Reports",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        st.subheader(
            "DataGuard AI Report",
            anchor=False
        )

        quality = quality_metrics(
            df
        )

        outliers = iqr_outliers(
            df
        )

        anomalies, normal = isolation_forest(
            df,
            st.session_state.sensitivity
        )

        st.markdown(
            f"""
### Executive Summary

**Dataset:** {st.session_state.dataset_name}

**Records:** {len(df):,}

**Columns:** {len(df.columns):,}

**Data Quality Score:** {quality["score"]:.2f}%

**Missing Values:** {quality["missing"]:,}

**Duplicate Rows:** {quality["duplicates"]:,}

**Invalid Values:** {quality["invalid"]:,}

**IQR Outliers:** {outliers:,}

**ML Anomalies:** {len(anomalies):,}

**Normal Records:** {len(normal):,}

**Anomaly Rate:** {
                (len(anomalies) / len(df) * 100)
                if len(df)
                else 0
            :.2f}%
"""
        )

        st.divider()

        report_file = create_report_excel(
            df
        )

        zip_file = create_zip(
            (
                st.session_state.cleaned_df
                if st.session_state.cleaned_df is not None
                else df
            )
        )

        c1, c2 = st.columns(2)

        with c1:

            st.download_button(
                "⬇️ Download Excel Report",
                report_file.getvalue(),
                file_name="DataGuard_AI_Report.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        with c2:

            st.download_button(
                "⬇️ Download Complete ZIP",
                zip_file.getvalue(),
                file_name="DataGuard_AI_Portfolio_Package.zip",
                mime="application/zip",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ DataGuard AI • Portfolio Edition • "
    "Python • Pandas • Scikit-learn • Streamlit • Gemini"
)
