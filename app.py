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
# SAFE CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f6f8fb;
        }

        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 3rem;
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
    "uploaded_file_id": None,
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
# DATETIME DETECTION
# ============================================================

def safe_to_datetime(series):

    try:
        converted = pd.to_datetime(
            series,
            errors="coerce",
        )

        if len(converted) == 0:
            return None

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.70:
            return converted

        return None

    except Exception:
        return None


def detect_datetime_columns(df):

    datetime_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            datetime_columns.append(column)
            continue

        if df[column].dtype == "object":

            converted = safe_to_datetime(
                df[column]
            )

            if converted is not None:
                datetime_columns.append(column)

    return datetime_columns


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

        column_lower = str(column).lower()

        unique_ratio = (
            df[column].nunique(dropna=True)
            / max(len(df), 1)
        )

        if (
            unique_ratio > 0.95
            and (
                "id" in column_lower
                or "code" in column_lower
                or "number" in column_lower
                or "invoice" in column_lower
                or "customer" in column_lower
            )
        ):
            identifier_columns.append(column)

    return identifier_columns


# ============================================================
# DATASET PROFILE
# ============================================================

def profile_dataset(df):

    datetime_columns = detect_datetime_columns(df)

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in datetime_columns
    ]

    missing_by_column = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "missing_by_column": missing_by_column,
        "total_cells": (
            df.shape[0] * df.shape[1]
        ),
    }


# ============================================================
# INVALID VALUE DETECTION
# ============================================================

def detect_invalid_values(df):

    invalid_details = {}

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numeric_columns:

        series = df[column]

        try:

            numeric_series = (
                series.dropna()
                .astype(float)
            )

            infinite_count = int(
                np.isinf(
                    numeric_series
                ).sum()
            )

        except Exception:

            infinite_count = 0

        if infinite_count > 0:
            invalid_details[column] = infinite_count

    invalid_count = sum(
        invalid_details.values()
    )

    return (
        invalid_count,
        invalid_details,
    )


# ============================================================
# CITY STANDARDIZATION
# ============================================================

def standardize_city_column(df):

    result = df.copy()

    for column in result.columns:

        if result[column].dtype != "object":
            continue

        column_lower = str(column).lower()

        if "city" not in column_lower:
            continue

        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(CITY_MAPPING)
        )

    return result


# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

def detect_iqr_outliers(df):

    results = []

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
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

        outlier_count = int(mask.sum())

        if outlier_count > 0:

            results.append(
                {
                    "Column": column,
                    "Q1": q1,
                    "Q3": q3,
                    "Lower_Bound": lower_bound,
                    "Upper_Bound": upper_bound,
                    "Outlier_Count": outlier_count,
                }
            )

    return pd.DataFrame(results)


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
        )
        .copy()
    )

    if numeric_df.empty:

        return {
            "anomaly_count": 0,
            "anomaly_indices": [],
            "status": (
                "No numeric columns available."
            ),
        }

    numeric_df = numeric_df.dropna(
        axis=1,
        how="all",
    )

    if numeric_df.empty:

        return {
            "anomaly_count": 0,
            "anomaly_indices": [],
            "status": (
                "No usable numeric columns."
            ),
        }

    numeric_df = numeric_df.fillna(
        numeric_df.median(
            numeric_only=True
        )
    )

    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique() > 1,
    ]

    if (
        numeric_df.empty
        or len(numeric_df) < 10
    ):

        return {
            "anomaly_count": 0,
            "anomaly_indices": [],
            "status": (
                "Not enough variation "
                "for Isolation Forest."
            ),
        }

    contamination = contamination_pct / 100

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    predictions = model.fit_predict(
        numeric_df
    )

    anomaly_mask = (
        predictions == -1
    )

    anomaly_indices = (
        numeric_df.index[
            anomaly_mask
        ].tolist()
    )

    return {
        "anomaly_count": int(
            anomaly_mask.sum()
        ),
        "anomaly_indices": anomaly_indices,
        "status": (
            "Isolation Forest analysis "
            "completed."
        ),
    }


# ============================================================
# QUALITY SCORE
# ============================================================

def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count,
):

    if total_cells == 0:
        return 0.0

    missing_penalty = (
        missing_count
        / total_cells
        * 100
    )

    duplicate_penalty = (
        duplicate_count
        / total_cells
        * 100
    )

    invalid_penalty = (
        invalid_count
        / total_cells
        * 100
    )

    score = 100 - (
        missing_penalty * 0.50
        + duplicate_penalty * 0.30
        + invalid_penalty * 0.20
    )

    return round(
        max(
            0.0,
            min(
                100.0,
                score,
            ),
        ),
        2,
    )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(df):

    cleaned = df.copy()

    cleaned = standardize_city_column(
        cleaned
    )

    cleaned = cleaned.drop_duplicates()

    numeric_columns = (
        cleaned.select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numeric_columns:

        cleaned[column] = (
            cleaned[column]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        if cleaned[column].isna().any():

            median_value = (
                cleaned[column].median()
            )

            if pd.notna(median_value):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

    object_columns = (
        cleaned.select_dtypes(
            include="object"
        )
        .columns
    )

    for column in object_columns:

        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
        )

        cleaned[column] = (
            cleaned[column]
            .replace(
                {
                    "nan": np.nan,
                    "None": np.nan,
                    "NULL": np.nan,
                    "null": np.nan,
                    "N/A": np.nan,
                    "n/a": np.nan,
                }
            )
        )

    return cleaned


# ============================================================
# FIND SALES COLUMN
# ============================================================

def find_sales_column(df):

    priority_names = [
        "sales",
        "sale",
        "revenue",
        "amount",
        "total_sales",
        "total sales",
        "price",
        "value",
        "profit",
    ]

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if not numeric_columns:
        return None

    for name in priority_names:

        for column in numeric_columns:

            if (
                str(column)
                .lower()
                .strip()
                == name
            ):
                return column

    for column in numeric_columns:

        column_lower = (
            str(column).lower()
        )

        if any(
            word in column_lower
            for word in [
                "sales",
                "revenue",
                "amount",
                "price",
                "value",
            ]
        ):
            return column

    return numeric_columns[0]


# ============================================================
# BUSINESS DATA
# ============================================================

def prepare_business_data(df):

    business_df = standardize_city_column(
        df.copy()
    )

    sales_column = find_sales_column(
        business_df
    )

    return (
        business_df,
        sales_column,
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

    exports[
        "DataGuard_Cleaned_Data.csv"
    ] = (
        cleaned_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    profile_rows = []

    for column in cleaned_df.columns:

        profile_rows.append(
            {
                "Column": column,
                "Data_Type": str(
                    cleaned_df[column].dtype
                ),
                "Missing_Count": int(
                    cleaned_df[column]
                    .isna()
                    .sum()
                ),
                "Unique_Count": int(
                    cleaned_df[column]
                    .nunique(
                        dropna=True
                    )
                ),
            }
        )

    profile_df = pd.DataFrame(
        profile_rows
    )

    exports[
        "DataGuard_Data_Profile.csv"
    ] = (
        profile_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    if sales_column is None:
        return exports

    sales = pd.to_numeric(
        business_df[sales_column],
        errors="coerce",
    )

    sales_summary = pd.DataFrame(
        {
            "Metric": [
                "Total Sales",
                "Average Sale",
                "Minimum Sale",
                "Maximum Sale",
                "Sales Records",
            ],
            "Value": [
                sales.sum(),
                sales.mean(),
                sales.min(),
                sales.max(),
                sales.count(),
            ],
        }
    )

    exports[
        "DataGuard_Sales_Summary.csv"
    ] = (
        sales_summary
        .to_csv(index=False)
        .encode("utf-8")
    )

    city_column = None

    for column in business_df.columns:

        if "city" in str(column).lower():

            city_column = column
            break

    if city_column:

        city_summary = (
            business_df
            .groupby(city_column)[
                sales_column
            ]
            .agg(
                Sales="sum",
                Records="count",
                Average_Sale="mean",
            )
            .reset_index()
            .sort_values(
                "Sales",
                ascending=False,
            )
        )

        exports[
            "DataGuard_City_Summary.csv"
        ] = (
            city_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    category_column = None

    for column in business_df.columns:

        column_lower = (
            str(column).lower()
        )

        if (
            "category" in column_lower
            or "segment" in column_lower
        ):

            category_column = column
            break

    if category_column:

        category_summary = (
            business_df
            .groupby(category_column)[
                sales_column
            ]
            .agg(
                Sales="sum",
                Records="count",
                Average_Sale="mean",
            )
            .reset_index()
            .sort_values(
                "Sales",
                ascending=False,
            )
        )

        exports[
            "DataGuard_Category_Summary.csv"
        ] = (
            category_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    product_column = None

    for column in business_df.columns:

        column_lower = (
            str(column).lower()
        )

        if (
            "product" in column_lower
            or "item" in column_lower
        ):

            product_column = column
            break

    if product_column:

        product_summary = (
            business_df
            .groupby(product_column)[
                sales_column
            ]
            .agg(
                Sales="sum",
                Records="count",
            )
            .reset_index()
            .sort_values(
                "Sales",
                ascending=False,
            )
        )

        exports[
            "DataGuard_Product_Summary.csv"
        ] = (
            product_summary
            .to_csv(index=False)
            .encode("utf-8")
        )

    datetime_columns = (
        detect_datetime_columns(
            business_df
        )
    )

    if datetime_columns:

        date_column = datetime_columns[0]

        monthly_df = business_df.copy()

        monthly_df["_DG_Date"] = (
            pd.to_datetime(
                monthly_df[date_column],
                errors="coerce",
            )
        )

        monthly_df = monthly_df.dropna(
            subset=["_DG_Date"]
        )

        if not monthly_df.empty:

            monthly_summary = (
                monthly_df
                .assign(
                    Month=monthly_df[
                        "_DG_Date"
                    ]
                    .dt
                    .to_period("M")
                    .astype(str)
                )
                .groupby("Month")[
                    sales_column
                ]
                .agg(
                    Sales="sum",
                    Records="count",
                )
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


# ============================================================
# ZIP CREATION
# ============================================================

def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for filename, content in exports.items():

            zip_file.writestr(
                filename,
                content,
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GEMINI API KEY
# ============================================================

def get_gemini_api_key():

    try:

        if "GEMINI_API_KEY" in st.secrets:

            key = st.secrets[
                "GEMINI_API_KEY"
            ]

            if key:
                return str(key).strip()

    except Exception:
        pass

    key = os.getenv(
        "GEMINI_API_KEY"
    )

    if key:
        return key.strip()

    return None


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def generate_gemini_analysis(analysis):

    api_key = get_gemini_api_key()

    if not api_key:
        return {
            "status": "error",
            "message": (
                "Gemini API key was not found. "
                "Add GEMINI_API_KEY to Streamlit Secrets."
            ),
        }

    try:
        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        report_data = {
            "Dataset Rows": analysis["rows"],
            "Dataset Columns": analysis["columns"],
            "Numeric Columns": analysis["numeric_count"],
            "Categorical Columns": analysis["categorical_count"],
            "Date Time Columns": analysis["datetime_count"],
            "Missing Cells": analysis["missing_count"],
            "Duplicate Records": analysis["duplicate_count"],
            "Invalid Values": analysis["invalid_count"],
            "Invalid Details": analysis["invalid_details"],
            "IQR Outliers": analysis["iqr_outlier_count"],
            "ML Anomalies": analysis["ml_anomaly_count"],
            "Quality Score": analysis["quality_score"],
            "Sales Column": analysis["sales_column"],
        }

        prompt = f"""
You are the AI analysis engine inside DataGuard AI.

DataGuard AI is an AI-powered data quality,
anomaly detection and business analytics platform.

Analyze ONLY the supplied DataGuard AI report.

Do not invent facts, numbers, columns, business
results, causes or conclusions.

Important rules:

- Missing values and duplicate records are confirmed
  data-quality findings.
- Infinite numeric values are invalid values.
- IQR outliers are statistical screening signals.
- Isolation Forest anomalies are ML screening signals,
  not confirmed errors.
- Do not say every anomaly is an error.
- Cleaning has already been performed by DataGuard AI.
- Do not claim that original records were deleted
  unless the report explicitly says so.
- Clearly distinguish confirmed findings from
  possible explanations.

Create a professional report with these sections:

## Overall Assessment

Give a concise assessment of dataset quality.

## Confirmed Data Quality Problems

Discuss only confirmed issues supported by the report.

## Statistical Outlier Findings

Explain the IQR findings.

## ML Anomaly Findings

Explain the Isolation Forest findings.

## Possible Root Causes

Give possible explanations only.
Clearly label them as possible causes.

## Cleaning Results

Explain the cleaning operations.

## Business Impact

Explain possible effects on reporting and analytics.

## Power BI Recommendations

Give practical recommendations for a reliable
Power BI dashboard.

DATA GUARD REPORT:

{report_data}
"""

        # ====================================================
        # CURRENT GEMINI INTERACTIONS API
        # ====================================================

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
        )

        # ====================================================
        # CORRECT RESPONSE EXTRACTION
        # ====================================================

        output_text = getattr(
            interaction,
            "output_text",
            None,
        )

        if output_text:

            output_text = str(
                output_text
            ).strip()

            if output_text:

                return {
                    "status": "success",
                    "content": output_text,
                }

        # ====================================================
        # FALLBACK FOR STEP-BASED RESPONSE
        # ====================================================

        steps = getattr(
            interaction,
            "steps",
            None,
        )

        if steps:

            text_parts = []

            for step in steps:

                step_type = getattr(
                    step,
                    "type",
                    None,
                )

                if step_type != "model_output":
                    continue

                content = getattr(
                    step,
                    "content",
                    None,
                )

                if not content:
                    continue

                for item in content:

                    item_type = getattr(
                        item,
                        "type",
                        None,
                    )

                    if item_type != "text":
                        continue

                    text = getattr(
                        item,
                        "text",
                        None,
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

            combined_text = "\n".join(
                text_parts
            ).strip()

            if combined_text:

                return {
                    "status": "success",
                    "content": combined_text,
                }

        return {
            "status": "error",
            "message": (
                "Gemini completed the request, "
                "but no text output was returned."
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "message": (
                f"Gemini API error: {exc}"
            ),
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🛡️ DataGuard AI"
    )

    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    st.write(
        "WORKSPACE"
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

    st.divider()

    st.write(
        "DETECTION SETTINGS"
    )

    contamination = st.slider(
        "Isolation Forest sensitivity",
        min_value=1,
        max_value=20,
        value=int(
            st.session_state[
                "contamination_pct"
            ]
        ),
        help=(
            "Higher sensitivity flags "
            "more records as unusual."
        ),
    )

    st.session_state[
        "contamination_pct"
    ] = contamination

    st.divider()

    st.write(
        "PIPELINE"
    )

    pipeline_steps = [
        "01  Upload",
        "02  Profile",
        "03  Quality",
        "04  Anomalies",
        "05  Clean",
        "06  Analytics",
        "07  Power BI",
        "08  AI",
    ]

    for step in pipeline_steps:
        st.caption(step)

    st.divider()

    st.caption(
        "DataGuard AI"
    )

    st.caption(
        "Portfolio Edition"
    )

    st.caption(
        "Python • Pandas • Scikit-learn"
    )

    st.caption(
        "Streamlit • Gemini"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.write(
    "DataGuard AI"
)

st.caption(
    "AI-powered data quality, anomaly detection "
    "and business analytics."
)


# ============================================================
# DATASET STATUS
# ============================================================

if (
    st.session_state.file_name
    and st.session_state.df is not None
):

    df = st.session_state.df

    if st.session_state.analysis_complete:

        st.info(
            f"📄 {st.session_state.file_name} • "
            f"{len(df):,} rows • "
            f"{len(df.columns):,} columns • "
            "✓ Analysis complete"
        )

    else:

        st.info(
            f"📄 {st.session_state.file_name} • "
            f"{len(df):,} rows • "
            f"{len(df.columns):,} columns • "
            "Ready for analysis"
        )


# ============================================================
# UPLOAD DATASET
# ============================================================

st.write(
    "Upload Dataset"
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)

if uploaded_file is not None:

    file_id = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if (
        st.session_state.uploaded_file_id
        != file_id
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

            if loaded_df.empty:

                st.error(
                    "The uploaded dataset is empty."
                )

            elif len(
                loaded_df.columns
            ) == 0:

                st.error(
                    "The dataset contains no columns."
                )

            else:

                st.session_state.df = (
                    loaded_df
                )

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.cleaned_df = None

                st.session_state.business_df = None

                st.session_state.analysis = None

                st.session_state.gemini_analysis = None

                st.session_state.analysis_complete = False

                st.session_state.uploaded_file_id = (
                    file_id
                )

                st.success(
                    f"Loaded {len(loaded_df):,} "
                    f"rows × "
                    f"{len(loaded_df.columns):,} "
                    f"columns."
                )

        except Exception as exc:

            st.error(
                f"Unable to read the file: {exc}"
            )


# ============================================================
# RUN ANALYSIS
# ============================================================

if st.session_state.df is not None:

    if st.button(
        "Run DataGuard Analysis",
        type="primary",
    ):

        with st.spinner(
            "Analyzing dataset..."
        ):

            df = (
                st.session_state.df.copy()
            )

            profile = profile_dataset(
                df
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

            (
                invalid_count,
                invalid_details,
            ) = detect_invalid_values(
                df
            )

            iqr_df = detect_iqr_outliers(
                df
            )

            if iqr_df.empty:

                iqr_outlier_count = 0

            else:

                iqr_outlier_count = int(
                    iqr_df[
                        "Outlier_Count"
                    ].sum()
                )

            ml_result = detect_ml_anomalies(
                df,
                st.session_state[
                    "contamination_pct"
                ],
            )

            quality_score = (
                build_quality_score(
                    profile[
                        "total_cells"
                    ],
                    missing_count,
                    duplicate_count,
                    invalid_count,
                )
            )

            cleaned_df = clean_dataset(
                df
            )

            (
                business_df,
                sales_column,
            ) = prepare_business_data(
                cleaned_df
            )

            analysis = {

                "rows": profile[
                    "rows"
                ],

                "columns": profile[
                    "columns"
                ],

                "numeric_count": len(
                    profile[
                        "numeric_columns"
                    ]
                ),

                "categorical_count": len(
                    profile[
                        "categorical_columns"
                    ]
                ),

                "datetime_count": len(
                    profile[
                        "datetime_columns"
                    ]
                ),

                "missing_count": missing_count,

                "duplicate_count": duplicate_count,

                "invalid_count": invalid_count,

                "invalid_details": invalid_details,

                "iqr_outlier_count": (
                    iqr_outlier_count
                ),

                "iqr_details": iqr_df,

                "ml_anomaly_count": (
                    ml_result[
                        "anomaly_count"
                    ]
                ),

                "ml_anomaly_indices": (
                    ml_result[
                        "anomaly_indices"
                    ]
                ),

                "ml_status": (
                    ml_result[
                        "status"
                    ]
                ),

                "quality_score": quality_score,

                "sales_column": sales_column,

                "analysis_timestamp": (
                    datetime.now()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                ),
            }

            st.session_state.analysis = (
                analysis
            )

            st.session_state.cleaned_df = (
                cleaned_df
            )

            st.session_state.business_df = (
                business_df
            )

            st.session_state.analysis_complete = (
                True
            )

            st.session_state.gemini_analysis = (
                None
            )

        st.success(
            "Analysis completed successfully."
        )


# ============================================================
# PAGE CONTENT
# ============================================================

analysis = st.session_state.analysis


if analysis is None:

    st.info(
        "Upload a dataset and click "
        "'Run DataGuard Analysis' to begin."
    )


else:

    df = st.session_state.df

    cleaned_df = (
        st.session_state.cleaned_df
    )

    business_df = (
        st.session_state.business_df
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        st.write(
            "Dashboard"
        )

        st.caption(
            "Monitor dataset health and quality signals."
        )

        score = analysis[
            "quality_score"
        ]

        st.write(
            "Overall Data Quality"
        )

        score_col, progress_col = (
            st.columns([1, 3])
        )

        with score_col:

            st.metric(
                "Quality Score",
                f"{score:.2f}/100",
            )

        with progress_col:

            st.progress(
                score / 100
            )

            if score >= 90:

                st.caption(
                    "Good data quality"
                )

            elif score >= 75:

                st.caption(
                    "Needs attention"
                )

            else:

                st.caption(
                    "Poor data quality"
                )

        st.divider()

        st.write(
            "Quality Signals"
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Missing Cells",
                f"{analysis['missing_count']:,}",
            )

        with c2:

            st.metric(
                "Duplicate Records",
                f"{analysis['duplicate_count']:,}",
            )

        with c3:

            st.metric(
                "IQR Outliers",
                f"{analysis['iqr_outlier_count']:,}",
            )

        with c4:

            st.metric(
                "ML Anomalies",
                f"{analysis['ml_anomaly_count']:,}",
            )

        st.divider()

        st.write(
            "Dataset Profile"
        )

        p1, p2, p3, p4, p5 = (
            st.columns(5)
        )

        with p1:

            st.metric(
                "Rows",
                f"{analysis['rows']:,}",
            )

        with p2:

            st.metric(
                "Columns",
                f"{analysis['columns']:,}",
            )

        with p3:

            st.metric(
                "Numeric",
                f"{analysis['numeric_count']:,}",
            )

        with p4:

            st.metric(
                "Categorical",
                f"{analysis['categorical_count']:,}",
            )

        with p5:

            st.metric(
                "Date / Time",
                f"{analysis['datetime_count']:,}",
            )

        st.divider()

        st.write(
            "Quality Monitoring"
        )

        monitoring_df = pd.DataFrame(
            {
                "Signal": [
                    "Missing",
                    "Duplicate",
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
                        "iqr_outlier_count"
                    ],
                    analysis[
                        "ml_anomaly_count"
                    ],
                ],
            }
        )

        st.dataframe(
            monitoring_df,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        st.write(
            "Business Snapshot"
        )

        sales_column = analysis[
            "sales_column"
        ]

        if sales_column:

            sales_series = pd.to_numeric(
                business_df[
                    sales_column
                ],
                errors="coerce",
            )

            b1, b2, b3 = (
                st.columns(3)
            )

            with b1:

                st.metric(
                    "Total Sales",
                    f"{sales_series.sum():,.2f}",
                )

            with b2:

                st.metric(
                    "Average Sale",
                    f"{sales_series.mean():,.2f}",
                )

            with b3:

                st.metric(
                    "Sales Records",
                    f"{sales_series.count():,}",
                )

        else:

            st.info(
                "No suitable sales/revenue "
                "column was detected."
            )

        st.divider()

        st.write(
            "Data Preview"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True,
            height=350,
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    elif page == "Data Quality":

        st.write(
            "Data Quality"
        )

        st.caption(
            "Detailed checks for missing, duplicate "
            "and invalid values."
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            st.metric(
                "Missing Cells",
                f"{analysis['missing_count']:,}",
            )

        with c2:

            st.metric(
                "Duplicate Rows",
                f"{analysis['duplicate_count']:,}",
            )

        with c3:

            st.metric(
                "Invalid Values",
                f"{analysis['invalid_count']:,}",
            )

        st.divider()

        st.write(
            "Missing Values by Column"
        )

        missing_table = (
            df.isna()
            .sum()
            .reset_index()
        )

        missing_table.columns = [
            "Column",
            "Missing_Count",
        ]

        missing_table = (
            missing_table[
                missing_table[
                    "Missing_Count"
                ]
                > 0
            ]
            .sort_values(
                "Missing_Count",
                ascending=False,
            )
        )

        if missing_table.empty:

            st.success(
                "No missing values detected."
            )

        else:

            st.dataframe(
                missing_table,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        st.write(
            "Invalid Values"
        )

        invalid_details = analysis.get(
            "invalid_details",
            {},
        )

        if (
            isinstance(
                invalid_details,
                dict,
            )
            and invalid_details
        ):

            invalid_table = pd.DataFrame(
                [
                    {
                        "Column": column,
                        "Invalid_Count": int(
                            count
                        ),
                    }
                    for column, count
                    in invalid_details.items()
                ]
            )

            st.dataframe(
                invalid_table,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.success(
                "No invalid numeric values detected."
            )

        st.divider()

        st.write(
            "Duplicate Records"
        )

        if analysis[
            "duplicate_count"
        ] == 0:

            st.success(
                "No duplicate records detected."
            )

        else:

            st.warning(
                f"{analysis['duplicate_count']:,} "
                "duplicate records detected."
            )

            duplicate_preview = (
                df[
                    df.duplicated(
                        keep=False
                    )
                ]
                .head(20)
            )

            st.dataframe(
                duplicate_preview,
                use_container_width=True,
                hide_index=True,
            )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        st.write(
            "Anomaly Detection"
        )

        st.caption(
            "Statistical and machine-learning "
            "based anomaly screening."
        )

        a1, a2 = (
            st.columns(2)
        )

        with a1:

            st.metric(
                "IQR Outliers",
                f"{analysis['iqr_outlier_count']:,}",
            )

        with a2:

            st.metric(
                "ML Anomalies",
                f"{analysis['ml_anomaly_count']:,}",
            )

        st.divider()

        st.write(
            "IQR Outlier Detection"
        )

        iqr_df = analysis[
            "iqr_details"
        ]

        if (
            iqr_df is None
            or iqr_df.empty
        ):

            st.success(
                "No IQR outliers detected."
            )

        else:

            display_iqr = (
                iqr_df.copy()
            )

            for column in [
                "Q1",
                "Q3",
                "Lower_Bound",
                "Upper_Bound",
            ]:

                if column in display_iqr.columns:

                    display_iqr[
                        column
                    ] = (
                        display_iqr[
                            column
                        ].round(2)
                    )

            st.dataframe(
                display_iqr,
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

        st.write(
            "Isolation Forest"
        )

        st.write(
            f"Current sensitivity: "
            f"{st.session_state.contamination_pct}%"
        )

        st.caption(
            "Higher sensitivity flags more records "
            "as unusual. These are screening signals, "
            "not automatically data errors."
        )

        st.info(
            analysis["ml_status"]
        )

        anomaly_indices = analysis[
            "ml_anomaly_indices"
        ]

        if anomaly_indices:

            valid_indices = [
                index
                for index in anomaly_indices
                if index in df.index
            ]

            if valid_indices:

                anomaly_preview = (
                    df.loc[
                        valid_indices
                    ]
                    .head(50)
                )

                st.write(
                    "Sample anomalous records"
                )

                st.dataframe(
                    anomaly_preview,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            st.success(
                "No ML anomalies were detected."
            )


    # ========================================================
    # CLEANING
    # ========================================================

    elif page == "Cleaning":

        st.write(
            "Data Cleaning"
        )

        st.caption(
            "Review the cleaned dataset before export."
        )

        original_rows = len(df)

        cleaned_rows = len(
            cleaned_df
        )

        c1, c2, c3 = (
            st.columns(3)
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
                f"{original_rows - cleaned_rows:,}",
            )

        st.divider()

        st.write(
            "Cleaning Actions"
        )

        st.markdown(
            """
- Standardizes supported city names
- Removes exact duplicate rows
- Replaces infinite numeric values
- Fills suitable numeric missing values using median
- Removes obvious text placeholders such as `NULL` and `N/A`
"""
        )

        st.divider()

        st.write(
            "Cleaned Data Preview"
        )

        st.dataframe(
            cleaned_df.head(50),
            use_container_width=True,
            hide_index=True,
            height=450,
        )

        st.divider()

        cleaned_csv = (
            cleaned_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "Download Cleaned CSV",
            data=cleaned_csv,
            file_name=(
                "DataGuard_Cleaned_Data.csv"
            ),
            mime="text/csv",
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "AI Analysis":

    st.write(
        "AI Analysis"
    )

    st.caption(
        "Gemini-powered interpretation of "
        "DataGuard AI findings."
    )

    if st.button(
        "Generate AI Analysis",
        type="primary",
    ):

        with st.spinner(
            "Generating AI analysis..."
        ):

            gemini_result = (
                generate_gemini_analysis(
                    analysis
                )
            )

            st.session_state[
                "gemini_analysis"
            ] = gemini_result

    gemini_result = (
        st.session_state[
            "gemini_analysis"
        ]
    )

    if gemini_result is None:

        st.info(
            "Click 'Generate AI Analysis' "
            "to analyze the current dataset."
        )

    elif (
        gemini_result["status"]
        == "error"
    ):

        st.error(
            gemini_result["message"]
        )

    else:

        st.markdown(
            gemini_result["content"]
        )

    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        st.write(
            "Power BI"
        )

        st.caption(
            "Export cleaned and business-ready "
            "CSV files for Power BI."
        )

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis[
                "sales_column"
            ],
        )

        st.write(
            "Available Exports"
        )

        for filename, content in exports.items():

            st.download_button(
                f"Download {filename}",
                data=content,
                file_name=filename,
                mime="text/csv",
                key=f"download_{filename}",
            )

        st.divider()

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "Download All Power BI Files",
            data=zip_data,
            file_name=(
                "DataGuard_PowerBI_Exports.zip"
            ),
            mime="application/zip",
            type="primary",
        )

        st.info(
            "Import DataGuard_Cleaned_Data.csv "
            "into Power BI as the main dataset. "
            "The summary files can be used for "
            "additional dashboard visuals."
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif page == "AI Analysis":

    st.write("AI Analysis")

    st.caption(
        "Gemini-powered interpretation of "
        "DataGuard AI findings."
    )

    if st.button(
        "Generate AI Analysis",
        type="primary",
    ):
        with st.spinner("Generating AI analysis..."):
            gemini_result = generate_gemini_analysis(analysis)
            st.session_state["gemini_analysis"] = gemini_result

    gemini_result = st.session_state["gemini_analysis"]

    if gemini_result is None:
        st.info(
            "Click 'Generate AI Analysis' "
            "to analyze the current dataset."
        )

    elif gemini_result["status"] == "error":
        st.error(gemini_result["message"])

    else:
        st.markdown(gemini_result["content"])
    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        st.write(
            "Reports"
        )

        st.caption(
            "Generate a portable DataGuard AI "
            "analysis report."
        )

        score = analysis[
            "quality_score"
        ]

        report_lines = [
            "DATAGUARD AI",
            "AI Data Quality & Anomaly Detection Platform",
            "",
            f"Dataset: {st.session_state.file_name}",
            (
                "Generated: "
                f"{analysis['analysis_timestamp']}"
            ),
            "",
            "DATASET PROFILE",
            (
                f"Rows: "
                f"{analysis['rows']:,}"
            ),
            (
                f"Columns: "
                f"{analysis['columns']:,}"
            ),
            (
                f"Numeric Columns: "
                f"{analysis['numeric_count']:,}"
            ),
            (
                f"Categorical Columns: "
                f"{analysis['categorical_count']:,}"
            ),
            (
                f"Date/Time Columns: "
                f"{analysis['datetime_count']:,}"
            ),
            "",
            "QUALITY RESULTS",
            (
                f"Quality Score: "
                f"{score:.2f}/100"
            ),
            (
                f"Missing Cells: "
                f"{analysis['missing_count']:,}"
            ),
            (
                f"Duplicate Rows: "
                f"{analysis['duplicate_count']:,}"
            ),
            (
                f"Invalid Values: "
                f"{analysis['invalid_count']:,}"
            ),
            "",
            "ANOMALY RESULTS",
            (
                f"IQR Outliers: "
                f"{analysis['iqr_outlier_count']:,}"
            ),
            (
                f"ML Anomalies: "
                f"{analysis['ml_anomaly_count']:,}"
            ),
            "",
            "BUSINESS",
            (
                "Sales Column: "
                f"{analysis['sales_column']}"
            ),
            "",
            "DataGuard AI completed "
            "the dataset analysis.",
        ]

        report_text = "\n".join(
            report_lines
        )

        st.text_area(
            "Report",
            value=report_text,
            height=450,
        )

        st.download_button(
            "Download Report",
            data=report_text,
            file_name=(
                "DataGuard_AI_Report.txt"
            ),
            mime="text/plain",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🛡️ DataGuard AI • AI Data Quality & "
    "Anomaly Detection Platform"
)

st.caption(
    "Built with Python • Pandas • "
    "Scikit-learn • Streamlit"
)
