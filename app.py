import io
import os
import zipfile

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
    "dataset_name": "",
    "uploaded_file_type": "",
    "cleaned_df": None,
    "cleaning_applied": False,
    "sensitivity": 5,
    "ai_analysis": None,
    "ai_analysis_signature": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# PREMIUM DARK UI
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(99,102,241,0.08),
                transparent 28%
            ),
            #070b14;
        color: #e5e7eb;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p, label, span, div {
        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background: #0b1020 !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-size: 23px !important;
        font-weight: 800 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.1rem !important;
    }

    [data-testid="stSidebar"]
    [data-testid="stCaptionContainer"] p {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }

    [data-testid="stSidebar"] p strong {
        color: #64748b !important;
        font-size: 11px !important;
        letter-spacing: 1.3px !important;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button *,
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        color: #cbd5e1 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        min-height: 42px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        text-align: left !important;
        padding: 8px 12px !important;
        margin: 3px 0 !important;
        transition: all 0.18s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:hover * {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.09) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 14px 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(34,197,94,0.07) !important;
        border: 1px solid rgba(34,197,94,0.18) !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: #86efac !important;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    .dg-card {
        background:
            linear-gradient(
                145deg,
                rgba(20,27,45,0.96),
                rgba(12,17,30,0.96)
            );
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }

    .dg-card-title {
        color: #f8fafc;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dg-card-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 15px;
    }


    /* ======================================================
       KPI
       ====================================================== */

    .kpi {
        background:
            linear-gradient(
                145deg,
                rgba(18,25,42,0.98),
                rgba(11,16,29,0.98)
            );
        border: 1px solid rgba(148,163,184,0.11);
        border-radius: 15px;
        padding: 18px;
        min-height: 115px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 750;
        line-height: 1.1;
    }

    .kpi-note {
        color: #64748b;
        font-size: 11px;
        margin-top: 8px;
    }


    /* ======================================================
       HEALTH CARDS
       ====================================================== */

    .health-card {
        background:
            linear-gradient(
                145deg,
                rgba(20,27,45,0.96),
                rgba(12,17,30,0.96)
            );
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 14px;
        padding: 16px;
        min-height: 105px;
    }

    .health-label {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .health-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 750;
    }

    .health-note {
        color: #64748b;
        font-size: 11px;
        margin-top: 5px;
    }


    /* ======================================================
       UPLOAD
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(15,23,42,0.45);
        border: 1px dashed rgba(148,163,184,0.25);
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 12px;
        overflow: hidden;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 9px !important;
        font-weight: 600 !important;
    }


    /* ======================================================
       BADGES
       ====================================================== */

    .badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.20);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 600;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .dg-footer {
        text-align: center;
        color: #64748b;
        font-size: 11px;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_uploaded_file(uploaded_file):
    """Read CSV, XLSX or XLS into pandas."""

    name = uploaded_file.name.lower()

    uploaded_file.seek(0)

    if name.endswith(".csv"):

        try:
            return pd.read_csv(uploaded_file)

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

    if name.endswith(".xlsx"):

        return pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

    if name.endswith(".xls"):

        return pd.read_excel(
            uploaded_file,
            engine="xlrd"
        )

    raise ValueError(
        "Unsupported file format."
    )


def numeric_columns(df):

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def categorical_columns(df):

    return df.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns.tolist()


def datetime_columns(df):

    return [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        )
    ]


def invalid_numeric_values(df):

    invalid = 0

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

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

        if numeric_ratio >= 0.80:

            invalid += int(
                converted.isna().sum()
            )

    return invalid


def quality_metrics(df):

    rows = len(df)
    columns = len(df.columns)

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    invalid = invalid_numeric_values(df)

    if rows == 0 or columns == 0:

        score = 0

    else:

        missing_rate = (
            missing / (rows * columns)
        )

        duplicate_rate = (
            duplicates / rows
        )

        invalid_rate = (
            invalid / rows
        )

        score = (
            100
            - missing_rate * 50
            - duplicate_rate * 30
            - invalid_rate * 20
        )

        score = max(
            0,
            min(100, score)
        )

    return {
        "score": round(score, 2),
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def quality_health(df, quality):

    total_cells = max(
        1,
        quality["rows"] * quality["columns"]
    )

    completeness = max(
        0,
        100 - (
            quality["missing"]
            / total_cells
            * 100
        )
    )

    uniqueness = max(
        0,
        100 - (
            quality["duplicates"]
            / max(1, quality["rows"])
            * 100
        )
    )

    validity = max(
        0,
        100 - (
            quality["invalid"]
            / max(1, quality["rows"])
            * 100
        )
    )

    return {
        "Completeness": round(
            completeness,
            2
        ),
        "Uniqueness": round(
            uniqueness,
            2
        ),
        "Validity": round(
            validity,
            2
        ),
    }


def iqr_outlier_mask(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:

        return pd.Series(
            False,
            index=df.index
        )

    row_mask = pd.Series(
        False,
        index=df.index
    )

    for column in numeric.columns:

        series = numeric[column]

        valid = series.dropna()

        if len(valid) < 4:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        column_mask = (
            (series < lower)
            | (series > upper)
        )

        row_mask = (
            row_mask
            | column_mask.fillna(False)
        )

    return row_mask


def iqr_outliers(df):

    return int(
        iqr_outlier_mask(df).sum()
    )


def isolation_forest(
    df,
    sensitivity=5
):

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty or len(df) < 2:

        return (
            len(df),
            0,
            pd.Series(
                False,
                index=df.index
            )
        )

    working = numeric.copy()

    working = working.replace(
        [np.inf, -np.inf],
        np.nan
    )

    working = working.fillna(
        working.median(numeric_only=True)
    )

    working = working.fillna(0)

    contamination = max(
        0.005,
        min(
            0.20,
            sensitivity / 100
        )
    )

    # Prevent contamination from exceeding
    # the safe range for very small datasets.
    max_contamination = (
        max(
            1,
            len(df) - 1
        ) / len(df)
    )

    contamination = min(
        contamination,
        max_contamination
    )

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42
    )

    predictions = model.fit_predict(
        working
    )

    anomaly_mask = predictions == -1

    anomalies = int(
        anomaly_mask.sum()
    )

    normal = int(
        (~anomaly_mask).sum()
    )

    return (
        normal,
        anomalies,
        pd.Series(
            anomaly_mask,
            index=df.index
        )
    )


def find_column(
    df,
    possible_names
):

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        key = str(
            name
        ).strip().lower()

        if key in normalized:

            return normalized[key]

    return None


def clean_dataset(df):

    cleaned = df.copy()

    rows_before = len(
        cleaned
    )

    duplicates_removed = int(
        cleaned.duplicated().sum()
    )

    cleaned = cleaned.drop_duplicates()

    filled_values = 0

    # --------------------------------------------------------
    # Numeric missing values
    # --------------------------------------------------------

    for column in numeric_columns(
        cleaned
    ):

        missing_count = int(
            cleaned[column].isna().sum()
        )

        if missing_count > 0:

            median_value = (
                cleaned[column].median()
            )

            if pd.notna(
                median_value
            ):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median_value)
                )

                filled_values += (
                    missing_count
                )

    # --------------------------------------------------------
    # Categorical missing values
    # --------------------------------------------------------

    for column in categorical_columns(
        cleaned
    ):

        missing_count = int(
            cleaned[column].isna().sum()
        )

        if missing_count > 0:

            mode = (
                cleaned[column]
                .mode()
            )

            if not mode.empty:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(mode.iloc[0])
                )

                filled_values += (
                    missing_count
                )

    # --------------------------------------------------------
    # City standardization
    # --------------------------------------------------------

    city_col = find_column(
        cleaned,
        ["city"]
    )

    city_standardized = 0

    if city_col:

        before = (
            cleaned[city_col]
            .copy()
        )

        cleaned[city_col] = (
            cleaned[city_col]
            .apply(
                lambda value:
                    value.strip().title()
                    if isinstance(
                        value,
                        str
                    )
                    else value
            )
        )

        changed = (
            before.notna()
            & cleaned[city_col].notna()
            & (
                before.astype(str)
                != cleaned[
                    city_col
                ].astype(str)
            )
        )

        city_standardized = int(
            changed.sum()
        )

    rows_after = len(
        cleaned
    )

    return cleaned, {
        "rows_removed":
            rows_before - rows_after,

        "duplicates_removed":
            duplicates_removed,

        "values_filled":
            filled_values,

        "city_standardized":
            city_standardized,
    }


def business_metrics(df):

    metrics = {}

    sales_col = find_column(
        df,
        [
            "sales",
            "revenue",
            "amount"
        ]
    )

    profit_col = find_column(
        df,
        ["profit"]
    )

    quantity_col = find_column(
        df,
        [
            "quantity",
            "qty"
        ]
    )

    if sales_col:

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).dropna()

        if not sales.empty:

            metrics["Total Sales"] = round(
                sales.sum(),
                2
            )

            metrics["Average Sale"] = round(
                sales.mean(),
                2
            )

            metrics["Highest Sale"] = round(
                sales.max(),
                2
            )

    if profit_col and sales_col:

        profit = pd.to_numeric(
            df[profit_col],
            errors="coerce"
        )

        sales = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        )

        sales_total = sales.sum()

        if sales_total != 0:

            metrics["Profit Margin"] = round(
                (
                    profit.sum()
                    / sales_total
                ) * 100,
                2
            )

    if quantity_col:

        quantity = pd.to_numeric(
            df[quantity_col],
            errors="coerce"
        ).dropna()

        if not quantity.empty:

            metrics["Total Quantity"] = round(
                quantity.sum(),
                2
            )

    return metrics


def create_excel(df):

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

    output.seek(0)

    return output


def create_powerbi_dataset(df):

    export_df = df.copy()

    _, _, anomaly_mask = (
        isolation_forest(
            export_df,
            st.session_state.sensitivity
        )
    )

    export_df[
        "Is_Anomaly"
    ] = anomaly_mask.values

    return export_df


def create_report_excel(df):

    quality = quality_metrics(
        df
    )

    outliers = iqr_outliers(
        df
    )

    normal, anomalies, _ = (
        isolation_forest(
            df,
            st.session_state.sensitivity
        )
    )

    business = business_metrics(
        df
    )

    anomaly_rate = (
        anomalies / len(df) * 100
        if len(df)
        else 0
    )

    report = pd.DataFrame(
        {
            "Metric": [
                "Dataset Rows",
                "Dataset Columns",
                "DataGuard Quality Score",
                "Missing Values",
                "Duplicate Rows",
                "Invalid Values",
                "IQR Outlier Records",
                "Normal Records",
                "ML Anomalies",
                "Anomaly Rate",
            ],
            "Value": [
                quality["rows"],
                quality["columns"],
                quality["score"],
                quality["missing"],
                quality["duplicates"],
                quality["invalid"],
                outliers,
                normal,
                anomalies,
                round(
                    anomaly_rate,
                    2
                ),
            ],
        }
    )

    business_df = pd.DataFrame(
        list(
            business.items()
        ),
        columns=[
            "Metric",
            "Value"
        ]
    )

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        report.to_excel(
            writer,
            index=False,
            sheet_name="Quality Report"
        )

        business_df.to_excel(
            writer,
            index=False,
            sheet_name="Business Metrics"
        )

        df.to_excel(
            writer,
            index=False,
            sheet_name="Dataset"
        )

    output.seek(0)

    return output


def create_zip(df):

    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED
    ) as z:

        cleaned = (
            st.session_state.cleaned_df
        )

        export_df = (
            cleaned
            if cleaned is not None
            else df
        )

        z.writestr(
            "dataset.csv",
            export_df.to_csv(
                index=False
            )
        )

        powerbi_df = (
            create_powerbi_dataset(
                export_df
            )
        )

        z.writestr(
            "powerbi_dataset.csv",
            powerbi_df.to_csv(
                index=False
            )
        )

        report = (
            create_report_excel(
                export_df
            )
        )

        z.writestr(
            "data_quality_report.xlsx",
            report.getvalue()
        )

    output.seek(0)

    return output


def generate_gemini_analysis(
    dataset_name,
    df,
    quality,
    outliers,
    normal,
    anomalies,
    sensitivity,
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "### Gemini API Configuration\n\n"
            "Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` "
            "to Streamlit Secrets."
        )

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        business = business_metrics(
            df
        )

        if business:

            business_text = "\n".join(
                f"- {key}: {value}"
                for key, value
                in business.items()
            )

        else:

            business_text = (
                "No recognized business metrics "
                "were detected in the dataset."
            )

        model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.6-flash"
        )

        prompt = f"""
You are a senior data quality and business analytics consultant.

Analyze ONLY the measured information supplied below.

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
IQR outlier records: {outliers}

MACHINE LEARNING ANOMALIES
Normal records: {normal}
Anomalies: {anomalies}
Sensitivity: {sensitivity}%

BUSINESS METRICS
{business_text}

IMPORTANT RULES

1. Use only the supplied measured numbers.
2. Do not invent numbers.
3. Do not change any measured number.
4. IQR outliers are statistical signals.
5. Isolation Forest anomalies are screening signals.
6. An anomaly does NOT automatically mean an incorrect record.
7. Clearly distinguish confirmed data-quality problems from statistical signals.
8. Do not call missing values "missing records".
9. Do not claim a value is wrong unless the supplied data confirms it.
10. Do not invent business context.
11. If no business metrics are detected, explicitly say so.
12. Give practical recommendations.
13. Explain possible root causes as hypotheses.
14. Keep the analysis professional and suitable for a data analytics portfolio.

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

        interaction = (
            client.interactions.create(
                model=model_name,
                input=prompt
            )
        )

        return interaction.output_text

    except Exception as error:

        return (
            "### Gemini Analysis Error\n\n"
            "Gemini analysis could not be generated.\n\n"
            f"`{error}`"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🛡️ DataGuard AI"
    )

    st.caption(
        "AI Data Quality Platform"
    )

    st.divider()

    if st.button(
        "⌂  Dashboard",
        key="nav_dashboard",
        use_container_width=True
    ):

        st.session_state.page = (
            "Dashboard"
        )

        st.rerun()

    st.markdown(
        "**WORKSPACE**"
    )

    navigation = [
        ("01", "Upload Data"),
        ("02", "Data Profile"),
        ("03", "Quality Checks"),
        ("04", "Anomaly Detection"),
        ("05", "Data Cleaning"),
        ("06", "Analytics"),
        ("07", "Power BI"),
        ("08", "AI Analysis"),
    ]

    for number, page_name in navigation:

        label = (
            f"{number}  {page_name}"
        )

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True
        ):

            st.session_state.page = (
                page_name
            )

            st.rerun()

    st.divider()

    st.markdown(
        "**SYSTEM**"
    )

    if st.button(
        "⚙️  Settings",
        key="settings",
        use_container_width=True
    ):

        st.session_state.page = (
            "Settings"
        )

        st.rerun()

    if st.button(
        "❓  Help",
        key="help",
        use_container_width=True
    ):

        st.session_state.page = (
            "Help"
        )

        st.rerun()

    st.divider()

    st.success(
        "● All systems operational"
    )


# ============================================================
# OPTIONAL DEMO DATA
# ============================================================

if (
    st.session_state.df is None
    and os.path.exists(
        "sales_data_raw.csv"
    )
):

    st.info(
        "A demo dataset is available. "
        "Use the Upload Data page to load your own dataset."
    )


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

page_to_step = {
    "Upload Data": 0,
    "Data Profile": 1,
    "Quality Checks": 2,
    "Anomaly Detection": 3,
    "Data Cleaning": 4,
    "Analytics": 5,
    "Power BI": 6,
    "AI Analysis": 7,
}

current_step = page_to_step.get(
    st.session_state.page,
    0
)

if (
    st.session_state.page
    != "Dashboard"
):

    cols = st.columns(8)

    for i, step in enumerate(
        workflow
    ):

        with cols[i]:

            if i < current_step:

                st.success(
                    f"✓ {step}"
                )

            elif i == current_step:

                st.info(
                    f"● {step}"
                )

            else:

                st.caption(
                    step
                )


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.df


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.title(
        "Dashboard",
        anchor=False
    )

    if df is None:

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Welcome to DataGuard AI
            </div>

            <div class="dg-card-subtitle">
            Upload a dataset to begin automated data quality,
            anomaly detection and analytics.
            </div>

            <span class="badge">
            AI DATA QUALITY PLATFORM
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        quality = quality_metrics(
            df
        )

        normal, anomalies, _ = (
            isolation_forest(
                df,
                st.session_state.sensitivity
            )
        )

        health = quality_health(
            df,
            quality
        )

        st.caption(
            f"Dataset: "
            f"{st.session_state.dataset_name}"
        )

        # ----------------------------------------------------
        # MAIN KPIs
        # ----------------------------------------------------

        c1, c2, c3, c4, c5, c6 = (
            st.columns(6)
        )

        metrics = [
            (
                c1,
                "Data Quality Score",
                f'{quality["score"]:.2f}%',
                "DataGuard score"
            ),
            (
                c2,
                "Total Records",
                f'{quality["rows"]:,}',
                "Rows"
            ),
            (
                c3,
                "Columns",
                f'{quality["columns"]:,}',
                "Features"
            ),
            (
                c4,
                "Missing Values",
                f'{quality["missing"]:,}',
                "Requires review"
            ),
            (
                c5,
                "Duplicates",
                f'{quality["duplicates"]:,}',
                "Duplicate rows"
            ),
            (
                c6,
                "Anomalies",
                f'{anomalies:,}',
                "ML screening"
            ),
        ]

        for col, label, value, note in metrics:

            with col:

                st.markdown(
                    f"""
                    <div class="kpi">

                    <div class="kpi-label">
                    {label}
                    </div>

                    <div class="kpi-value">
                    {value}
                    </div>

                    <div class="kpi-note">
                    {note}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        # ----------------------------------------------------
        # DATASET HEALTH
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Dataset Health
            </div>

            <div class="dg-card-subtitle">
            Current quality indicators
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        h1, h2, h3 = st.columns(3)

        health_values = [
            (
                h1,
                "Completeness",
                health["Completeness"],
                "Based on missing values"
            ),
            (
                h2,
                "Uniqueness",
                health["Uniqueness"],
                "Based on duplicate rows"
            ),
            (
                h3,
                "Validity",
                health["Validity"],
                "Based on detected invalid values"
            ),
        ]

        for (
            col,
            label,
            value,
            note
        ) in health_values:

            with col:

                st.markdown(
                    f"""
                    <div class="health-card">

                    <div class="health-label">
                    {label}
                    </div>

                    <div class="health-value">
                    {value:.2f}%
                    </div>

                    <div class="health-note">
                    {note}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.write("")

        # ----------------------------------------------------
        # ANOMALY DETECTION
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Anomaly Detection
            </div>

            <div class="dg-card-subtitle">
            Isolation Forest screening
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Normal",
                f"{normal:,}"
            )

        with b:

            st.metric(
                "Anomalies",
                f"{anomalies:,}"
            )

        with c:

            rate = (
                anomalies
                / len(df)
                * 100
                if len(df)
                else 0
            )

            st.metric(
                "Rate",
                f"{rate:.2f}%"
            )

        st.caption(
            "Anomalies are screening signals and should "
            "be reviewed before being treated as incorrect data."
        )


# ============================================================
# UPLOAD DATA
# ============================================================

elif st.session_state.page == "Upload Data":

    st.title(
        "Upload Data",
        anchor=False
    )

    st.caption(
        "Upload CSV, XLSX or XLS files for data quality analysis."
    )

    st.write("")

    st.markdown(
        """
        <div class="dg-card">

        <div class="dg-card-title">
        Choose your dataset
        </div>

        <div class="dg-card-subtitle">
        Supported formats: CSV, XLSX, XLS • Maximum size: 50 MB
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drop your dataset here or browse files",
        type=[
            "csv",
            "xlsx",
            "xls"
        ],
        key="dataset_uploader"
    )

    if uploaded_file is not None:

        try:

            loaded_df = read_uploaded_file(
                uploaded_file
            )

            if loaded_df.empty:

                raise ValueError(
                    "The uploaded dataset contains no rows."
                )

            if len(loaded_df.columns) == 0:

                raise ValueError(
                    "The uploaded dataset contains no columns."
                )

            st.session_state.df = (
                loaded_df
            )

            st.session_state.dataset_name = (
                uploaded_file.name
            )

            st.session_state.uploaded_file_type = (
                uploaded_file.name
                .split(".")[-1]
                .upper()
            )

            st.session_state.cleaned_df = None
            st.session_state.cleaning_applied = False
            st.session_state.ai_analysis = None
            st.session_state.ai_analysis_signature = None

            st.success(
                f"Successfully loaded "
                f"{uploaded_file.name}"
            )

        except Exception as error:

            st.error(
                f"Could not read the file: {error}"
            )

    df = st.session_state.df

    if df is not None:

        st.write("")

        st.markdown(
            f"""
            <div class="dg-card">

            <div class="dg-card-title">
            {st.session_state.dataset_name}
            </div>

            <div class="dg-card-subtitle">
            Dataset successfully loaded and ready for analysis.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Rows",
                f"{len(df):,}"
            )

        with c2:

            st.metric(
                "Columns",
                f"{len(df.columns):,}"
            )

        with c3:

            st.metric(
                "File Type",
                st.session_state.uploaded_file_type
            )

        with c4:

            memory_mb = (
                df.memory_usage(
                    deep=True
                ).sum()
                / 1024**2
            )

            st.metric(
                "Memory",
                f"{memory_mb:.2f} MB"
            )

        st.write("")

        with st.expander(
            "Preview dataset",
            expanded=False
        ):

            st.dataframe(
                df.head(10),
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "Dataset is ready. Continue to Data Profile "
            "to inspect columns, data types and distributions."
        )


# ============================================================
# DATA PROFILE
# ============================================================

elif st.session_state.page == "Data Profile":

    st.title(
        "Data Profile",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        numeric_count = len(
            numeric_columns(df)
        )

        categorical_count = len(
            categorical_columns(df)
        )

        datetime_count = len(
            datetime_columns(df)
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Numeric Columns",
                numeric_count
            )

        with c2:

            st.metric(
                "Categorical Columns",
                categorical_count
            )

        with c3:

            st.metric(
                "Date/Time Columns",
                datetime_count
            )

        st.write("")

        profile_rows = []

        for column in df.columns:

            missing = int(
                df[column].isna().sum()
            )

            unique = int(
                df[column].nunique(
                    dropna=True
                )
            )

            total = len(df)

            missing_pct = (
                missing / total * 100
                if total
                else 0
            )

            unique_pct = (
                unique / total * 100
                if total
                else 0
            )

            profile_rows.append(
                {
                    "Column": column,
                    "Data Type": str(
                        df[column].dtype
                    ),
                    "Non-Null": int(
                        df[column].notna().sum()
                    ),
                    "Missing": missing,
                    "Missing %": round(
                        missing_pct,
                        2
                    ),
                    "Unique": unique,
                    "Unique %": round(
                        unique_pct,
                        2
                    ),
                }
            )

        profile = pd.DataFrame(
            profile_rows
        )

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# QUALITY CHECKS
# ============================================================

elif st.session_state.page == "Quality Checks":

    st.title(
        "Quality Checks",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        quality = quality_metrics(
            df
        )

        health = quality_health(
            df,
            quality
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Quality Score",
                f'{quality["score"]:.2f}%'
            )

        with c2:

            st.metric(
                "Missing",
                f'{quality["missing"]:,}'
            )

        with c3:

            st.metric(
                "Duplicates",
                f'{quality["duplicates"]:,}'
            )

        with c4:

            st.metric(
                "Invalid",
                f'{quality["invalid"]:,}'
            )

        st.write("")

        st.subheader(
            "Quality Dimensions"
        )

        q1, q2, q3 = st.columns(3)

        with q1:

            st.metric(
                "Completeness",
                f'{health["Completeness"]:.2f}%'
            )

        with q2:

            st.metric(
                "Uniqueness",
                f'{health["Uniqueness"]:.2f}%'
            )

        with q3:

            st.metric(
                "Validity",
                f'{health["Validity"]:.2f}%'
            )

        st.write("")

        st.subheader(
            "Detailed Checks"
        )

        quality_table = pd.DataFrame(
            {
                "Check": [
                    "Missing Values",
                    "Duplicate Rows",
                    "Invalid Values",
                    "IQR Outlier Records",
                ],
                "Result": [
                    quality["missing"],
                    quality["duplicates"],
                    quality["invalid"],
                    iqr_outliers(df),
                ],
            }
        )

        st.dataframe(
            quality_table,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

elif st.session_state.page == "Anomaly Detection":

    st.title(
        "Anomaly Detection",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        sensitivity = st.slider(
            "Isolation Forest sensitivity",
            min_value=1,
            max_value=20,
            value=st.session_state.sensitivity
        )

        if sensitivity != (
            st.session_state.sensitivity
        ):

            st.session_state.sensitivity = (
                sensitivity
            )

            st.session_state.ai_analysis = None
            st.session_state.ai_analysis_signature = None

            st.rerun()

        normal, anomalies, mask = (
            isolation_forest(
                df,
                sensitivity
            )
        )

        rate = (
            anomalies
            / len(df)
            * 100
            if len(df)
            else 0
        )

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            st.metric(
                "Normal",
                f"{normal:,}"
            )

        with c2:

            st.metric(
                "Anomalies",
                f"{anomalies:,}"
            )

        with c3:

            st.metric(
                "Anomaly Rate",
                f"{rate:.2f}%"
            )

        st.info(
            "Isolation Forest identifies unusual patterns. "
            "These are screening signals, not automatic proof "
            "of bad data."
        )

        anomaly_df = df.loc[
            mask
        ].copy()

        anomaly_df[
            "Is_Anomaly"
        ] = True

        st.subheader(
            "Detected Anomalies"
        )

        if anomaly_df.empty:

            st.success(
                "No anomalies detected at the current sensitivity."
            )

        else:

            st.dataframe(
                anomaly_df.head(100),
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# DATA CLEANING
# ============================================================

elif st.session_state.page == "Data Cleaning":

    st.title(
        "Data Cleaning",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        cleaned, stats = (
            clean_dataset(df)
        )

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        with c1:

            st.metric(
                "Rows Removed",
                f'{stats["rows_removed"]:,}'
            )

        with c2:

            st.metric(
                "Values Filled",
                f'{stats["values_filled"]:,}'
            )

        with c3:

            st.metric(
                "Duplicates Removed",
                f'{stats["duplicates_removed"]:,}'
            )

        with c4:

            st.metric(
                "City Standardized",
                f'{stats["city_standardized"]:,}'
            )

        st.write("")

        if st.button(
            "Apply Cleaning",
            type="primary"
        ):

            st.session_state.cleaned_df = (
                cleaned
            )

            st.session_state.cleaning_applied = (
                True
            )

            st.success(
                "Cleaning applied successfully."
            )

        if (
            st.session_state.cleaned_df
            is not None
        ):

            cleaned_df = (
                st.session_state.cleaned_df
            )

            st.subheader(
                "Cleaned Dataset Preview"
            )

            st.dataframe(
                cleaned_df.head(20),
                use_container_width=True,
                hide_index=True
            )

            before_quality = (
                quality_metrics(df)
            )

            after_quality = (
                quality_metrics(
                    cleaned_df
                )
            )

            st.write("")

            st.subheader(
                "Quality Improvement"
            )

            q1, q2 = st.columns(2)

            with q1:

                st.metric(
                    "Before Cleaning",
                    f'{before_quality["score"]:.2f}%'
                )

            with q2:

                st.metric(
                    "After Cleaning",
                    f'{after_quality["score"]:.2f}%'
                )

            cleaned_file = create_excel(
                cleaned_df
            )

            st.download_button(
                "Download Cleaned Excel",
                data=cleaned_file,
                file_name=(
                    "dataguard_cleaned_data.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


# ============================================================
# ANALYTICS
# ============================================================

elif st.session_state.page == "Analytics":

    st.title(
        "Analytics",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        metrics = business_metrics(
            df
        )

        if metrics:

            metric_columns = st.columns(
                min(
                    len(metrics),
                    4
                )
            )

            for col, (
                label,
                value
            ) in zip(
                metric_columns,
                metrics.items()
            ):

                with col:

                    if "Margin" in label:

                        st.metric(
                            label,
                            f"{value:.2f}%"
                        )

                    else:

                        st.metric(
                            label,
                            f"{value:,.2f}"
                        )

        else:

            st.info(
                "No recognized business metrics were detected. "
                "Showing statistical analysis of numeric columns."
            )

        st.write("")

        numeric = numeric_columns(
            df
        )

        if numeric:

            selected_column = (
                st.selectbox(
                    "Select numeric column",
                    numeric
                )
            )

            series = pd.to_numeric(
                df[selected_column],
                errors="coerce"
            ).dropna()

            if not series.empty:

                mean_value = (
                    series.mean()
                )

                median_value = (
                    series.median()
                )

                min_value = (
                    series.min()
                )

                max_value = (
                    series.max()
                )

                c1, c2, c3, c4 = (
                    st.columns(4)
                )

                with c1:

                    st.metric(
                        "Mean",
                        f"{mean_value:,.2f}"
                    )

                with c2:

                    st.metric(
                        "Median",
                        f"{median_value:,.2f}"
                    )

                with c3:

                    st.metric(
                        "Minimum",
                        f"{min_value:,.2f}"
                    )

                with c4:

                    st.metric(
                        "Maximum",
                        f"{max_value:,.2f}"
                    )

                st.write("")

                st.subheader(
                    f"{selected_column} Distribution"
                )

                if series.nunique() > 1:

                    bin_count = min(
                        10,
                        max(
                            2,
                            series.nunique()
                        )
                    )

                    histogram = pd.cut(
                        series,
                        bins=bin_count,
                        include_lowest=True
                    ).value_counts(
                        sort=False
                    )

                    histogram.index = (
                        histogram.index
                        .astype(str)
                    )

                    st.bar_chart(
                        histogram,
                        height=350
                    )

                else:

                    st.info(
                        "This column contains only one "
                        "unique numeric value."
                    )

            else:

                st.info(
                    "No valid numeric values are available "
                    "for this column."
                )

        else:

            st.info(
                "No numeric columns available for analytics."
            )


# ============================================================
# POWER BI
# ============================================================

elif st.session_state.page == "Power BI":

    st.title(
        "Power BI Export",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        if (
            st.session_state.cleaned_df
            is not None
        ):

            export_source = (
                st.session_state.cleaned_df
            )

            source_label = (
                "Cleaned dataset"
            )

        else:

            export_source = df

            source_label = (
                "Original dataset"
            )

        powerbi_df = (
            create_powerbi_dataset(
                export_source
            )
        )

        st.markdown(
            f"""
            <div class="dg-card">

            <div class="dg-card-title">
            Power BI Ready Dataset
            </div>

            <div class="dg-card-subtitle">
            Export a prepared dataset with anomaly flags
            for Power BI reporting.
            </div>

            <span class="badge">
            {source_label.upper()}
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        c1, c2, c3 = (
            st.columns(3)
        )

        with c1:

            st.metric(
                "Rows",
                f"{len(powerbi_df):,}"
            )

        with c2:

            st.metric(
                "Columns",
                f"{len(powerbi_df.columns):,}"
            )

        with c3:

            anomaly_count = int(
                powerbi_df[
                    "Is_Anomaly"
                ].sum()
            )

            st.metric(
                "Anomaly Flags",
                f"{anomaly_count:,}"
            )

        st.write("")

        st.subheader(
            "Power BI Dataset Preview"
        )

        st.dataframe(
            powerbi_df.head(10),
            use_container_width=True,
            hide_index=True
        )

        csv_data = (
            powerbi_df.to_csv(
                index=False
            )
        )

        st.download_button(
            "Download Power BI Dataset",
            data=csv_data,
            file_name=(
                "dataguard_powerbi_dataset.csv"
            ),
            mime="text/csv",
            type="primary"
        )

        report_file = (
            create_report_excel(
                export_source
            )
        )

        st.download_button(
            "Download Quality Report",
            data=report_file,
            file_name=(
                "dataguard_quality_report.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

        st.info(
            "Import the CSV into Power BI Desktop. "
            "The Is_Anomaly column can be used for filtering, "
            "conditional formatting and anomaly reporting."
        )


# ============================================================
# AI ANALYSIS
# ============================================================

elif st.session_state.page == "AI Analysis":

    st.title(
        "AI Analysis",
        anchor=False
    )

    if df is None:

        st.warning(
            "Upload a dataset first."
        )

    else:

        quality = quality_metrics(
            df
        )

        outliers = iqr_outliers(
            df
        )

        normal, anomalies, _ = (
            isolation_forest(
                df,
                st.session_state.sensitivity
            )
        )

        current_signature = (
            st.session_state.dataset_name,
            len(df),
            len(df.columns),
            quality["score"],
            quality["missing"],
            quality["duplicates"],
            quality["invalid"],
            outliers,
            normal,
            anomalies,
            st.session_state.sensitivity,
        )

        st.markdown(
            """
            <div class="dg-card">

            <div class="dg-card-title">
            Gemini Data Intelligence
            </div>

            <div class="dg-card-subtitle">
            AI-generated interpretation of measured data quality
            and business signals.
            </div>

            <span class="badge">
            AI ANALYSIS
            </span>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "Generate AI Analysis",
            type="primary"
        ):

            with st.spinner(
                "Gemini is analyzing your dataset..."
            ):

                st.session_state.ai_analysis = (
                    generate_gemini_analysis(
                        st.session_state.dataset_name,
                        df,
                        quality,
                        outliers,
                        normal,
                        anomalies,
                        st.session_state.sensitivity
                    )
                )

                st.session_state.ai_analysis_signature = (
                    current_signature
                )

        if (
            st.session_state.ai_analysis
            and
            st.session_state.ai_analysis_signature
            == current_signature
        ):

            st.markdown(
                st.session_state.ai_analysis
            )

            st.download_button(
                "Download AI Analysis",
                data=st.session_state.ai_analysis,
                file_name="dataguard_ai_analysis.md",
                mime="text/markdown"
            )

        elif st.session_state.ai_analysis:

            st.info(
                "Dataset metrics have changed. "
                "Generate a new AI analysis to refresh the results."
            )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.title(
        "Settings",
        anchor=False
    )

    st.subheader(
        "Detection Settings"
    )

    sensitivity = st.slider(
        "Isolation Forest sensitivity",
        1,
        20,
        st.session_state.sensitivity
    )

    if sensitivity != (
        st.session_state.sensitivity
    ):

        st.session_state.sensitivity = (
            sensitivity
        )

        st.session_state.ai_analysis = None
        st.session_state.ai_analysis_signature = None

        st.rerun()

    st.caption(
        "Higher sensitivity identifies more records "
        "as potential anomalies."
    )

    st.subheader(
        "Quality Score"
    )

    st.write(
        "DataGuard Quality Score is a custom weighted score "
        "based on missing, duplicate and invalid-value rates."
    )

    st.subheader(
        "Dataset"
    )

    if df is not None:

        st.write(
            f"Current dataset: "
            f"**{st.session_state.dataset_name}**"
        )

    else:

        st.write(
            "No dataset loaded."
        )


# ============================================================
# HELP
# ============================================================

elif st.session_state.page == "Help":

    st.title(
        "Help",
        anchor=False
    )

    st.markdown(
        """
        ### How DataGuard AI works

        **01 Upload**

        Upload your CSV or Excel dataset.

        **02 Profile**

        Inspect columns, data types, missing values
        and uniqueness.

        **03 Quality**

        Identify missing values, duplicates,
        invalid values and IQR-based outlier records.

        **04 Anomalies**

        Use Isolation Forest to identify unusual records.

        **05 Clean**

        Remove duplicates and fill missing values
        using median/mode-based imputation.

        **06 Analytics**

        Explore business metrics and statistical distributions.

        **07 Power BI**

        Export the prepared dataset with an
        `Is_Anomaly` flag for Power BI reporting.

        **08 AI**

        Generate an AI-assisted data quality
        and business analysis.

        ### Important

        ML anomalies are screening signals,
        not automatic proof that a record is incorrect.

        Always validate anomalies using business context.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="dg-footer">

    🛡️ DataGuard AI
    &nbsp;•&nbsp;
    Portfolio Edition
    &nbsp;•&nbsp;
    Python
    &nbsp;•&nbsp;
    Pandas
    &nbsp;•&nbsp;
    Scikit-learn
    &nbsp;•&nbsp;
    Streamlit
    &nbsp;•&nbsp;
    Gemini

    </div>
    """,
    unsafe_allow_html=True
)
