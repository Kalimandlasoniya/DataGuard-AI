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
# SAFE HTML RENDERER
# ============================================================

def render_html(content):
    """
    Render custom HTML safely.
    IMPORTANT:
    No indentation before the HTML is sent to Streamlit.
    """
    st.markdown(
        content.strip(),
        unsafe_allow_html=True
    )


# ============================================================
# CSS
# ============================================================

render_html("""
<style>

.stApp {
    background: #f6f8fc;
    color: #0f172a;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] > div {
    background: #0b1220;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.sidebar-brand {
    padding: 8px 4px 24px 4px;
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 11px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.brand-name {
    color: white;
    font-size: 18px;
    font-weight: 800;
}

.brand-caption {
    color: #94a3b8 !important;
    font-size: 10px;
    margin-top: 2px;
}

.sidebar-section {
    color: #64748b !important;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 20px;
    margin-bottom: 7px;
}

.pipeline-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px;
}

.pipeline-step {
    color: #94a3b8 !important;
    font-size: 10px;
    padding: 5px 0;
}

.pipeline-number {
    color: #475569 !important;
    font-weight: 800;
    display: inline-block;
    width: 25px;
}

.sidebar-note {
    color: #64748b !important;
    font-size: 10px;
    line-height: 1.5;
}

.sidebar-footer {
    color: #475569 !important;
    font-size: 9px;
    line-height: 1.6;
    margin-top: 25px;
}

section[data-testid="stSidebar"]
div[data-testid="stRadio"] label {
    background: transparent;
    border-radius: 9px;
    padding: 7px 9px;
}

/* ================= HEADER ================= */

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 24px;
}

.eyebrow {
    color: #64748b;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.page-title {
    color: #0f172a;
    font-size: 32px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-top: 4px;
}

.page-subtitle {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.engine-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 9px;
    font-weight: 800;
}

.engine-dot {
    width: 7px;
    height: 7px;
    background: #10b981;
    border-radius: 50%;
}

/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554,
        #312e81
    );
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 22px;
    box-shadow: 0 18px 45px rgba(15,23,42,.14);
}

.hero-label {
    color: #93c5fd;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1.2px;
}

.hero-title {
    color: white;
    font-size: 29px;
    font-weight: 850;
    margin-top: 7px;
}

.hero-text {
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.6;
    max-width: 700px;
    margin-top: 7px;
}

.hero-tags {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin-top: 17px;
}

.hero-tag {
    color: #dbeafe;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 9px;
    font-weight: 700;
}

/* ================= UPLOAD ================= */

.upload-box {
    background: white;
    border: 1px dashed #94a3b8;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    text-align: center;
}

.upload-icon {
    width: 45px;
    height: 45px;
    border-radius: 12px;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
    font-size: 20px;
}

.upload-title {
    color: #111827;
    font-size: 15px;
    font-weight: 800;
    margin-top: 10px;
}

.upload-text {
    color: #64748b;
    font-size: 11px;
    margin-top: 4px;
}

.upload-meta {
    color: #94a3b8;
    font-size: 9px;
    margin-top: 7px;
}

/* ================= SECTIONS ================= */

.section-title {
    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    margin-top: 24px;
}

.section-subtitle {
    color: #64748b;
    font-size: 11px;
    margin-top: 3px;
    margin-bottom: 14px;
}

/* ================= CARDS ================= */

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(15,23,42,.035);
}

.card-title {
    color: #111827;
    font-size: 13px;
    font-weight: 800;
}

.card-subtitle {
    color: #94a3b8;
    font-size: 9px;
    margin-top: 3px;
}

/* ================= KPI ================= */

.kpi-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    min-height: 105px;
}

.kpi-label {
    color: #64748b;
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .6px;
}

.kpi-value {
    color: #0f172a;
    font-size: 25px;
    font-weight: 850;
    margin-top: 8px;
}

.kpi-caption {
    color: #94a3b8;
    font-size: 9px;
    margin-top: 2px;
}

/* ================= SCORE ================= */

.score-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 22px;
    min-height: 205px;
}

.score-label {
    color: #64748b;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: 1px;
}

.score-number {
    color: #0f172a;
    font-size: 50px;
    font-weight: 900;
    letter-spacing: -2px;
    margin-top: 8px;
}

.score-number-small {
    color: #94a3b8;
    font-size: 15px;
    font-weight: 600;
}

.score-badge {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 9px;
    font-weight: 800;
    margin-top: 5px;
}

.badge-good {
    background: #dcfce7;
    color: #166534;
}

.badge-warning {
    background: #fef3c7;
    color: #92400e;
}

.badge-danger {
    background: #fee2e2;
    color: #991b1b;
}

.score-track {
    width: 100%;
    height: 7px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 17px;
}

.score-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
}

/* ================= FEATURE ================= */

.feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 18px;
    min-height: 145px;
}

.feature-icon {
    width: 35px;
    height: 35px;
    border-radius: 10px;
    background: #eff6ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    margin-bottom: 11px;
}

.feature-title {
    color: #111827;
    font-size: 13px;
    font-weight: 800;
}

.feature-text {
    color: #64748b;
    font-size: 10px;
    line-height: 1.55;
    margin-top: 5px;
}

/* ================= DATASET ================= */

.dataset-strip {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 13px;
    padding: 12px 15px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.dataset-name {
    color: #111827;
    font-size: 12px;
    font-weight: 800;
}

.dataset-meta {
    color: #64748b;
    font-size: 10px;
    margin-top: 3px;
}

/* ================= NOTICES ================= */

.notice {
    border-radius: 10px;
    padding: 11px 13px;
    font-size: 10px;
    line-height: 1.55;
    margin: 10px 0;
}

.notice-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
}

.notice-green {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857;
}

.notice-yellow {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #92400e;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 9px;
    margin-top: 45px;
    padding-top: 18px;
    border-top: 1px solid #e5e7eb;
}

</style>
""")


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
# CITY MAPPING
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
# FUNCTIONS
# ============================================================

def safe_to_datetime(series):

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

    result = []

    tokens = [
        "date",
        "time",
        "datetime",
        "timestamp",
        "created",
        "updated",
        "month",
        "year"
    ]

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):
            result.append(column)
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

        if any(
            token in name
            for token in tokens
        ):
            if ratio >= .50:
                result.append(column)

        elif ratio >= .95:
            result.append(column)

    return list(dict.fromkeys(result))


def detect_identifier_columns(
    df,
    datetime_columns=None
):

    if datetime_columns is None:
        datetime_columns = []

    result = []

    tokens = [
        "_id",
        "id_",
        "identifier",
        "customerid",
        "customer_id",
        "orderid",
        "order_id",
        "productid",
        "product_id"
    ]

    for column in df.columns:

        if column in datetime_columns:
            continue

        name = str(column).lower()

        if any(
            token in name
            for token in tokens
        ):
            result.append(column)
            continue

        if len(df) > 0:

            unique_ratio = (
                df[column].nunique(
                    dropna=True
                ) / len(df)
            )

            if unique_ratio >= .98:
                result.append(column)

    return list(dict.fromkeys(result))


def profile_dataset(df):

    return pd.DataFrame({
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
            int(
                df[c].nunique(
                    dropna=True
                )
            )
            for c in df.columns
        ]
    })


def detect_invalid_values(df):

    details = []
    total_invalid = 0

    tokens = [
        "age",
        "quantity",
        "qty",
        "sales",
        "revenue",
        "amount",
        "price",
        "profit",
        "discount",
        "salary"
    ]

    for column in df.columns:

        name = str(column).lower()

        if not any(
            token in name
            for token in tokens
        ):
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        failures = (
            numeric.isna()
            & df[column].notna()
        )

        negatives = (
            numeric < 0
        )

        count = int(
            failures.sum()
            + negatives.sum()
        )

        if count > 0:

            total_invalid += count

            details.append({
                "Column": column,
                "Invalid Values": count
            })

    return (
        total_invalid,
        pd.DataFrame(details)
    )


def standardize_city_column(df):

    result = df.copy()
    total_changes = 0

    city_columns = [
        c
        for c in result.columns
        if "city" in str(c).lower()
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
            != standardized.fillna("")
        )

        total_changes += int(
            changes.sum()
        )

        result[column] = standardized

    return (
        result,
        total_changes
    )


def detect_iqr_outliers(df):

    details = []
    total = 0

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(.25)
        q3 = series.quantile(.75)

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

            total += count

            details.append({
                "Column": column,
                "Q1": q1,
                "Q3": q3,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outliers": count
            })

    return (
        total,
        pd.DataFrame(details)
    )


def detect_ml_anomalies(
    df,
    contamination_pct
):

    numeric_df = (
        df.select_dtypes(
            include=np.number
        ).copy()
    )

    if numeric_df.shape[1] == 0:

        return (
            pd.Series(
                False,
                index=df.index
            ),
            0
        )

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
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
                index=df.index
            ),
            0
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination_pct / 100,
        random_state=42
    )

    predictions = model.fit_predict(
        numeric_df
    )

    mask = predictions == -1

    return (
        pd.Series(
            mask,
            index=df.index
        ),
        int(mask.sum())
    )


def build_quality_score(
    total_cells,
    missing_count,
    duplicate_count,
    invalid_count
):

    if total_cells <= 0:
        return 100

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
        missing_penalty * .50
        + duplicate_penalty * .30
        + invalid_penalty * .20
    )

    return round(
        max(0, min(100, score)),
        2
    )


def clean_dataset(df):

    cleaned = df.copy()

    original_rows = len(cleaned)

    cleaned = cleaned.drop_duplicates()

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
        cleaned.select_dtypes(
            include=np.number
        ).columns
    )

    for column in numeric_columns:

        missing = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing > 0:

            median = cleaned[
                column
            ].median()

            if pd.notna(median):

                cleaned[column] = (
                    cleaned[column]
                    .fillna(median)
                )

                values_filled += missing

    categorical_columns = (
        cleaned.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    for column in categorical_columns:

        missing = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing > 0:

            modes = cleaned[
                column
            ].mode(dropna=True)

            if len(modes) > 0:

                cleaned[column] = (
                    cleaned[column]
                    .fillna(modes.iloc[0])
                )

                values_filled += missing

    return (
        cleaned,
        rows_removed,
        values_filled,
        city_changes
    )


def find_sales_column(df):

    preferred = [
        "sales",
        "revenue",
        "amount",
        "total_sales",
        "total_revenue"
    ]

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in preferred:

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
                "amount"
            ]
        ):

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

    return None


def prepare_business_data(df):

    sales_column = find_sales_column(df)

    if sales_column is None:
        return None, None, None

    business_df = df.copy()

    business_df[
        sales_column
    ] = pd.to_numeric(
        business_df[
            sales_column
        ],
        errors="coerce"
    )

    business_df = business_df[
        business_df[
            sales_column
        ].notna()
    ]

    business_df = business_df[
        business_df[
            sales_column
        ] >= 0
    ]

    if len(business_df) == 0:
        return None, sales_column, None

    q1 = business_df[
        sales_column
    ].quantile(.25)

    q3 = business_df[
        sales_column
    ].quantile(.75)

    iqr = q3 - q1

    upper_bound = q3 + 1.5 * iqr

    chart_df = business_df[
        business_df[
            sales_column
        ] <= upper_bound
    ].copy()

    return (
        chart_df,
        sales_column,
        upper_bound
    )


def create_powerbi_exports(
    cleaned_df,
    business_df,
    sales_column
):

    exports = {}

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
        profile_dataset(
            cleaned_df
        )
        .to_csv(index=False)
        .encode("utf-8")
    )

    if (
        business_df is not None
        and sales_column is not None
    ):

        summary = pd.DataFrame({
            "Metric": [
                "Total Sales",
                "Average Sales",
                "Minimum Sales",
                "Maximum Sales",
                "Record Count"
            ],
            "Value": [
                business_df[
                    sales_column
                ].sum(),
                business_df[
                    sales_column
                ].mean(),
                business_df[
                    sales_column
                ].min(),
                business_df[
                    sales_column
                ].max(),
                len(business_df)
            ]
        })

        exports[
            "DataGuard_Sales_Summary.csv"
        ] = (
            summary
            .to_csv(index=False)
            .encode("utf-8")
        )

        city_columns = [
            c
            for c in cleaned_df.columns
            if "city" in str(c).lower()
        ]

        if city_columns:

            city = city_columns[0]

            city_summary = (
                business_df
                .groupby(city)[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count"
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False
                )
            )

            exports[
                "DataGuard_City_Summary.csv"
            ] = (
                city_summary
                .to_csv(index=False)
                .encode("utf-8")
            )

        category_columns = [
            c
            for c in cleaned_df.columns
            if "category"
            in str(c).lower()
        ]

        if category_columns:

            category = category_columns[0]

            category_summary = (
                business_df
                .groupby(category)[
                    sales_column
                ]
                .agg(
                    Total_Sales="sum",
                    Average_Sales="mean",
                    Record_Count="count"
                )
                .reset_index()
                .sort_values(
                    "Total_Sales",
                    ascending=False
                )
            )

            exports[
                "DataGuard_Category_Summary.csv"
            ] = (
                category_summary
                .to_csv(index=False)
                .encode("utf-8")
            )

    return exports


def create_zip(exports):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as z:

        for filename, data in exports.items():
            z.writestr(
                filename,
                data
            )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html("""
<div class="sidebar-brand">
    <div class="brand-row">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">DataGuard AI</div>
            <div class="brand-caption">
                AI Data Quality Platform
            </div>
        </div>
    </div>
</div>
""")

    render_html("""
<div class="sidebar-section">
    Workspace
</div>
""")

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
            "Reports"
        ],
        label_visibility="collapsed"
    )

    render_html("""
<div class="sidebar-section">
    Detection Settings
</div>
""")

    contamination = st.slider(
        "Isolation Forest sensitivity",
        1,
        20,
        st.session_state.contamination_pct
    )

    if (
        contamination
        != st.session_state.contamination_pct
    ):

        st.session_state.contamination_pct = (
            contamination
        )

        st.session_state.analysis_complete = False
        st.session_state.gemini_analysis = None

        st.rerun()

    render_html("""
<div class="sidebar-note">
    Higher sensitivity flags more records as unusual.
    ML anomalies are screening signals.
</div>
""")

    render_html("""
<div class="sidebar-section">
    Pipeline
</div>

<div class="pipeline-card">

    <div class="pipeline-step">
        <span class="pipeline-number">01</span>
        Upload
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">02</span>
        Profile
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">03</span>
        Quality Checks
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">04</span>
        Anomaly Detection
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">05</span>
        Cleaning
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">06</span>
        Analytics
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">07</span>
        Power BI
    </div>

    <div class="pipeline-step">
        <span class="pipeline-number">08</span>
        AI Analysis
    </div>

</div>
""")

    render_html("""
<div class="sidebar-footer">
    DataGuard AI<br>
    Portfolio Edition<br><br>
    Python • Pandas • Scikit-learn<br>
    Streamlit • Gemini
</div>
""")


# ============================================================
# MAIN HEADER
# ============================================================

render_html("""
<div class="page-header">

    <div>
        <div class="eyebrow">
            Data Intelligence Workspace
        </div>

        <div class="page-title">
            DataGuard AI
        </div>

        <div class="page-subtitle">
            AI-powered data quality, anomaly detection
            and business analytics
        </div>
    </div>

    <div class="engine-pill">
        <span class="engine-dot"></span>
        Data Quality Engine
    </div>

</div>
""")


# ============================================================
# UPLOAD SCREEN
# ============================================================

if st.session_state.df is None:

    render_html("""
<div class="hero">

    <div class="hero-label">
        DATA QUALITY PLATFORM
    </div>

    <div class="hero-title">
        Protect the quality of your data
    </div>

    <div class="hero-text">
        Upload your dataset and let DataGuard AI
        profile, validate, detect anomalies, clean
        and prepare your data for business analytics.
    </div>

    <div class="hero-tags">
        <div class="hero-tag">CSV</div>
        <div class="hero-tag">XLSX</div>
        <div class="hero-tag">Data Profiling</div>
        <div class="hero-tag">IQR Detection</div>
        <div class="hero-tag">Isolation Forest</div>
        <div class="hero-tag">Power BI Ready</div>
    </div>

</div>
""")

    render_html("""
<div class="upload-box">

    <div class="upload-icon">
        ↑
    </div>

    <div class="upload-title">
        Upload your dataset
    </div>

    <div class="upload-text">
        Drag and drop your CSV or Excel file,
        or browse your computer.
    </div>

    <div class="upload-meta">
        CSV • XLSX • XLS • Maximum 200 MB
    </div>

</div>
""")

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

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

            st.session_state.analysis_complete = False
            st.session_state.cleaned_df = None
            st.session_state.business_df = None
            st.session_state.analysis = None
            st.session_state.gemini_analysis = None

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to read the file: {error}"
            )

    render_html("""
<div class="section-title">
    What DataGuard AI checks
</div>

<div class="section-subtitle">
    One workflow from raw dataset to analysis-ready data.
</div>
""")

    col1, col2, col3, col4 = st.columns(4)

    features = [
        (
            "🔍",
            "Profile Data",
            "Understand columns, data types, missing values and unique values."
        ),
        (
            "◈",
            "Detect Anomalies",
            "Combine IQR statistical detection with Isolation Forest."
        ),
        (
            "🧹",
            "Clean Data",
            "Remove duplicates, standardize values and handle missing data."
        ),
        (
            "📊",
            "Analyze & Export",
            "Explore business trends and create Power BI-ready datasets."
        )
    ]

    for column, feature in zip(
        [col1, col2, col3, col4],
        features
    ):

        with column:

            render_html(f"""
<div class="feature-card">

    <div class="feature-icon">
        {feature[0]}
    </div>

    <div class="feature-title">
        {feature[1]}
    </div>

    <div class="feature-text">
        {feature[2]}
    </div>

</div>
""")


# ============================================================
# RUN ANALYSIS
# ============================================================

if (
    st.session_state.df is not None
    and not st.session_state.analysis_complete
):

    df = st.session_state.df

    with st.spinner(
        "Running DataGuard AI quality engine..."
    ):

        rows = len(df)
        columns = len(df.columns)

        total_cells = (
            rows * columns
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

        quality_score = build_quality_score(
            total_cells,
            missing_count,
            duplicate_count,
            invalid_count
        )

        datetime_columns = (
            detect_datetime_columns(df)
        )

        identifier_columns = (
            detect_identifier_columns(
                df,
                datetime_columns
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
                    "category"
                ]
            ).columns
        )

        iqr_count, iqr_details = (
            detect_iqr_outliers(df)
        )

        anomaly_mask, ml_count = (
            detect_ml_anomalies(
                df,
                st.session_state.contamination_pct
            )
        )

        normal_count = (
            rows - ml_count
        )

        (
            cleaned_df,
            rows_removed,
            values_filled,
            city_changes
        ) = clean_dataset(df)

        (
            business_df,
            sales_column,
            upper_bound
        ) = prepare_business_data(
            cleaned_df
        )

        st.session_state.cleaned_df = cleaned_df
        st.session_state.business_df = business_df

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
            "iqr_outlier_count": iqr_count,
            "iqr_details": iqr_details,
            "anomaly_mask": anomaly_mask,
            "ml_anomaly_count": ml_count,
            "normal_count": normal_count,
            "invalid_details": invalid_details,
            "rows_removed": rows_removed,
            "values_filled": values_filled,
            "city_changes": city_changes,
            "sales_column": sales_column,
            "upper_bound": upper_bound
        }

        st.session_state.analysis_complete = True

    st.rerun()


# ============================================================
# ANALYZED DATASET
# ============================================================

if (
    st.session_state.df is not None
    and st.session_state.analysis_complete
):

    df = st.session_state.df
    cleaned_df = st.session_state.cleaned_df
    business_df = st.session_state.business_df
    analysis = st.session_state.analysis

    render_html(f"""
<div class="dataset-strip">

    <div>

        <div class="dataset-name">
            {st.session_state.file_name}
        </div>

        <div class="dataset-meta">
            {analysis["rows"]:,} rows
            &nbsp; • &nbsp;
            {analysis["columns"]} columns
            &nbsp; • &nbsp;
            Analysis complete
        </div>

    </div>

    <div class="engine-pill">
        <span class="engine-dot"></span>
        Ready
    </div>

</div>
""")


    # ========================================================
    # DASHBOARD
    # ========================================================

    if page == "Dashboard":

        render_html("""
<div class="section-title">
    Dashboard
</div>

<div class="section-subtitle">
    Monitor dataset health and quality signals.
</div>
""")

        score = analysis["quality_score"]

        if score >= 95:
            status = "Good"
            badge = "badge-good"
        elif score >= 85:
            status = "Needs Attention"
            badge = "badge-warning"
        else:
            status = "Critical"
            badge = "badge-danger"

        left, right = st.columns(
            [1, 2]
        )

        with left:

            render_html(f"""
<div class="score-card">

    <div class="score-label">
        DATA QUALITY SCORE
    </div>

    <div class="score-number">
        {score}
        <span class="score-number-small">
            /100
        </span>
    </div>

    <div class="score-badge {badge}">
        {status}
    </div>

    <div class="score-track">
        <div
            class="score-fill"
            style="width:{score}%"
        ></div>
    </div>

</div>
""")

        with right:

            k1, k2 = st.columns(2)

            with k1:

                render_html(f"""
<div class="kpi-card">

    <div class="kpi-label">
        Missing Cells
    </div>

    <div class="kpi-value">
        {analysis["missing_count"]:,}
    </div>

    <div class="kpi-caption">
        Missing observations
    </div>

</div>
""")

            with k2:

                render_html(f"""
<div class="kpi-card">

    <div class="kpi-label">
        Duplicate Records
    </div>

    <div class="kpi-value">
        {analysis["duplicate_count"]:,}
    </div>

    <div class="kpi-caption">
        Duplicate rows
    </div>

</div>
""")

            k3, k4 = st.columns(2)

            with k3:

                render_html(f"""
<div class="kpi-card">

    <div class="kpi-label">
        IQR Outliers
    </div>

    <div class="kpi-value">
        {analysis["iqr_outlier_count"]:,}
    </div>

    <div class="kpi-caption">
        Statistical signals
    </div>

</div>
""")

            with k4:

                render_html(f"""
<div class="kpi-card">

    <div class="kpi-label">
        ML Anomalies
    </div>

    <div class="kpi-value">
        {analysis["ml_anomaly_count"]:,}
    </div>

    <div class="kpi-caption">
        {st.session_state.contamination_pct}% sensitivity
    </div>

</div>
""")

        render_html("""
<div class="section-title">
    Dataset Profile
</div>

<div class="section-subtitle">
    Structural overview of the uploaded data.
</div>
""")

        p1, p2, p3, p4, p5 = st.columns(5)

        p1.metric(
            "Rows",
            f'{analysis["rows"]:,}'
        )

        p2.metric(
            "Columns",
            analysis["columns"]
        )

        p3.metric(
            "Numeric",
            analysis["numeric_count"]
        )

        p4.metric(
            "Categorical",
            analysis["categorical_count"]
        )

        p5.metric(
            "Date / Time",
            len(
                analysis[
                    "datetime_columns"
                ]
            )
        )

        render_html("""
<div class="section-title">
    Quality Monitoring
</div>

<div class="section-subtitle">
    Detected quality and anomaly signals.
</div>
""")

        quality_df = pd.DataFrame({
            "Metric": [
                "Missing",
                "Duplicates",
                "Invalid",
                "IQR Outliers",
                "ML Anomalies"
            ],
            "Count": [
                analysis["missing_count"],
                analysis["duplicate_count"],
                analysis["invalid_count"],
                analysis["iqr_outlier_count"],
                analysis["ml_anomaly_count"]
            ]
        })

        st.bar_chart(
            quality_df.set_index("Metric"),
            height=300
        )

        if business_df is not None:

            sales_column = analysis[
                "sales_column"
            ]

            render_html("""
<div class="section-title">
    Business Snapshot
</div>

<div class="section-subtitle">
    Commercial metrics detected in the dataset.
</div>
""")

            b1, b2, b3 = st.columns(3)

            b1.metric(
                "Total Sales",
                f'{business_df[sales_column].sum():,.0f}'
            )

            b2.metric(
                "Average Sale",
                f'{business_df[sales_column].mean():,.2f}'
            )

            b3.metric(
                "Sales Records",
                f'{len(business_df):,}'
            )

        render_html("""
<div class="section-title">
    Data Preview
</div>
""")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            height=350
        )


    # ========================================================
    # DATA QUALITY
    # ========================================================

    elif page == "Data Quality":

        render_html("""
<div class="section-title">
    Data Quality
</div>

<div class="section-subtitle">
    Detailed profiling and rule-based validation.
</div>
""")

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Quality Score",
            f'{analysis["quality_score"]}/100'
        )

        q2.metric(
            "Missing",
            f'{analysis["missing_count"]:,}'
        )

        q3.metric(
            "Duplicates",
            f'{analysis["duplicate_count"]:,}'
        )

        q4.metric(
            "Invalid",
            f'{analysis["invalid_count"]:,}'
        )

        render_html("""
<div class="section-title">
    Column Profile
</div>
""")

        st.dataframe(
            profile_dataset(df),
            use_container_width=True,
            height=450
        )

        render_html("""
<div class="section-title">
    Invalid Values
</div>
""")

        if analysis[
            "invalid_details"
        ].empty:

            render_html("""
<div class="notice notice-green">
    No invalid values detected by the configured validation rules.
</div>
""")

        else:

            st.dataframe(
                analysis["invalid_details"],
                use_container_width=True
            )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif page == "Anomalies":

        render_html("""
<div class="section-title">
    Anomaly Detection
</div>

<div class="section-subtitle">
    Statistical and machine-learning screening signals.
</div>

<div class="notice notice-yellow">
    IQR outliers and Isolation Forest anomalies are screening
    signals and are not automatically confirmed data errors.
</div>
""")

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "IQR Outliers",
            f'{analysis["iqr_outlier_count"]:,}'
        )

        a2.metric(
            "ML Anomalies",
            f'{analysis["ml_anomaly_count"]:,}'
        )

        a3.metric(
            "Normal Records",
            f'{analysis["normal_count"]:,}'
        )

        render_html("""
<div class="section-title">
    IQR Statistical Detection
</div>
""")

        if analysis[
            "iqr_details"
        ].empty:

            st.success(
                "No IQR outliers detected."
            )

        else:

            st.dataframe(
                analysis["iqr_details"],
                use_container_width=True
            )

        render_html(f"""
<div class="section-title">
    Isolation Forest
</div>

<div class="notice notice-blue">
    Current sensitivity:
    <b>{st.session_state.contamination_pct}%</b>.
    Change it from the sidebar to rerun the model.
</div>
""")

        anomaly_df = df.copy()

        anomaly_df["ML_Anomaly"] = np.where(
            analysis["anomaly_mask"],
            "Anomaly",
            "Normal"
        )

        anomaly_records = anomaly_df[
            anomaly_df["ML_Anomaly"] == "Anomaly"
        ]

        st.dataframe(
            anomaly_records.head(500),
            use_container_width=True,
            height=450
        )


    # ========================================================
    # CLEANING
    # ========================================================

    elif page == "Cleaning":

        render_html("""
<div class="section-title">
    Automated Cleaning
</div>

<div class="section-subtitle">
    Transform raw data into an analysis-ready dataset.
</div>

<div class="notice notice-green">
    Cleaning completed. Duplicate records were removed,
    city values standardized and missing values handled.
</div>
""")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Original Rows",
            f'{len(df):,}'
        )

        c2.metric(
            "Cleaned Rows",
            f'{len(cleaned_df):,}'
        )

        c3.metric(
            "Rows Removed",
            f'{analysis["rows_removed"]:,}'
        )

        c4.metric(
            "Values Filled",
            f'{analysis["values_filled"]:,}'
        )

        render_html("""
<div class="section-title">
    Cleaned Dataset
</div>
""")

        st.dataframe(
            cleaned_df.head(10),
            use_container_width=True,
            height=350
        )

        st.download_button(
            "Download Cleaned CSV",
            data=(
                cleaned_df
                .to_csv(index=False)
                .encode("utf-8")
            ),
            file_name="DataGuard_Cleaned_Data.csv",
            mime="text/csv",
            type="primary"
        )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif page == "Analytics":

        render_html("""
<div class="section-title">
    Business Analytics
</div>

<div class="section-subtitle">
    Explore trends and commercial patterns from the cleaned dataset.
</div>
""")

        if business_df is None:

            st.warning(
                "No Sales, Revenue or Amount column was detected."
            )

        else:

            sales_column = analysis[
                "sales_column"
            ]

            render_html(f"""
<div class="notice notice-blue">
    Business charts use an IQR upper bound of
    <b>{analysis["upper_bound"]:,.2f}</b>.
    This affects visualization only and does not delete
    records from the cleaned dataset.
</div>
""")

            s1, s2, s3, s4 = st.columns(4)

            s1.metric(
                "Total Sales",
                f'{business_df[sales_column].sum():,.0f}'
            )

            s2.metric(
                "Average Sale",
                f'{business_df[sales_column].mean():,.2f}'
            )

            s3.metric(
                "Highest Sale",
                f'{business_df[sales_column].max():,.2f}'
            )

            s4.metric(
                "Records",
                f'{len(business_df):,}'
            )

            render_html("""
<div class="section-title">
    Sales Distribution
</div>
""")

            hist, bins = np.histogram(
                business_df[sales_column],
                bins=10
            )

            hist_df = pd.DataFrame({
                "Sales Range": [
                    f"{bins[i]:,.0f} - {bins[i+1]:,.0f}"
                    for i in range(
                        len(bins) - 1
                    )
                ],
                "Records": hist
            })

            st.bar_chart(
                hist_df.set_index(
                    "Sales Range"
                ),
                height=320
            )

            city_columns = [
                c
                for c in cleaned_df.columns
                if "city" in str(c).lower()
            ]

            if city_columns:

                city = city_columns[0]

                city_sales = (
                    business_df
                    .groupby(city)[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                    .head(10)
                )

                render_html("""
<div class="section-title">
    Sales by City
</div>
""")

                st.bar_chart(
                    city_sales,
                    height=300
                )

            category_columns = [
                c
                for c in cleaned_df.columns
                if "category"
                in str(c).lower()
            ]

            if category_columns:

                category = category_columns[0]

                category_sales = (
                    business_df
                    .groupby(category)[
                        sales_column
                    ]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                render_html("""
<div class="section-title">
    Sales by Category
</div>
""")

                st.bar_chart(
                    category_sales,
                    height=300
                )


    # ========================================================
    # POWER BI
    # ========================================================

    elif page == "Power BI":

        render_html("""
<div class="section-title">
    Power BI Export Center
</div>

<div class="section-subtitle">
    Business-ready datasets prepared for Power BI.
</div>

<div class="notice notice-blue">
    Import these CSV files into Power BI Desktop using
    <b>Get Data → Text/CSV</b>.
</div>
""")

        exports = create_powerbi_exports(
            cleaned_df,
            business_df,
            analysis["sales_column"]
        )

        for filename, data in exports.items():

            col1, col2 = st.columns(
                [5, 1]
            )

            with col1:

                render_html(f"""
<div class="card">
    <div class="card-title">
        {filename}
    </div>
    <div class="card-subtitle">
        Power BI-ready CSV
    </div>
</div>
""")

            with col2:

                st.download_button(
                    "Download",
                    data=data,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}"
                )

        zip_data = create_zip(
            exports
        )

        st.download_button(
            "Download Complete Power BI Package",
            data=zip_data,
            file_name="DataGuard_PowerBI_Exports.zip",
            mime="application/zip",
            type="primary"
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif page == "AI Analysis":

        render_html("""
<div class="section-title">
    AI Analysis
</div>

<div class="section-subtitle">
    Interpret measured DataGuard signals using Gemini.
</div>
""")

        try:
            GEMINI_API_KEY = st.secrets.get(
                "GEMINI_API_KEY",
                ""
            )
        except Exception:
            GEMINI_API_KEY = ""

        if not GEMINI_API_KEY:
            GEMINI_API_KEY = os.getenv(
                "GEMINI_API_KEY",
                ""
            )

        if not GEMINI_API_KEY:

            render_html("""
<div class="notice notice-yellow">
    Gemini is not configured. Add
    <b>GEMINI_API_KEY</b> to Streamlit secrets
    to enable AI analysis.
</div>
""")

        else:

            if st.button(
                "Generate AI Analysis",
                type="primary"
            ):

                try:

                    from google import genai

                    client = genai.Client(
                        api_key=GEMINI_API_KEY
                    )

                    prompt = f"""
Analyze this DataGuard AI report.

Do not invent facts.

IQR and Isolation Forest findings
are screening signals.

Quality Score:
{analysis["quality_score"]}/100

Rows:
{analysis["rows"]}

Columns:
{analysis["columns"]}

Missing:
{analysis["missing_count"]}

Duplicates:
{analysis["duplicate_count"]}

Invalid:
{analysis["invalid_count"]}

IQR Outliers:
{analysis["iqr_outlier_count"]}

ML Anomalies:
{analysis["ml_anomaly_count"]}

Rows Removed:
{analysis["rows_removed"]}

Values Filled:
{analysis["values_filled"]}

Provide:

1. Overall Assessment
2. Confirmed Data Quality Problems
3. Statistical Outlier Findings
4. ML Anomaly Findings
5. Possible Root Causes
6. Cleaning Results
7. Business Impact
8. Power BI Recommendations
"""

                    with st.spinner(
                        "Generating AI assessment..."
                    ):

                        response = client.interactions.create(
                            model="gemini-3.6-flash",
                            input=prompt,
                            generation_config={
                                "temperature": 0.1
                            }
                        )

                    st.session_state.gemini_analysis = (
                        response.output_text
                    )

                except Exception as error:

                    st.error(
                        f"Gemini error: {error}"
                    )

            if st.session_state.gemini_analysis:

                render_html("""
<div class="section-title">
    AI Assessment
</div>
""")

                st.markdown(
                    st.session_state.gemini_analysis
                )


    # ========================================================
    # REPORTS
    # ========================================================

    elif page == "Reports":

        render_html("""
<div class="section-title">
    Data Quality Report
</div>

<div class="section-subtitle">
    Complete analysis summary for documentation
    and portfolio presentation.
</div>
""")

        report = f"""
DATAGUARD AI
DATA QUALITY REPORT

Dataset:
{st.session_state.file_name}

Rows:
{analysis["rows"]:,}

Columns:
{analysis["columns"]}

Quality Score:
{analysis["quality_score"]}/100

Missing Cells:
{analysis["missing_count"]:,}

Duplicate Records:
{analysis["duplicate_count"]:,}

Invalid Values:
{analysis["invalid_count"]:,}

IQR Outliers:
{analysis["iqr_outlier_count"]:,}

Isolation Forest Sensitivity:
{st.session_state.contamination_pct}%

ML Anomalies:
{analysis["ml_anomaly_count"]:,}

Rows Removed:
{analysis["rows_removed"]:,}

Values Filled:
{analysis["values_filled"]:,}

City Values Standardized:
{analysis["city_changes"]:,}

Sales Column:
{analysis["sales_column"]}

Business Visualization Upper Bound:
{analysis["upper_bound"]}

NOTE:
IQR outliers and Isolation Forest anomalies are
screening signals and are not automatically confirmed
data errors.
"""

        st.code(
            report,
            language="text"
        )

        st.download_button(
            "Download Report",
            data=report.encode("utf-8"),
            file_name="DataGuard_Data_Quality_Report.txt",
            mime="text/plain",
            type="primary"
        )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">
    🛡️ <b>DataGuard AI</b>
    &nbsp; • &nbsp;
    AI Data Quality & Anomaly Detection Platform
    <br>
    Built with Python • Pandas • Scikit-learn • Streamlit
</div>
""")
