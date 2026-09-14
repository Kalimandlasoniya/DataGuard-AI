import io
import os
import zipfile
import re

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

    .stApp {
        background: #080D18;
        color: #F5F7FA;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #0B1120;
        border-right: 1px solid #202A40;
    }

    .brand {
        padding: 8px 4px 25px 4px;
    }

    .brand-icon {
        font-size: 30px;
    }

    .brand-title {
        font-size: 21px;
        font-weight: 800;
        color: #F5F7FA;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #7F8BA0;
    }

    .page-title {
        font-size: 30px;
        font-weight: 800;
        color: #F5F7FA;
        margin-bottom: 5px;
    }

    .page-subtitle {
        font-size: 14px;
        color: #8995AA;
        margin-bottom: 25px;
    }

    .card {
        background: #11192B;
        border: 1px solid #222D45;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 16px;
        font-weight: 750;
        color: #F5F7FA;
        margin-bottom: 6px;
    }

    .card-description {
        font-size: 12px;
        color: #8995AA;
        margin-bottom: 18px;
    }

    .kpi {
        background: #11192B;
        border: 1px solid #222D45;
        border-radius: 14px;
        padding: 16px;
        min-height: 100px;
    }

    .kpi-label {
        font-size: 12px;
        color: #8995AA;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #F5F7FA;
    }

    .status {
        background: #10192A;
        border: 1px solid #26334D;
        border-radius: 12px;
        padding: 12px;
        margin-top: 18px;
    }

    .green {
        color: #48D597;
    }

    .workflow {
        display: flex;
        align-items: center;
        gap: 7px;
        overflow-x: auto;
        padding: 14px 0 22px 0;
    }

    .step {
        white-space: nowrap;
        padding: 9px 13px;
        border-radius: 9px;
        font-size: 12px;
        font-weight: 650;
        background: #11192B;
        border: 1px solid #202A40;
        color: #737F94;
    }

    .step-done {
        color: #5FDCA0;
        border-color: #285540;
    }

    .step-active {
        color: #F5F7FA;
        background: #18243B;
        border-color: #4C6794;
    }

    .arrow {
        color: #526079;
    }

    .footer {
        margin-top: 35px;
        padding-top: 20px;
        border-top: 1px solid #202A40;
        text-align: center;
        color: #68748A;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_info" not in st.session_state:
    st.session_state.cleaning_info = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "gemini_analysis" not in st.session_state:
    st.session_state.gemini_analysis = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = 5


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def page_header(title, description):

    st.markdown(
        f"""
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{description}</div>
        """,
        unsafe_allow_html=True
    )


def html_card(title, description=None):

    description_html = ""

    if description:
        description_html = (
            f'<div class="card-description">'
            f'{description}'
            f'</div>'
        )

    html = (
        '<div class="card">'
        f'<div class="card-title">{title}</div>'
        f'{description_html}'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def kpi(label, value):

    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def detect_datetime_columns(df):

    result = []

    for col in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[col]
        ):
            result.append(col)
            continue

        if df[col].dtype == "object":

            sample = df[col].dropna().astype(str).head(200)

            if len(sample) == 0:
                continue

            converted = pd.to_datetime(
                sample,
                errors="coerce"
            )

            if converted.notna().mean() >= 0.75:
                result.append(col)

    return result


def invalid_values(df):

    total = 0

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        total += int(
            np.isinf(
                df[col]
            ).sum()
        )

    return total


def iqr_outliers(df):

    total = 0

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        total += int(
            (
                (values < lower)
                |
                (values > upper)
            ).sum()
        )

    return total


def ml_anomalies(df, sensitivity):

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    if numeric.empty or len(df) < 10:
        return 0, len(df)

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.fillna(
        numeric.median(numeric_only=True)
    )

    numeric = numeric.fillna(0)

    model = IsolationForest(
        contamination=sensitivity / 100,
        random_state=42,
        n_estimators=200
    )

    prediction = model.fit_predict(
        numeric
    )

    anomalies = int(
        (prediction == -1).sum()
    )

    normal = int(
        (prediction == 1).sum()
    )

    return anomalies, normal


def analyze(df):

    rows = len(df)
    columns = len(df.columns)

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    invalid = invalid_values(df)

    outliers = iqr_outliers(df)

    anomalies, normal = ml_anomalies(
        df,
        st.session_state.sensitivity
    )

    missing_rate = (
        missing / (rows * columns)
        if rows and columns
        else 0
    )

    duplicate_rate = (
        duplicates / rows
        if rows
        else 0
    )

    invalid_rate = (
        invalid / rows
        if rows
        else 0
    )

    quality = (
        100
        - missing_rate * 50
        - duplicate_rate * 30
        - invalid_rate * 20
    )

    quality = max(
        0,
        min(100, quality)
    )

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "duplicates": duplicates,
        "invalid": invalid,
        "outliers": outliers,
        "anomalies": anomalies,
        "normal": normal,
        "anomaly_rate": (
            anomalies / rows * 100
            if rows
            else 0
        ),
        "quality": quality,
        "numeric": len(
            df.select_dtypes(
                include=np.number
            ).columns
        ),
        "categorical": len(
            df.select_dtypes(
                include=["object", "category"]
            ).columns
        ),
        "datetime": len(
            detect_datetime_columns(df)
        ),
    }


def clean_data(df):

    result = df.copy()

    before = len(result)

    duplicates = int(
        result.duplicated().sum()
    )

    result = result.drop_duplicates()

    filled = 0

    numeric_cols = result.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        count = int(
            result[col].isna().sum()
        )

        if count:

            result[col] = result[col].fillna(
                result[col].median()
            )

            filled += count

    categorical_cols = result.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:

        count = int(
            result[col].isna().sum()
        )

        if count:

            mode = result[col].mode()

            if len(mode):

                result[col] = result[col].fillna(
                    mode.iloc[0]
                )

                filled += count

    return result, {
        "before": before,
        "after": len(result),
        "removed": before - len(result),
        "duplicates": duplicates,
        "filled": filled,
    }


def sales_column(df):

    for col in [
        "Sales",
        "sales",
        "Revenue",
        "revenue",
        "Amount",
        "amount"
    ]:

        if col in df.columns:
            return col

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🛡️</div>
            <div class="brand-title">DataGuard AI</div>
            <div class="brand-subtitle">
                AI Data Quality Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**WORKSPACE**")

    navigation = [
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

    for item in navigation:

        if st.button(
            item,
            key="nav_" + item,
            use_container_width=True
        ):

            st.session_state.page = item
            st.rerun()

    st.divider()

    st.markdown(
        "**DETECTION SETTINGS**"
    )

    st.markdown(
        """
        <div style="
            font-size:13px;
            font-weight:650;
            color:#C9D1DE;
            margin-bottom:4px;
        ">
            Isolation Forest sensitivity
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.sensitivity = st.slider(
        "Sensitivity",
        min_value=1,
        max_value=20,
        value=st.session_state.sensitivity,
        label_visibility="collapsed"
    )

    st.caption(
        "Higher sensitivity flags more records as unusual."
    )

    st.markdown(
        """
        <div class="status">
            <div style="
                font-size:11px;
                color:#77839A;
            ">
                SYSTEM
            </div>

            <div style="
                margin-top:5px;
                font-size:13px;
            ">
                <span class="green">●</span>
                All systems operational
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# WORKFLOW
# ============================================================

workflow = [
    "Upload",
    "Profile",
    "Quality",
    "Anomalies",
    "Clean",
    "Analytics",
    "Power BI",
    "AI",
]

workflow_step = {
    "Dashboard": 0,
    "Upload Data": 1,
    "Data Profile": 2,
    "Quality Checks": 3,
    "Anomaly Detection": 4,
    "Data Cleaning": 5,
    "Analytics": 6,
    "Power BI": 7,
    "AI Analysis": 8,
    "Reports": 8,
}

current_step = workflow_step[
    st.session_state.page
]

workflow_html = '<div class="workflow">'

for number, name in enumerate(
    workflow,
    start=1
):

    if current_step == 0:

        css = "step"
        label = f"{number:02d} {name}"

    elif number < current_step:

        css = "step step-done"
        label = f"✓ {name}"

    elif number == current_step:

        css = "step step-active"
        label = f"{number:02d} {name}"

    else:

        css = "step"
        label = f"{number:02d} {name}"

    workflow_html += (
        f'<div class="{css}">'
        f'{label}'
        f'</div>'
    )

    if number < 8:

        workflow_html += (
            '<div class="arrow">→</div>'
        )

workflow_html += "</div>"

st.markdown(
    workflow_html,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD / EMPTY STATE
# ============================================================

if st.session_state.df is None:

    page_header(
        "Welcome to DataGuard AI",
        "Upload a dataset to begin your data intelligence workflow."
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Upload your dataset
            </div>

            <div class="card-description">
                CSV, XLSX or XLS files • Maximum 50 MB
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded:

        file_size = (
            uploaded.size /
            (1024 * 1024)
        )

        if file_size > 50:

            st.error(
                "File exceeds the 50 MB limit."
            )

        else:

            try:

                if uploaded.name.lower().endswith(
                    ".csv"
                ):

                    data = pd.read_csv(
                        uploaded
                    )

                else:

                    data = pd.read_excel(
                        uploaded
                    )

                st.session_state.df = data

                st.session_state.file_name = (
                    uploaded.name
                )

                st.session_state.analysis = (
                    analyze(data)
                )

                st.session_state.page = (
                    "Data Profile"
                )

                st.rerun()

            except Exception as error:

                st.error(
                    f"Unable to read file: {error}"
                )


# ============================================================
# MAIN APPLICATION
# ============================================================

else:

    df = st.session_state.df

    st.session_state.analysis = analyze(
        df
    )

    analysis = st.session_state.analysis


    # ========================================================
    # DASHBOARD
    # ========================================================

    if st.session_state.page == "Dashboard":

        page_header(
            "Data Quality Overview",
            "Monitor, analyze, and improve the quality of your dataset."
        )

        columns = st.columns(6)

        dashboard_values = [
            (
                "Data Quality",
                f"{analysis['quality']:.2f}%"
            ),
            (
                "Total Records",
                f"{analysis['rows']:,}"
            ),
            (
                "Columns",
                f"{analysis['columns']:,}"
            ),
            (
                "Missing Values",
                f"{analysis['missing']:,}"
            ),
            (
                "Duplicates",
                f"{analysis['duplicates']:,}"
            ),
            (
                "Anomalies",
                f"{analysis['anomalies']:,}"
            ),
        ]

        for column, item in zip(
            columns,
            dashboard_values
        ):

            with column:
                kpi(
                    item[0],
                    item[1]
                )

        st.write("")

        left, right = st.columns(
            [1, 1.5]
        )

        with left:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Overall Data Quality Score
                    </div>

                    <div class="card-description">
                        Current measured quality score
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    font-size:42px;
                    font-weight:850;
                    color:#F5F7FA;
                ">
                    {analysis['quality']:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                analysis["quality"] / 100
            )

        with right:

            st.markdown(
                """
                <div class="card">

                    <div class="card-title">
                        Issues Detected
                    </div>

                    <div class="card-description">
                        Measured signals from the dataset
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            issue_columns = st.columns(5)

            issue_values = [
                (
                    "Missing",
                    analysis["missing"]
                ),
                (
                    "Duplicates",
                    analysis["duplicates"]
                ),
                (
                    "Outliers",
                    analysis["outliers"]
                ),
                (
                    "Invalid",
                    analysis["invalid"]
                ),
                (
                    "ML Anomalies",
                    analysis["anomalies"]
                ),
            ]

            for column, item in zip(
                issue_columns,
                issue_values
            ):

                with column:

                    kpi(
                        item[0],
                        f"{item[1]:,}"
                    )

        st.write("")

        html_card(
            "Anomaly Detection",
            "Isolation Forest screening results."
        )

        anomaly_columns = st.columns(4)

        with anomaly_columns[0]:
            kpi(
                "Normal Records",
                f"{analysis['normal']:,}"
            )

        with anomaly_columns[1]:
            kpi(
                "Anomalies",
                f"{analysis['anomalies']:,}"
            )

        with anomaly_columns[2]:
            kpi(
                "Anomaly Rate",
                f"{analysis['anomaly_rate']:.2f}%"
            )

        with anomaly_columns[3]:
            kpi(
                "Algorithm",
                "Isolation Forest"
            )

        st.caption(
            "ML anomalies are screening signals, not confirmed errors."
        )


    # ========================================================
    # UPLOAD DATA
    # ========================================================

    elif st.session_state.page == "Upload Data":

        page_header(
            "Upload Data",
            "Upload CSV or Excel data to begin the DataGuard workflow."
        )

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    Upload your dataset
                </div>

                <div class="card-description">
                    CSV, XLSX or XLS files • Maximum 50 MB
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Choose a dataset",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

        if uploaded:

            size_mb = (
                uploaded.size /
                (1024 * 1024)
            )

            if size_mb > 50:

                st.error(
                    "File exceeds the 50 MB limit."
                )

            else:

                try:

                    if uploaded.name.lower().endswith(
                        ".csv"
                    ):

                        new_df = pd.read_csv(
                            uploaded
                        )

                    else:

                        new_df = pd.read_excel(
                            uploaded
                        )

                    st.session_state.df = new_df
                    st.session_state.file_name = uploaded.name
                    st.session_state.cleaned_df = None
                    st.session_state.gemini_analysis = None

                    st.session_state.analysis = analyze(
                        new_df
                    )

                    st.success(
                        "Dataset uploaded successfully."
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Unable to read file: {error}"
                    )

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Current Dataset
                </div>

                <div style="
                    color:#C8D0DD;
                    font-size:14px;
                ">
                    📄 {st.session_state.file_name}
                </div>

                <div style="
                    color:#7F8BA0;
                    font-size:12px;
                    margin-top:6px;
                ">
                    {len(df):,} rows
                    •
                    {len(df.columns):,} columns
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # DATA PROFILE
    # ========================================================

    elif st.session_state.page == "Data Profile":

        page_header(
            "Data Profile",
            "Understand the structure and completeness of your dataset."
        )

        profile = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [
                str(df[column].dtype)
                for column in df.columns
            ],
            "Non-Null": [
                int(df[column].notna().sum())
                for column in df.columns
            ],
            "Missing": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
            "Unique": [
                int(df[column].nunique())
                for column in df.columns
            ],
        })

        st.dataframe(
            profile,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # QUALITY CHECKS
    # ========================================================

    elif st.session_state.page == "Quality Checks":

        page_header(
            "Quality Checks",
            "Measure completeness, consistency, validity and uniqueness."
        )

        columns = st.columns(4)

        with columns[0]:
            completeness = (
                100 -
                (
                    analysis["missing"] /
                    max(
                        1,
                        analysis["rows"] *
                        analysis["columns"]
                    )
                ) * 100
            )

            kpi(
                "Completeness",
                f"{completeness:.2f}%"
            )

        with columns[1]:
            kpi(
                "Duplicates",
                f"{analysis['duplicates']:,}"
            )

        with columns[2]:
            kpi(
                "Invalid Values",
                f"{analysis['invalid']:,}"
            )

        with columns[3]:
            kpi(
                "IQR Outliers",
                f"{analysis['outliers']:,}"
            )

        st.write("")

        quality = pd.DataFrame({
            "Check": [
                "Completeness",
                "Duplicates",
                "Validity",
                "Uniqueness",
                "IQR Outliers",
            ],
            "Count": [
                analysis["missing"],
                analysis["duplicates"],
                analysis["invalid"],
                analysis["duplicates"],
                analysis["outliers"],
            ],
        })

        st.dataframe(
            quality,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # ANOMALIES
    # ========================================================

    elif st.session_state.page == "Anomaly Detection":

        page_header(
            "Anomaly Detection",
            "Identify unusual records using Isolation Forest."
        )

        columns = st.columns(4)

        with columns[0]:
            kpi(
                "Normal Records",
                f"{analysis['normal']:,}"
            )

        with columns[1]:
            kpi(
                "Anomalies",
                f"{analysis['anomalies']:,}"
            )

        with columns[2]:
            kpi(
                "Anomaly Rate",
                f"{analysis['anomaly_rate']:.2f}%"
            )

        with columns[3]:
            kpi(
                "Sensitivity",
                f"{st.session_state.sensitivity}%"
            )

        numeric_columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        if len(numeric_columns) >= 2:

            x = st.selectbox(
                "X-axis",
                numeric_columns
            )

            y = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=1
            )

            plot_data = df[
                [x, y]
            ].copy()

            plot_data[x] = pd.to_numeric(
                plot_data[x],
                errors="coerce"
            )

            plot_data[y] = pd.to_numeric(
                plot_data[y],
                errors="coerce"
            )

            plot_data = plot_data.dropna()

            st.scatter_chart(
                plot_data,
                x=x,
                y=y,
                use_container_width=True
            )

        st.caption(
            "Anomalies are screening signals and should be reviewed before treating them as confirmed errors."
        )


    # ========================================================
    # DATA CLEANING
    # ========================================================

    elif st.session_state.page == "Data Cleaning":

        page_header(
            "Data Cleaning",
            "Remove duplicates and safely fill missing values."
        )

        if st.session_state.cleaned_df is None:

            if st.button(
                "Run Data Cleaning",
                type="primary"
            ):

                cleaned, info = clean_data(
                    df
                )

                st.session_state.cleaned_df = cleaned
                st.session_state.cleaning_info = info

                st.rerun()

        else:

            cleaned = (
                st.session_state.cleaned_df
            )

            info = (
                st.session_state.cleaning_info
            )

            columns = st.columns(4)

            with columns[0]:
                kpi(
                    "Rows Before",
                    f"{info['before']:,}"
                )

            with columns[1]:
                kpi(
                    "Rows After",
                    f"{info['after']:,}"
                )

            with columns[2]:
                kpi(
                    "Rows Removed",
                    f"{info['removed']:,}"
                )

            with columns[3]:
                kpi(
                    "Values Filled",
                    f"{info['filled']:,}"
                )

            st.write("")

            st.dataframe(
                cleaned.head(20),
                use_container_width=True,
                hide_index=True
            )

            csv = cleaned.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Cleaned Dataset",
                data=csv,
                file_name="DataGuard_Cleaned_Data.csv",
                mime="text/csv"
            )


    # ========================================================
    # ANALYTICS
    # ========================================================

    elif st.session_state.page == "Analytics":

        page_header(
            "Business Analytics",
            "Explore sales performance and business trends."
        )

        sales = sales_column(df)

        if sales is None:

            st.warning(
                "No Sales, Revenue or Amount column was detected."
            )

        else:

            values = pd.to_numeric(
                df[sales],
                errors="coerce"
            ).dropna()

            c1, c2, c3 = st.columns(3)

            with c1:
                kpi(
                    "Total Sales",
                    f"{values.sum():,.2f}"
                )

            with c2:
                kpi(
                    "Average Sale",
                    f"{values.mean():,.2f}"
                )

            with c3:
                kpi(
                    "Highest Sale",
                    f"{values.max():,.2f}"
                )

            st.write("")

            chart = pd.DataFrame({
                sales: values.head(100)
            })

            st.bar_chart(
                chart,
                use_container_width=True
            )


    # ========================================================
    # POWER BI
    # ========================================================

    elif st.session_state.page == "Power BI":

        page_header(
            "Power BI",
            "Prepare analysis-ready data for Power BI."
        )

        export_df = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        csv = export_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Power BI Dataset",
            data=csv,
            file_name="DataGuard_PowerBI_Data.csv",
            mime="text/csv",
            type="primary"
        )

        st.write("")

        st.info(
            "Use the downloaded CSV in Power BI Desktop to build reports and dashboards."
        )


    # ========================================================
    # AI ANALYSIS
    # ========================================================

    elif st.session_state.page == "AI Analysis":

        page_header(
            "AI Analysis",
            "AI-powered interpretation of measured data quality signals."
        )

        columns = st.columns(4)

        with columns[0]:
            kpi(
                "Quality",
                f"{analysis['quality']:.2f}%"
            )

        with columns[1]:
            kpi(
                "Records",
                f"{analysis['rows']:,}"
            )

        with columns[2]:
            kpi(
                "Anomalies",
                f"{analysis['anomalies']:,}"
            )

        with columns[3]:
            kpi(
                "Missing",
                f"{analysis['missing']:,}"
            )

        st.write("")

        st.info(
            "Gemini integration can be connected here using your GEMINI_API_KEY."
        )


    # ========================================================
    # REPORTS
    # ========================================================

    elif st.session_state.page == "Reports":

        page_header(
            "Reports",
            "Download your DataGuard AI quality summary."
        )

        report = pd.DataFrame({
            "Metric": [
                "Data Quality Score",
                "Total Records",
                "Columns",
                "Missing Values",
                "Duplicates",
                "Invalid Values",
                "IQR Outliers",
                "ML Anomalies",
                "Anomaly Rate",
            ],
            "Value": [
                f"{analysis['quality']:.2f}%",
                analysis["rows"],
                analysis["columns"],
                analysis["missing"],
                analysis["duplicates"],
                analysis["invalid"],
                analysis["outliers"],
                analysis["anomalies"],
                f"{analysis['anomaly_rate']:.2f}%",
            ],
        })

        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )

        report_csv = report.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Report",
            data=report_csv,
            file_name="DataGuard_Quality_Report.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🛡️ DataGuard AI • AI Data Quality & Anomaly Detection Platform
        <br>
        Python • Pandas • Scikit-learn • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
