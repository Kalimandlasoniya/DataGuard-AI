%%writefile /content/app.py
import io
import os
import zipfile
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
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
# CONSTANTS
# ============================================================
APP_NAME = "DataGuard AI"
GEMINI_MODEL = "gemini-3.6-flash"

CITY_MAPPING = {
    "Bangalore": "Bengaluru",
    "bangalore": "Bengaluru",
    "BLR": "Bengaluru",
    "BENGALURU": "Bengaluru",
}

AGE_MIN = 18
AGE_MAX = 100

POWERBI_BLOB_FILES = [
    "quality_summary.csv",
    "quality_issues.csv",
    "anomaly_results.csv",
    "cleaning_log.csv",
    "cleaned_data.csv",
]


# ============================================================
# SECRETS / ENVIRONMENT
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

POWERBI_TENANT_ID = get_setting("POWERBI_TENANT_ID")
POWERBI_CLIENT_ID = get_setting("POWERBI_CLIENT_ID")
POWERBI_CLIENT_SECRET = get_setting("POWERBI_CLIENT_SECRET")
POWERBI_WORKSPACE_ID = get_setting("POWERBI_WORKSPACE_ID")
POWERBI_DATASET_ID = get_setting("POWERBI_DATASET_ID")

AZURE_BLOB_CONTAINER_SAS_URL = get_setting(
    "AZURE_BLOB_CONTAINER_SAS_URL"
)

AZURE_BLOB_PREFIX = get_setting(
    "AZURE_BLOB_PREFIX",
    ""
).strip("/")


# ============================================================
# GEMINI
# ============================================================
@st.cache_resource(show_spinner=False)
def get_gemini_client():

    if not GEMINI_API_KEY:
        return None

    try:
        from google import genai

        return genai.Client(
            api_key=GEMINI_API_KEY
        )

    except Exception:
        return None


def ask_gemini(issue_summary: str):

    client = get_gemini_client()

    if client is None:
        return (
            "Gemini AI is not configured. Add GEMINI_API_KEY "
            "to Streamlit Secrets or the environment."
        )

    prompt = f"""
You are an expert Data Quality Analyst.

Analyze the following detected data-quality findings:

{issue_summary}

For each important issue, provide:
1. Likely root cause
2. Business impact
3. Recommended action

Rules:
- Treat root causes as possible explanations, not confirmed facts.
- Do not invent facts.
- If the exact cause cannot be determined, say:
  "The exact root cause requires further investigation."
- Distinguish data errors from statistical outliers and ML anomalies.
- Do not call an anomaly fraud.
- Do not recommend automatically deleting anomalies.
- Use concise, professional language.
"""

    try:

        response = client.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
            generation_config={
                "temperature": 0.2
            },
        )

        return response.output_text

    except Exception as exc:

        return f"Gemini analysis failed: {exc}"


# ============================================================
# POWER BI + AZURE BLOB AUTOMATION
# ============================================================
def powerbi_config_status():

    required = {
        "POWERBI_TENANT_ID": POWERBI_TENANT_ID,
        "POWERBI_CLIENT_ID": POWERBI_CLIENT_ID,
        "POWERBI_CLIENT_SECRET": POWERBI_CLIENT_SECRET,
        "POWERBI_WORKSPACE_ID": POWERBI_WORKSPACE_ID,
        "POWERBI_DATASET_ID": POWERBI_DATASET_ID,
    }

    missing = [
        key
        for key, value in required.items()
        if not value
    ]

    return missing


def blob_config_status():

    if AZURE_BLOB_CONTAINER_SAS_URL:
        return []

    return [
        "AZURE_BLOB_CONTAINER_SAS_URL"
    ]


def build_blob_url(blob_name: str):

    prefix = AZURE_BLOB_PREFIX

    final_name = (
        f"{prefix}/{blob_name}"
        if prefix
        else blob_name
    )

    parts = urlsplit(
        AZURE_BLOB_CONTAINER_SAS_URL
    )

    path = (
        parts.path.rstrip("/")
        + "/"
        + quote(final_name, safe="/")
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            path,
            parts.query,
            "",
        )
    )


def upload_to_azure_blob(
    blob_name: str,
    data: bytes,
):

    if not AZURE_BLOB_CONTAINER_SAS_URL:

        return {
            "success": False,
            "message": (
                "Azure Blob automation is not configured."
            ),
        }

    try:

        blob_url = build_blob_url(
            blob_name
        )

        response = requests.put(
            blob_url,
            data=data,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "text/csv",
            },
            timeout=60,
        )

        if response.status_code in (
            200,
            201,
        ):

            return {
                "success": True,
                "message": (
                    f"Uploaded {blob_name} "
                    "to Azure Blob Storage."
                ),
            }

        return {
            "success": False,
            "message": (
                f"Azure Blob upload failed for "
                f"{blob_name}. "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Azure Blob upload error: {exc}"
            ),
        }


def upload_powerbi_files_to_blob(
    export_files: dict
):

    results = []

    for filename, data in export_files.items():

        if filename in POWERBI_BLOB_FILES:

            results.append(
                upload_to_azure_blob(
                    filename,
                    data,
                )
            )

    failed = [
        item
        for item in results
        if not item["success"]
    ]

    return {
        "success": len(failed) == 0,
        "results": results,
        "message": (
            "All Power BI source files were "
            "uploaded successfully."
            if not failed
            else
            "One or more Power BI source files "
            "failed to upload."
        ),
    }


def get_powerbi_access_token():

    if not all(
        [
            POWERBI_TENANT_ID,
            POWERBI_CLIENT_ID,
            POWERBI_CLIENT_SECRET,
        ]
    ):

        return (
            None,
            "Power BI authentication settings are incomplete."
        )

    token_url = (
        "https://login.microsoftonline.com/"
        f"{POWERBI_TENANT_ID}/oauth2/v2.0/token"
    )

    token_data = {
        "client_id": POWERBI_CLIENT_ID,
        "client_secret": POWERBI_CLIENT_SECRET,
        "scope": (
            "https://analysis.windows.net/"
            "powerbi/api/.default"
        ),
        "grant_type": "client_credentials",
    }

    response = requests.post(
        token_url,
        data=token_data,
        timeout=30,
    )

    if response.status_code != 200:

        return (
            None,
            "Microsoft Entra authentication failed: "
            f"HTTP {response.status_code} - "
            f"{response.text[:500]}",
        )

    access_token = response.json().get(
        "access_token"
    )

    if not access_token:

        return (
            None,
            "No Power BI access token was returned."
        )

    return access_token, None


def trigger_powerbi_refresh():

    missing = powerbi_config_status()

    if missing:

        return {
            "success": False,
            "message": (
                "Power BI automation is not configured. "
                "Missing: "
                + ", ".join(missing)
            ),
        }

    try:

        access_token, token_error = (
            get_powerbi_access_token()
        )

        if token_error:

            return {
                "success": False,
                "message": token_error,
            }

        refresh_url = (
            "https://api.powerbi.com/v1.0/myorg/"
            f"groups/{POWERBI_WORKSPACE_ID}/"
            f"datasets/{POWERBI_DATASET_ID}/refreshes"
        )

        response = requests.post(
            refresh_url,
            headers={
                "Authorization":
                    f"Bearer {access_token}",
                "Content-Type":
                    "application/json",
            },
            json={},
            timeout=30,
        )

        if response.status_code == 202:

            return {
                "success": True,
                "message": (
                    "Power BI dataset refresh was "
                    "successfully triggered."
                ),
            }

        return {
            "success": False,
            "message": (
                f"Power BI refresh failed. "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            ),
        }

    except Exception as exc:

        return {
            "success": False,
            "message": (
                f"Power BI automation error: {exc}"
            ),
        }


def run_powerbi_automation(
    export_files: dict
):

    blob_missing = blob_config_status()

    if blob_missing:

        return {
            "success": False,
            "stage": "configuration",
            "message": (
                "Automatic Power BI update needs "
                "Azure Blob Storage. Missing: "
                + ", ".join(blob_missing)
            ),
        }

    powerbi_missing = powerbi_config_status()

    if powerbi_missing:

        return {
            "success": False,
            "stage": "configuration",
            "message": (
                "Automatic Power BI update needs "
                "Power BI API credentials. Missing: "
                + ", ".join(powerbi_missing)
            ),
        }

    upload_result = (
        upload_powerbi_files_to_blob(
            export_files
        )
    )

    if not upload_result["success"]:

        return {
            "success": False,
            "stage": "blob_upload",
            "message": upload_result["message"],
            "details": upload_result["results"],
        }

    refresh_result = (
        trigger_powerbi_refresh()
    )

    return {
        "success": refresh_result["success"],
        "stage": "powerbi_refresh",
        "message": refresh_result["message"],
        "details": upload_result["results"],
    }


# ============================================================
# DATA LOADING
# ============================================================
def load_uploaded_file(uploaded_file):

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):

        return pd.read_csv(
            uploaded_file
        )

    if filename.endswith(
        (".xlsx", ".xls")
    ):

        return pd.read_excel(
            uploaded_file
        )

    raise ValueError(
        "Unsupported file type."
    )


# ============================================================
# SAFE DATE/TIME DETECTION
# ============================================================
def detect_datetime_columns(
    df: pd.DataFrame
):

    date_columns = []

    for column in df.columns:

        series = df[column]

        # ----------------------------------------------------
        # Already datetime
        # ----------------------------------------------------
        if pd.api.types.is_datetime64_any_dtype(
            series
        ):

            date_columns.append(
                column
            )

            continue

        # ----------------------------------------------------
        # Numeric columns are not automatically dates
        # ----------------------------------------------------
        if pd.api.types.is_numeric_dtype(
            series
        ):

            continue

        non_null = series.dropna()

        if len(non_null) == 0:

            continue

        # ----------------------------------------------------
        # Convert to strings first.
        # This prevents dtype-related errors.
        # ----------------------------------------------------
        values = (
            non_null
            .astype("string")
            .str.strip()
        )

        # ----------------------------------------------------
        # Strong column-name signal
        # ----------------------------------------------------
        name_signal = any(
            token in str(column).lower()
            for token in [
                "date",
                "time",
                "timestamp",
                "datetime",
            ]
        )

        # ----------------------------------------------------
        # Date/time-like column name
        # ----------------------------------------------------
        if name_signal:

            try:

                parsed = pd.to_datetime(
                    values,
                    errors="coerce",
                    format="mixed",
                )

            except (
                TypeError,
                ValueError,
            ):

                parsed = pd.to_datetime(
                    values,
                    errors="coerce",
                )

            success_rate = (
                parsed.notna().mean()
            )

            if success_rate >= 0.80:

                date_columns.append(
                    column
                )

                continue

        # ----------------------------------------------------
        # Generic date detection
        # ----------------------------------------------------
        sample = values.head(500)

        try:

            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed",
            )

        except (
            TypeError,
            ValueError,
        ):

            parsed = pd.to_datetime(
                sample,
                errors="coerce",
            )

        if len(sample) > 0:

            success_rate = (
                parsed.notna().mean()
            )

            if success_rate >= 0.95:

                date_columns.append(
                    column
                )

    return date_columns


def classify_columns(
    df: pd.DataFrame
):

    date_columns = (
        detect_datetime_columns(df)
    )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    identifier_columns = []

    for column in df.columns:

        name = str(column).lower()

        if (
            column in numeric_columns
            and (
                name == "id"
                or name.endswith("_id")
                or name.endswith("id")
                or "identifier" in name
            )
        ):

            identifier_columns.append(
                column
            )

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in identifier_columns
    ]

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in date_columns
        and column not in identifier_columns
    ]

    return (
        numeric_columns,
        categorical_columns,
        date_columns,
        identifier_columns,
    )


# ============================================================
# QUALITY CHECKS
# ============================================================
def detect_invalid_values(
    df: pd.DataFrame
):

    invalid_rows = []

    if "Age" in df.columns:

        age = pd.to_numeric(
            df["Age"],
            errors="coerce",
        )

        mask = (
            (age < AGE_MIN)
            | (age > AGE_MAX)
        )

        for idx in df.index[
            mask.fillna(False)
        ]:

            invalid_rows.append(
                {
                    "Row": int(idx),
                    "Column": "Age",
                    "Value": df.loc[
                        idx,
                        "Age"
                    ],
                    "Issue": (
                        f"Age outside valid "
                        f"range {AGE_MIN}-{AGE_MAX}"
                    ),
                }
            )

    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce",
        )

        mask = quantity < 0

        for idx in df.index[
            mask.fillna(False)
        ]:

            invalid_rows.append(
                {
                    "Row": int(idx),
                    "Column": "Quantity",
                    "Value": df.loc[
                        idx,
                        "Quantity"
                    ],
                    "Issue": "Negative quantity",
                }
            )

    return pd.DataFrame(
        invalid_rows,
        columns=[
            "Row",
            "Column",
            "Value",
            "Issue",
        ],
    )


def detect_city_inconsistencies(
    df: pd.DataFrame
):

    if "City" not in df.columns:

        return (
            pd.DataFrame(
                columns=[
                    "Observed Value",
                    "Standard Value",
                    "Count",
                ]
            ),
            [],
        )

    city_series = (
        df["City"]
        .astype("string")
    )

    recognized = city_series[
        city_series.str.lower().isin(
            {
                "bangalore",
                "bengaluru",
                "blr",
            }
        )
    ]

    variations = sorted(
        recognized
        .dropna()
        .unique()
        .tolist()
    )

    inconsistent_values = [
        value
        for value in variations
        if str(value) != "Bengaluru"
    ]

    report = []

    for value in variations:

        standard = CITY_MAPPING.get(
            str(value),
            value,
        )

        count = int(
            (
                city_series == value
            ).sum()
        )

        report.append(
            {
                "Observed Value": value,
                "Standard Value": standard,
                "Count": count,
            }
        )

    return (
        pd.DataFrame(report),
        inconsistent_values,
    )


def detect_iqr_outliers(
    df: pd.DataFrame,
    numeric_columns
):

    records = []
    masks = {}

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if pd.isna(iqr):

            continue

        lower = (
            q1 - 1.5 * iqr
        )

        upper = (
            q3 + 1.5 * iqr
        )

        mask = (
            (series < lower)
            | (series > upper)
        ).fillna(False)

        masks[column] = mask

        records.append(
            {
                "Column": column,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count":
                    int(mask.sum()),
            }
        )

    report = pd.DataFrame(
        records
    )

    if masks:

        combined_mask = pd.Series(
            False,
            index=df.index,
        )

        for mask in masks.values():

            combined_mask = (
                combined_mask
                | mask
            )

        total = int(
            combined_mask.sum()
        )

    else:

        total = 0

    return (
        report,
        masks,
        total,
    )


def run_ml_anomaly_detection(
    df: pd.DataFrame,
    features
):

    if len(features) < 2:

        return (
            df.copy(),
            0,
        )

    model_data = (
        df[features].copy()
    )

    for column in features:

        model_data[column] = (
            pd.to_numeric(
                model_data[column],
                errors="coerce",
            )
        )

        model_data[column] = (
            model_data[column]
            .fillna(
                model_data[column]
                .median()
            )
        )

    if len(model_data) < 10:

        return (
            df.copy(),
            0,
        )

    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42,
    )

    predictions = (
        model.fit_predict(
            model_data
        )
    )

    result = df.copy()

    result["ML_Anomaly"] = (
        predictions
    )

    anomaly_count = int(
        (
            predictions == -1
        ).sum()
    )

    return (
        result,
        anomaly_count,
    )


# ============================================================
# CLEANING
# ============================================================
def clean_dataset(
    df: pd.DataFrame
):

    cleaned = df.copy()

    original_rows = len(
        cleaned
    )

    cleaning_actions = []

    # --------------------------------------------------------
    # Standardize cities
    # --------------------------------------------------------
    if "City" in cleaned.columns:

        before = (
            cleaned["City"].copy()
        )

        cleaned["City"] = (
            cleaned["City"]
            .replace(CITY_MAPPING)
        )

        changed = int(
            (
                before.astype("string")
                != cleaned["City"]
                .astype("string")
            ).sum()
        )

        if changed:

            cleaning_actions.append(
                {
                    "Action":
                        "Standardized City values",
                    "Column":
                        "City",
                    "Count":
                        changed,
                }
            )

    # --------------------------------------------------------
    # Numeric cleaning
    # --------------------------------------------------------
    for column, rule_name in [
        (
            "Age",
            "Invalid Age converted to missing",
        ),
        (
            "Quantity",
            "Invalid Quantity converted to missing",
        ),
        (
            "Sales",
            "Sales converted to numeric",
        ),
        (
            "Discount",
            "Discount converted to numeric",
        ),
    ]:

        if column not in cleaned.columns:

            continue

        cleaned[column] = (
            pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )
        )

        if column == "Age":

            mask = (
                (cleaned[column] < AGE_MIN)
                | (cleaned[column] > AGE_MAX)
            ).fillna(False)

            count = int(
                mask.sum()
            )

            if count:

                cleaned.loc[
                    mask,
                    column
                ] = np.nan

                cleaning_actions.append(
                    {
                        "Action": rule_name,
                        "Column": column,
                        "Count": count,
                    }
                )

        if column == "Quantity":

            mask = (
                cleaned[column] < 0
            ).fillna(False)

            count = int(
                mask.sum()
            )

            if count:

                cleaned.loc[
                    mask,
                    column
                ] = np.nan

                cleaning_actions.append(
                    {
                        "Action": rule_name,
                        "Column": column,
                        "Count": count,
                    }
                )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------
    for column in [
        "Age",
        "Sales",
        "Quantity",
        "Discount",
    ]:

        if column not in cleaned.columns:

            continue

        missing_before = int(
            cleaned[column]
            .isna()
            .sum()
        )

        if missing_before == 0:

            continue

        median_value = (
            cleaned[column]
            .median()
        )

        if pd.isna(median_value):

            continue

        cleaned[column] = (
            cleaned[column]
            .fillna(median_value)
        )

        cleaning_actions.append(
            {
                "Action":
                    "Filled missing values with median",
                "Column":
                    column,
                "Count":
                    missing_before,
            }
        )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------
    duplicates_before = int(
        cleaned.duplicated().sum()
    )

    cleaned = (
        cleaned
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if duplicates_before:

        cleaning_actions.append(
            {
                "Action":
                    "Removed duplicate records",
                "Column":
                    "All columns",
                "Count":
                    duplicates_before,
            }
        )

    cleaned_rows = len(
        cleaned
    )

    removed_rows = (
        original_rows
        - cleaned_rows
    )

    values_filled = sum(
        item["Count"]
        for item in cleaning_actions
        if "Filled missing"
        in item["Action"]
    )

    return (
        cleaned,
        original_rows,
        cleaned_rows,
        removed_rows,
        values_filled,
        pd.DataFrame(
            cleaning_actions
        ),
    )


# ============================================================
# REPORT / EXPORT
# ============================================================
def dataframe_bytes(
    df: pd.DataFrame
):

    return (
        df.to_csv(
            index=False
        )
        .encode("utf-8")
    )


def create_powerbi_exports(
    quality_summary,
    quality_issues,
    anomaly_results,
    cleaning_log,
    cleaned_df,
):

    export_files = {
        "quality_summary.csv":
            dataframe_bytes(
                quality_summary
            ),

        "quality_issues.csv":
            dataframe_bytes(
                quality_issues
            ),

        "anomaly_results.csv":
            dataframe_bytes(
                anomaly_results
            ),

        "cleaning_log.csv":
            dataframe_bytes(
                cleaning_log
            ),

        "cleaned_data.csv":
            dataframe_bytes(
                cleaned_df
            ),
    }

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        for filename, data in (
            export_files.items()
        ):

            archive.writestr(
                filename,
                data,
            )

    return (
        export_files,
        zip_buffer.getvalue(),
    )


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
        / max(
            1,
            total_cells
        )
    ) * 100

    invalid_penalty = (
        invalid_count
        / max(
            1,
            total_cells
        )
    ) * 100

    score = 100 - (
        missing_penalty * 0.50
        + duplicate_penalty * 0.30
        + invalid_penalty * 0.20
    )

    return round(
        max(
            0,
            min(
                100,
                score
            )
        ),
        2,
    )


def quality_status(
    score
):

    if score >= 95:

        return "🟢 Excellent"

    if score >= 85:

        return "🟡 Good"

    if score >= 70:

        return "🟠 Needs Attention"

    return "🔴 Poor"


# ============================================================
# UI HELPERS
# ============================================================
def apply_custom_css():

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.2rem;
        }

        .pipeline {
            padding: 0.9rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 1rem;
        }

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_pipeline():

    st.markdown(
        """
        <div class="pipeline">
        <b>Pipeline</b><br><br>
        1. Upload Data →
        2. Profile Dataset →
        3. Detect Quality Issues →
        4. Detect Statistical Outliers →
        5. Detect ML Anomalies →
        6. Analyze Root Causes →
        7. Clean Data →
        8. Sync Power BI →
        9. Generate Report
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICATION
# ============================================================
apply_custom_css()

st.markdown(
    '<div class="main-title">🛡️ DataGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Intelligent Data Quality & Root-Cause Analysis System"
    "</div>",
    unsafe_allow_html=True,
)

show_pipeline()

gemini_client = (
    get_gemini_client()
)

if gemini_client:

    st.success(
        "Gemini AI: Connected"
    )

else:

    st.info(
        "Gemini AI is optional. Add GEMINI_API_KEY "
        "to enable AI root-cause analysis."
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.header(
        "⚙️ Configuration"
    )

    st.write(
        "Upload a CSV or Excel dataset to run "
        "the complete DataGuard AI pipeline."
    )

    st.divider()

    st.subheader(
        "Automation Status"
    )

    blob_missing = (
        blob_config_status()
    )

    powerbi_missing = (
        powerbi_config_status()
    )

    if (
        not blob_missing
        and not powerbi_missing
    ):

        st.success(
            "Azure Blob + Power BI automation configured"
        )

    else:

        st.warning(
            "Power BI automation not fully configured"
        )

    st.caption(
        "Secrets are read from environment variables "
        "or Streamlit Secrets and are never displayed."
    )


# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "📂 Upload your CSV or Excel file",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
    help=(
        "Supported formats: CSV, XLSX, XLS"
    ),
)

if uploaded_file is None:

    st.info(
        "Upload a dataset to start the "
        "DataGuard AI pipeline."
    )

    st.markdown(
        "### What DataGuard AI checks"
    )

    st.markdown(
        """
        - Missing values
        - Duplicate records
        - Invalid values
        - Category inconsistencies
        - Date/time fields
        - Statistical outliers
        - ML-based anomalies
        - Root-cause hypotheses
        - Automated cleaning
        - Power BI-ready exports
        """
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================
try:

    df_original = (
        load_uploaded_file(
            uploaded_file
        )
    )

except Exception as exc:

    st.error(
        f"Unable to load the dataset: {exc}"
    )

    st.stop()


if df_original.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


df = df_original.copy()

st.success(
    f"Successfully uploaded: "
    f"{uploaded_file.name}"
)


# ============================================================
# BASIC PROFILE
# ============================================================
rows = len(df)

columns = len(
    df.columns
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
    numeric_columns,
    categorical_columns,
    date_columns,
    identifier_columns,
) = classify_columns(df)


# ============================================================
# CONVERT DETECTED DATE COLUMNS
# ============================================================
for column in date_columns:

    try:

        try:

            df[column] = (
                pd.to_datetime(
                    df[column],
                    errors="coerce",
                    format="mixed",
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            df[column] = (
                pd.to_datetime(
                    df[column],
                    errors="coerce",
                )
            )

    except Exception:
        pass


# ============================================================
# DATASET PREVIEW
# ============================================================
st.header(
    "1️⃣ Dataset Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True,
    height=320,
)


# ============================================================
# PROFILE
# ============================================================
st.header(
    "2️⃣ Dataset Profile"
)

c1, c2, c3, c4 = (
    st.columns(4)
)

c1.metric(
    "Rows",
    f"{rows:,}"
)

c2.metric(
    "Columns",
    columns
)

c3.metric(
    "Missing Cells",
    f"{missing_count:,}"
)

c4.metric(
    "Duplicates",
    f"{duplicate_count:,}"
)


with st.expander(
    "Column Classification",
    expanded=True,
):

    p1, p2, p3, p4 = (
        st.columns(4)
    )

    p1.write(
        "**Numeric Columns**"
    )

    p1.write(
        numeric_columns or "None"
    )

    p2.write(
        "**Categorical Columns**"
    )

    p2.write(
        categorical_columns or "None"
    )

    p3.write(
        "**Date/Time Columns**"
    )

    p3.write(
        date_columns or "None"
    )

    p4.write(
        "**Identifier Columns**"
    )

    p4.write(
        identifier_columns or "None"
    )


if date_columns:

    st.success(
        "Date/time detection successful: "
        + ", ".join(date_columns)
    )


# ============================================================
# QUALITY CHECKS
# ============================================================
st.header(
    "3️⃣ Data Quality Checks"
)

invalid_report = (
    detect_invalid_values(df)
)

invalid_count = len(
    invalid_report
)

(
    city_report,
    detected_city_variations,
) = detect_city_inconsistencies(df)

city_inconsistency_count = 0

if not city_report.empty:

    city_inconsistency_count = int(
        city_report.loc[
            city_report[
                "Observed Value"
            ].astype(str)
            != city_report[
                "Standard Value"
            ].astype(str),
            "Count",
        ].sum()
    )


q1, q2, q3, q4 = (
    st.columns(4)
)

q1.metric(
    "Missing Cells",
    f"{missing_count:,}"
)

q2.metric(
    "Duplicates",
    f"{duplicate_count:,}"
)

q3.metric(
    "Invalid Values",
    f"{invalid_count:,}"
)

q4.metric(
    "City Inconsistencies",
    f"{city_inconsistency_count:,}"
)


with st.expander(
    "Missing Values",
    expanded=True,
):

    missing_series = (
        df.isna()
        .sum()
    )

    missing_series = (
        missing_series[
            missing_series > 0
        ]
    )

    if len(missing_series):

        missing_report = (
            pd.DataFrame(
                {
                    "Column":
                        missing_series.index,

                    "Missing Count":
                        missing_series.values,

                    "Missing %":
                        (
                            missing_series.values
                            / len(df)
                            * 100
                        ).round(2),
                }
            )
        )

        st.dataframe(
            missing_report,
            use_container_width=True,
        )

        st.bar_chart(
            missing_report
            .set_index("Column")[
                "Missing Count"
            ]
        )

    else:

        st.success(
            "No missing values detected."
        )


with st.expander(
    "Duplicate Records"
):

    if duplicate_count:

        st.warning(
            f"{duplicate_count} "
            "duplicate records detected."
        )

    else:

        st.success(
            "No duplicate records detected."
        )


with st.expander(
    "Invalid Values"
):

    if invalid_report.empty:

        st.success(
            "No rule-based invalid values detected."
        )

    else:

        st.dataframe(
            invalid_report,
            use_container_width=True,
        )


with st.expander(
    "City Consistency"
):

    if city_report.empty:

        st.info(
            "No City column detected."
        )

    else:

        st.dataframe(
            city_report,
            use_container_width=True,
        )


# ============================================================
# STATISTICAL OUTLIERS
# ============================================================
st.header(
    "4️⃣ Statistical Outlier Detection"
)

(
    outlier_report,
    outlier_masks,
    total_outliers,
) = detect_iqr_outliers(
    df,
    numeric_columns,
)

st.info(
    "IQR outliers identify statistically unusual "
    "observations. They are not automatically errors "
    "and should be investigated."
)

st.metric(
    "Total Statistical Outlier Flags",
    f"{total_outliers:,}",
)

if not outlier_report.empty:

    st.dataframe(
        outlier_report,
        use_container_width=True,
    )

    outlier_chart = (
        outlier_report[
            [
                "Column",
                "Outlier Count",
            ]
        ]
        .set_index("Column")
    )

    st.bar_chart(
        outlier_chart
    )


# ============================================================
# BUSINESS VISUALS
# ============================================================
st.header(
    "5️⃣ Business Quality & Dataset Insights"
)

visual_columns = st.columns(2)


with visual_columns[0]:

    if "Sales" in df.columns:

        sales_numeric = (
            pd.to_numeric(
                df["Sales"],
                errors="coerce",
            )
        )

        st.subheader(
            "Sales Distribution"
        )

        sales_hist = pd.DataFrame(
            {
                "Sales":
                    sales_numeric.dropna()
            }
        )

        if not sales_hist.empty:

            st.bar_chart(
                sales_hist[
                    "Sales"
                ]
                .value_counts(
                    bins=10
                )
                .sort_index()
            )


with visual_columns[1]:

    if "City" in df.columns:

        st.subheader(
            "Records by City"
        )

        city_counts = (
            df["City"]
            .astype("string")
            .value_counts()
            .head(10)
        )

        st.bar_chart(
            city_counts
        )


if (
    "Product" in df.columns
    and "Sales" in df.columns
):

    st.subheader(
        "Sales by Product"
    )

    product_sales = (
        df.assign(
            Sales_Numeric=pd.to_numeric(
                df["Sales"],
                errors="coerce",
            )
        )
        .groupby(
            "Product"
        )[
            "Sales_Numeric"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    st.bar_chart(
        product_sales
    )


if (
    "Category" in df.columns
    and "Sales" in df.columns
):

    st.subheader(
        "Sales by Category"
    )

    category_sales = (
        df.assign(
            Sales_Numeric=pd.to_numeric(
                df["Sales"],
                errors="coerce",
            )
        )
        .groupby(
            "Category"
        )[
            "Sales_Numeric"
        ]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        category_sales
    )


if (
    date_columns
    and "Sales" in df.columns
):

    date_col = (
        date_columns[0]
    )

    time_data = df.copy()

    time_data[
        "Sales_Numeric"
    ] = pd.to_numeric(
        time_data["Sales"],
        errors="coerce",
    )

    try:

        time_data[
            date_col
        ] = pd.to_datetime(
            time_data[
                date_col
            ],
            errors="coerce",
            format="mixed",
        )

    except (
        TypeError,
        ValueError,
    ):

        time_data[
            date_col
        ] = pd.to_datetime(
            time_data[
                date_col
            ],
            errors="coerce",
        )

    monthly_sales = (
        time_data
        .dropna(
            subset=[date_col]
        )
        .set_index(
            date_col
        )[
            "Sales_Numeric"
        ]
        .resample("MS")
        .sum()
    )

    if not monthly_sales.empty:

        st.subheader(
            f"Sales Trend by Month — "
            f"{date_col}"
        )

        st.line_chart(
            monthly_sales
        )


# ============================================================
# ML ANOMALIES
# ============================================================
st.header(
    "6️⃣ Machine Learning Anomaly Detection"
)

ml_features = [
    column
    for column in [
        "Age",
        "Quantity",
        "Sales",
        "Discount",
    ]
    if column in df.columns
]


if len(ml_features) >= 2:

    df, anomaly_count = (
        run_ml_anomaly_detection(
            df,
            ml_features,
        )
    )

    m1, m2 = (
        st.columns(2)
    )

    m1.metric(
        "ML Anomalies",
        f"{anomaly_count:,}",
    )

    m2.metric(
        "Normal Records",
        f"{len(df) - anomaly_count:,}",
    )

    st.write(
        "**ML Features:** "
        + ", ".join(
            ml_features
        )
    )

    st.info(
        "ML anomalies represent unusual multivariate "
        "patterns. They are not automatically data errors "
        "or fraud."
    )

    anomaly_results = (
        df[
            df["ML_Anomaly"] == -1
        ].copy()
    )

    if not anomaly_results.empty:

        st.dataframe(
            anomaly_results.head(50),
            use_container_width=True,
        )

else:

    anomaly_count = 0

    df["ML_Anomaly"] = 1

    anomaly_results = (
        df.iloc[0:0].copy()
    )

    st.warning(
        "Not enough numeric features "
        "for ML anomaly detection."
    )


# ============================================================
# QUALITY SCORE
# ============================================================
st.header(
    "7️⃣ Overall Data Quality Score"
)

total_cells = (
    rows
    * max(columns, 1)
)

quality_score = (
    build_quality_score(
        total_cells,
        missing_count,
        duplicate_count,
        invalid_count,
    )
)

status = (
    quality_status(
        quality_score
    )
)


score_col1, score_col2 = (
    st.columns(2)
)

score_col1.metric(
    "Data Quality Score",
    f"{quality_score}/100",
)

score_col2.metric(
    "Status",
    status,
)

st.caption(
    "The score is a project-defined indicator based "
    "on missing values, duplicates, and rule-based "
    "invalid values. Statistical outliers and ML "
    "anomalies are not automatically treated as errors."
)


# ============================================================
# ROOT CAUSE
# ============================================================
st.header(
    "8️⃣ Root-Cause Analysis"
)

root_causes = []


if missing_count:

    root_causes.append(
        {
            "Issue":
                "Missing values",

            "Possible Root Cause":
                "Incomplete data entry or missing "
                "source-system information.",

            "Recommended Action":
                "Validate the source and apply "
                "an appropriate imputation rule.",
        }
    )


if duplicate_count:

    root_causes.append(
        {
            "Issue":
                "Duplicate records",

            "Possible Root Cause":
                "Repeated ingestion, duplicate "
                "transactions, or repeated exports.",

            "Recommended Action":
                "Check record keys and ingestion "
                "logic before removing duplicates.",
        }
    )


if invalid_count:

    root_causes.append(
        {
            "Issue":
                "Invalid values",

            "Possible Root Cause":
                "Validation failures or incorrect "
                "source-system entries.",

            "Recommended Action":
                "Validate business rules and "
                "correct the source where possible.",
        }
    )


if city_inconsistency_count:

    root_causes.append(
        {
            "Issue":
                "City inconsistencies",

            "Possible Root Cause":
                "Different abbreviations or naming "
                "conventions across sources.",

            "Recommended Action":
                "Standardize categorical values "
                "using a controlled mapping.",
        }
    )


if total_outliers:

    root_causes.append(
        {
            "Issue":
                "Statistical outliers",

            "Possible Root Cause":
                "Legitimate extreme transactions "
                "or unusual observations.",

            "Recommended Action":
                "Investigate before removing "
                "or modifying them.",
        }
    )


if anomaly_count:

    root_causes.append(
        {
            "Issue":
                "ML anomalies",

            "Possible Root Cause":
                "Unusual combinations of "
                "numeric features.",

            "Recommended Action":
                "Review anomalous records "
                "with domain context.",
        }
    )


root_cause_df = (
    pd.DataFrame(
        root_causes
    )
)


if root_cause_df.empty:

    st.success(
        "No major rule-based issues were detected."
    )

else:

    st.dataframe(
        root_cause_df,
        use_container_width=True,
    )


# ============================================================
# GEMINI AI
# ============================================================
st.header(
    "9️⃣ Gemini AI Analysis"
)

issue_summary = f"""
Dataset: {uploaded_file.name}
Rows: {rows}
Columns: {columns}
Missing cells: {missing_count}
Duplicate records: {duplicate_count}
Invalid values: {invalid_count}
City inconsistencies: {city_inconsistency_count}
Statistical outlier flags: {total_outliers}
ML anomalies: {anomaly_count}
Date/time columns: {
    ", ".join(date_columns)
    if date_columns
    else "None"
}
Numeric columns: {
    ", ".join(numeric_columns)
    if numeric_columns
    else "None"
}
"""


if gemini_client:

    if st.button(
        "🔍 Analyze with Gemini AI",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Gemini is analyzing the detected "
            "data-quality findings..."
        ):

            ai_result = ask_gemini(
                issue_summary
            )

        st.success(
            "Gemini AI analysis completed."
        )

        st.markdown(
            ai_result
        )

else:

    st.info(
        "Add GEMINI_API_KEY in Streamlit Secrets "
        "to enable this section."
    )


# ============================================================
# CLEANING
# ============================================================
st.header(
    "🔟 Automated Data Cleaning"
)

(
    cleaned_df,
    original_rows,
    cleaned_rows,
    removed_rows,
    values_filled,
    cleaning_log,
) = clean_dataset(
    df_original
)


c1, c2, c3, c4 = (
    st.columns(4)
)

c1.metric(
    "Original Rows",
    f"{original_rows:,}",
)

c2.metric(
    "Cleaned Rows",
    f"{cleaned_rows:,}",
)

c3.metric(
    "Rows Removed",
    f"{removed_rows:,}",
)

c4.metric(
    "Values Filled",
    f"{values_filled:,}",
)


with st.expander(
    "Cleaning Actions",
    expanded=True,
):

    if cleaning_log.empty:

        st.info(
            "No cleaning actions were required."
        )

    else:

        st.dataframe(
            cleaning_log,
            use_container_width=True,
        )


st.subheader(
    "Cleaned Dataset Preview"
)

st.dataframe(
    cleaned_df.head(10),
    use_container_width=True,
    height=300,
)


# ============================================================
# EXPORT TABLES
# ============================================================
quality_issues_rows = []


if missing_count:

    missing_series = (
        df_original
        .isna()
        .sum()
    )

    for column, count in (
        missing_series[
            missing_series > 0
        ].items()
    ):

        quality_issues_rows.append(
            {
                "Issue Type":
                    "Missing Values",

                "Column":
                    column,

                "Count":
                    int(count),
            }
        )


if duplicate_count:

    quality_issues_rows.append(
        {
            "Issue Type":
                "Duplicates",

            "Column":
                "All columns",

            "Count":
                duplicate_count,
        }
    )


if invalid_count:

    for _, row in (
        invalid_report
        .iterrows()
    ):

        quality_issues_rows.append(
            {
                "Issue Type":
                    "Invalid Value",

                "Column":
                    row["Column"],

                "Count":
                    1,
            }
        )


if city_inconsistency_count:

    quality_issues_rows.append(
        {
            "Issue Type":
                "City Inconsistency",

            "Column":
                "City",

            "Count":
                city_inconsistency_count,
        }
    )


if total_outliers:

    for _, row in (
        outlier_report[
            outlier_report[
                "Outlier Count"
            ] > 0
        ]
        .iterrows()
    ):

        quality_issues_rows.append(
            {
                "Issue Type":
                    "Statistical Outlier",

                "Column":
                    row["Column"],

                "Count":
                    int(
                        row["Outlier Count"]
                    ),
            }
        )


quality_issues = (
    pd.DataFrame(
        quality_issues_rows,
        columns=[
            "Issue Type",
            "Column",
            "Count",
        ],
    )
)


if quality_issues.empty:

    quality_issues = pd.DataFrame(
        columns=[
            "Issue Type",
            "Column",
            "Count",
        ]
    )


quality_summary = pd.DataFrame(
    [
        {
            "Metric":
                "Rows",

            "Value":
                rows,
        },

        {
            "Metric":
                "Columns",

            "Value":
                columns,
        },

        {
            "Metric":
                "Missing Cells",

            "Value":
                missing_count,
        },

        {
            "Metric":
                "Duplicates",

            "Value":
                duplicate_count,
        },

        {
            "Metric":
                "Invalid Values",

            "Value":
                invalid_count,
        },

        {
            "Metric":
                "City Inconsistencies",

            "Value":
                city_inconsistency_count,
        },

        {
            "Metric":
                "Statistical Outliers",

            "Value":
                total_outliers,
        },

        {
            "Metric":
                "ML Anomalies",

            "Value":
                anomaly_count,
        },

        {
            "Metric":
                "Data Quality Score",

            "Value":
                quality_score,
        },

        {
            "Metric":
                "Original Rows",

            "Value":
                original_rows,
        },

        {
            "Metric":
                "Cleaned Rows",

            "Value":
                cleaned_rows,
        },

        {
            "Metric":
                "Rows Removed",

            "Value":
                removed_rows,
        },

        {
            "Metric":
                "Values Filled",

            "Value":
                values_filled,
        },
    ]
)


# ============================================================
# ANOMALY EXPORT
# ============================================================
anomaly_export = df.copy()


if "ML_Anomaly" in (
    anomaly_export.columns
):

    anomaly_export[
        "ML_Anomaly_Label"
    ] = np.where(
        anomaly_export[
            "ML_Anomaly"
        ] == -1,
        "Anomaly",
        "Normal",
    )


# ============================================================
# POWER BI EXPORT
# ============================================================
st.header(
    "1️⃣1️⃣ Power BI Integration"
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
    "DataGuard AI creates stable CSV filenames "
    "so a Power BI dataset can use the same source "
    "files after every upload."
)


download_col, automate_col = (
    st.columns(2)
)


with download_col:

    st.download_button(
        "📦 Download Power BI Export ZIP",
        data=zip_bytes,
        file_name=(
            "DataGuard_AI_PowerBI_Export.zip"
        ),
        mime="application/zip",
        use_container_width=True,
    )


with automate_col:

    automation_ready = (
        not blob_config_status()
        and not powerbi_config_status()
    )

    if automation_ready:

        if st.button(
            "🚀 Sync Data + Refresh Power BI",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Uploading processed files and "
                "triggering Power BI refresh..."
            ):

                result = (
                    run_powerbi_automation(
                        export_files
                    )
                )

            if result["success"]:

                st.success(
                    "✅ "
                    + result["message"]
                )

                st.caption(
                    "Power BI refresh has been "
                    "triggered. The dataset must "
                    "be configured to read these "
                    "stable Azure Blob source files."
                )

            else:

                st.error(
                    "❌ "
                    + result["message"]
                )

    else:

        st.warning(
            "Automatic Power BI sync is not "
            "configured yet."
        )

        missing_blob = (
            blob_config_status()
        )

        missing_pbi = (
            powerbi_config_status()
        )

        if missing_blob:

            st.caption(
                "Missing Azure configuration: "
                + ", ".join(
                    missing_blob
                )
            )

        if missing_pbi:

            st.caption(
                "Missing Power BI configuration: "
                + ", ".join(
                    missing_pbi
                )
            )


st.info(
    "For true automatic dashboard updates, "
    "Power BI Desktop must first be configured "
    "to read the stable CSV files from Azure Blob "
    "Storage. After that one-time setup, DataGuard "
    "AI can overwrite those files and trigger the "
    "Power BI dataset refresh."
)


# ============================================================
# REPORT
# ============================================================
st.header(
    "1️⃣2️⃣ Data Quality Report"
)

report_text = f"""
# DataGuard AI — Data Quality Report

## Dataset
{uploaded_file.name}

## Profile
- Rows: {rows}
- Columns: {columns}
- Missing cells: {missing_count}
- Duplicates: {duplicate_count}

## Quality Issues
- Invalid values: {invalid_count}
- City inconsistencies: {city_inconsistency_count}
- Statistical outliers: {total_outliers}
- ML anomalies: {anomaly_count}

## Data Quality Score
{quality_score}/100 — {status}

## Cleaning
- Original rows: {original_rows}
- Cleaned rows: {cleaned_rows}
- Rows removed: {removed_rows}
- Values filled: {values_filled}

## Date/Time Columns
{
    ", ".join(date_columns)
    if date_columns
    else "None"
}

## ML Features
{
    ", ".join(ml_features)
    if ml_features
    else "None"
}

## Interpretation
Statistical outliers and ML anomalies are investigative
signals. They are not automatically treated as data errors
or fraud.
"""


st.text_area(
    "Report Preview",
    report_text,
    height=320,
)


st.download_button(
    "📄 Download Data Quality Report",
    data=report_text.encode(
        "utf-8"
    ),
    file_name=(
        "DataGuard_AI_Data_Quality_Report.txt"
    ),
    mime="text/plain",
    use_container_width=True,
)


# ============================================================
# FINAL SUMMARY
# ============================================================
st.header(
    "📋 Final Summary"
)

summary_cols = (
    st.columns(5)
)


summary_cols[0].metric(
    "Quality Score",
    f"{quality_score}/100",
)


summary_cols[1].metric(
    "Issues",
    f"{
        invalid_count
        + city_inconsistency_count
    :,}",
)


summary_cols[2].metric(
    "Statistical Outliers",
    f"{total_outliers:,}",
)


summary_cols[3].metric(
    "ML Anomalies",
    f"{anomaly_count:,}",
)


summary_cols[4].metric(
    "Rows Removed",
    f"{removed_rows:,}",
)


st.success(
    "DataGuard AI pipeline completed successfully: "
    "Profile → Quality Checks → Outlier Detection → "
    "ML Anomaly Detection → Root-Cause Analysis → "
    "Cleaning → Power BI Export."
)


st.caption(
    "DataGuard AI | Data Quality • Statistical Analysis • "
    "Machine Learning • Root-Cause Analysis • Power BI"
)
