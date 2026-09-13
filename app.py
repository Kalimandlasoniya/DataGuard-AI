# ============================================================
# POWER BI EXPORT
# ============================================================

st.header(
    "1️⃣1️⃣ Power BI Export"
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
