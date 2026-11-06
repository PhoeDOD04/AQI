
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel(
    "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_October_2024_Hourly.xlsx"
)

# Select hourly AQI columns
hour_cols = df.columns.drop("Date")

# Handle missing values
df[hour_cols] = df[hour_cols].replace(
    ["NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""],
    np.nan
)

# Convert AQI readings to numeric
df[hour_cols] = df[hour_cols].apply(
    pd.to_numeric, errors="coerce"
)

# ==========================================
# 2. CONVERT WIDE DATA INTO LONG FORMAT
# ==========================================

aqi_data = df.melt(
    id_vars=["Date"],
    var_name="Hour",
    value_name="AQI"
)

aqi_data["AQI"] = pd.to_numeric(
    aqi_data["AQI"], errors="coerce"
)

# Remove missing readings from analysis
aqi_data = aqi_data.dropna(
    subset=["AQI"]
)

# ==========================================
# 3. CALCULATE IQR
# ==========================================

Q1 = aqi_data["AQI"].quantile(0.25)
Q3 = aqi_data["AQI"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

print("IQR OUTLIER ANALYSIS")
print("--------------------")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# ==========================================
# 4. IDENTIFY OUTLIERS
# ==========================================

outliers = aqi_data[
    (aqi_data["AQI"] < lower_bound) |
    (aqi_data["AQI"] > upper_bound)
].copy()

# Add outlier classification
outliers["Outlier_Type"] = np.where(
    outliers["AQI"] < lower_bound,
    "Lower Outlier",
    "Upper Outlier"
)

# Add IQR boundaries
outliers["Lower_Bound"] = lower_bound
outliers["Upper_Bound"] = upper_bound

# Arrange columns
outliers = outliers[
    [
        "Date",
        "Hour",
        "AQI",
        "Outlier_Type",
        "Lower_Bound",
        "Upper_Bound"
    ]
]


# ==========================================
# 6. EXPORT OUTLIERS TO EXCEL
# ==========================================

summary = pd.DataFrame({
    "Metric": [
        "Total Valid Readings",
        "Q1",
        "Q3",
        "IQR",
        "Lower Bound",
        "Upper Bound",
        "Total Outliers",
        "Lower Outliers",
        "Upper Outliers"
    ],
    "Value": [
        len(aqi_data),
        Q1,
        Q3,
        IQR,
        lower_bound,
        upper_bound,
        len(outliers),
        (outliers["Outlier_Type"] == "Lower Outlier").sum(),
        (outliers["Outlier_Type"] == "Upper Outlier").sum()
    ]
})

with pd.ExcelWriter(
    "AQI_October_2024_Outliers.xlsx"
) as writer:

    outliers.to_excel(
        writer,
        sheet_name="Detected Outliers",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="IQR Summary",
        index=False
    )

print("\nOutlier analysis completed!")
print("Outliers exported to AQI_October_2024_Outliers.xlsx")
