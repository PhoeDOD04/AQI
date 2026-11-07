
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

# ==========================================
# 1. LOAD ALL MONTHLY DATASETS
# ==========================================

files = glob.glob(
     "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_April_2024_Hourly.xlsx"
)

month_data = []

for file in files:

    # Extract month from filename
    month_name = re.search(
        r"DPCC_(\w+)_2024_Hourly",
        os.path.basename(file)
    ).group(1)

    df = pd.read_excel(file)

    # Select hourly columns
    hour_cols = df.columns.drop("Date")

    # Handle missing values
    df[hour_cols] = df[hour_cols].replace(
        ["NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""],
        np.nan
    )

    df[hour_cols] = df[hour_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Convert wide data into long format
    temp = df.melt(
        id_vars=["Date"],
        var_name="Hour",
        value_name="AQI"
    )

    # Convert date and hour into datetime
    temp["Date"] = pd.to_numeric(
        temp["Date"], errors="coerce"
    )

    temp["datetime"] = pd.to_datetime(
        dict(
            year=2024,
            month=pd.to_datetime(month_name, format="%B").month,
            day=temp["Date"],
            hour=pd.to_datetime(
                temp["Hour"], format="%H:%M:%S"
            ).dt.hour
        ),
        errors="coerce"
    )

    temp["Month"] = month_name

    month_data.append(
        temp[["datetime", "Month", "AQI"]]
    )

# Combine all months
combined_df = pd.concat(
    month_data, ignore_index=True
)

combined_df["AQI"] = pd.to_numeric(
    combined_df["AQI"], errors="coerce"
)

combined_df = combined_df.dropna(
    subset=["datetime"]
)

combined_df = combined_df.sort_values(
    "datetime"
)

# ==========================================
# 2. DAILY AQI STATISTICS
# ==========================================

daily_stats = combined_df.groupby(
    combined_df["datetime"].dt.date
).agg(
    Daily_Mean=("AQI", "mean"),
    Daily_Max=("AQI", "max"),
    Daily_Min=("AQI", "min"),
    Daily_Std=("AQI", "std"),
    Valid_Readings=("AQI", "count")
).reset_index()

daily_stats.rename(
    columns={"datetime": "Date"},
    inplace=True
)

daily_stats["Date"] = pd.to_datetime(
    daily_stats["Date"]
)

daily_stats["Month"] = daily_stats["Date"].dt.strftime(
    "%B"
)

print("\nDAILY AQI STATISTICS")
print(daily_stats.to_string(index=False))

# ==========================================
# 3. DAILY AVERAGE AQI TREND
# ==========================================

plt.figure(figsize=(14, 6))

sns.lineplot(
    data=daily_stats,
    x="Date",
    y="Daily_Mean",
    marker="o",
    linewidth=2,
    label="Daily Average AQI"
)

plt.title("Daily Average AQI Trend - Anand Vihar, 2024")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 4. DAILY MAXIMUM AND MINIMUM AQI
# ==========================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_stats["Date"],
    daily_stats["Daily_Max"],
    marker="o",
    label="Daily Maximum AQI"
)

plt.plot(
    daily_stats["Date"],
    daily_stats["Daily_Min"],
    marker="o",
    label="Daily Minimum AQI"
)

plt.plot(
    daily_stats["Date"],
    daily_stats["Daily_Mean"],
    linestyle="--",
    linewidth=2,
    label="Daily Average AQI"
)

plt.title("Daily Maximum, Minimum and Average AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. DAILY AQI VARIABILITY
# ==========================================

plt.figure(figsize=(14, 5))

sns.lineplot(
    data=daily_stats,
    x="Date",
    y="Daily_Std",
    marker="o",
    label="Daily AQI Standard Deviation"
)

plt.title("Daily AQI Variability")
plt.xlabel("Date")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 6. HOURLY AQI HEATMAP
# ==========================================

combined_df["Day"] = combined_df[
    "datetime"
].dt.strftime("%Y-%m-%d")

combined_df["Hour"] = combined_df[
    "datetime"
].dt.hour

heatmap_data = combined_df.pivot_table(
    index="Day",
    columns="Hour",
    values="AQI",
    aggfunc="mean"
)

plt.figure(figsize=(16, 10))

sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    annot=False,
    linewidths=0.2,
    cbar_kws={"label": "AQI"}
)

plt.title("Hourly AQI Variation Across Days - 2024")
plt.xlabel("Hour of Day")
plt.ylabel("Date")
plt.tight_layout()
plt.show()

# ==========================================
# 7. MONTHLY DAILY AQI COMPARISON
# ==========================================

daily_stats["Day"] = daily_stats[
    "Date"
].dt.day

plt.figure(figsize=(13, 6))

sns.lineplot(
    data=daily_stats,
    x="Day",
    y="Daily_Mean",
    hue="Month",
    marker="o"
)

plt.title("Daily Average AQI Comparison Across Months")
plt.xlabel("Day of Month")
plt.ylabel("Average AQI")
plt.legend(
    title="Month",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 8. MONTHLY AQI DISTRIBUTION
# ==========================================

plt.figure(figsize=(13, 6))

sns.boxplot(
    data=combined_df,
    x="Month",
    y="AQI",
    order=[
        "January", "February", "March",
        "April", "May", "June",
        "July", "August", "September",
        "October", "November", "December"
    ]
)

plt.title("Monthly AQI Distribution - 2024")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.legend(
    handles=[],
    labels=[],
    title="AQI distribution by month"
)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 9. HOURLY AQI PROFILE
# ==========================================

hourly_profile = combined_df.groupby(
    combined_df["datetime"].dt.hour
)["AQI"].mean()

plt.figure(figsize=(12, 5))

plt.plot(
    hourly_profile.index,
    hourly_profile.values,
    marker="o",
    linewidth=2,
    label="Average AQI by Hour"
)

plt.title("Average Hourly AQI Profile - 2024")
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")
plt.xticks(range(24))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 10. SAVE DAILY STATISTICS
# ==========================================

daily_stats.to_excel(
    "AQI_2024_Daily_Statistics.xlsx",
    index=False
)

print("\nDaily statistics exported successfully!")
