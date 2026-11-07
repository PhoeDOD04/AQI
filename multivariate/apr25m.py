
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

# ==========================================
# 1. LOAD ALL 2025 MONTHLY DATASETS
# ==========================================

files = glob.glob(
    "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_*_2025_Hourly*.xlsx"
)

month_data = []

print("Files found:", len(files))

if not files:
    raise FileNotFoundError("No 2025 AQI Excel files found!")

for file in files:

    # Extract month and year from filename
    match = re.search(
        r"DPCC_(\w+)_(\d{4})_Hourly",
        os.path.basename(file)
    )

    if not match:
        print("Skipping file:", file)
        continue

    month_name = match.group(1)
    year = int(match.group(2))

    print("Processing:", month_name, year)

    df = pd.read_excel(file)

    # Select hourly columns
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

    # Convert wide data into long format
    temp = df.melt(
        id_vars=["Date"],
        var_name="Hour",
        value_name="AQI"
    )

    # Convert date into numeric
    temp["Date"] = pd.to_numeric(
        temp["Date"], errors="coerce"
    )

    # Extract hour from column names (00:00, 01:00, etc.)
    hour = pd.to_numeric(
        temp["Hour"].astype(str).str.extract(
            r"^(\d{1,2})"
        )[0],
        errors="coerce"
    )

    # Create datetime column
    temp["datetime"] = pd.to_datetime(
        dict(
            year=year,
            month=pd.to_datetime(
                month_name, format="%B"
            ).month,
            day=temp["Date"],
            hour=hour
        ),
        errors="coerce"
    )

    # Add month name
    temp["Month"] = month_name

    # IMPORTANT: Append processed data
    month_data.append(
        temp[["datetime", "Month", "AQI"]]
    )

# ==========================================
# 2. COMBINE ALL MONTHS
# ==========================================

if not month_data:
    raise ValueError("No monthly datasets were successfully processed!")

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
).reset_index(drop=True)

print("\nCombined dataset shape:", combined_df.shape)
print("Months included:", combined_df["Month"].unique())

# ==========================================
# 3. DAILY AQI STATISTICS
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

daily_stats["Month"] = daily_stats[
    "Date"
].dt.strftime("%B")

print("\nDAILY AQI STATISTICS")
print(daily_stats.to_string(index=False))

# ==========================================
# 4. DAILY AVERAGE AQI TREND
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

plt.title("Daily Average AQI Trend - Anand Vihar, 2025")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 5. DAILY MAXIMUM, MINIMUM AND AVERAGE AQI
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

plt.title("Daily Maximum, Minimum and Average AQI - 2025")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 6. DAILY AQI VARIABILITY
# ==========================================

plt.figure(figsize=(14, 5))

sns.lineplot(
    data=daily_stats,
    x="Date",
    y="Daily_Std",
    marker="o",
    label="Daily AQI Standard Deviation"
)

plt.title("Daily AQI Variability - 2025")
plt.xlabel("Date")
plt.ylabel("Standard Deviation")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 7. HOURLY AQI HEATMAP
# ==========================================


# ==========================================
# HOURLY AQI HEATMAP - IMPROVED
# ==========================================

combined_df["Date"] = combined_df["datetime"].dt.normalize()
combined_df["Hour"] = combined_df["datetime"].dt.hour

heatmap_data = combined_df.pivot_table(
    index="Date",
    columns="Hour",
    values="AQI",
    aggfunc="mean"
)

# Ensure all dates and hours are represented
date_range = pd.date_range(
    start="2025-01-01",
    end="2025-12-31",
    freq="D"
)

heatmap_data = heatmap_data.reindex(
    index=date_range,
    columns=range(24)
)

# Plot heatmap
plt.figure(figsize=(17, 12))

sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    vmin=0,
    vmax=500,
    linewidths=0,
    cbar_kws={
        "label": "AQI",
        "shrink": 0.8
    }
)

plt.title(
    "Hourly AQI Variation Across Days - Anand Vihar, 2025",
    fontsize=15
)

plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Date", fontsize=12)

plt.xticks(
    np.arange(24) + 0.5,
    range(24)
)

# Show monthly boundaries
for month_start in pd.date_range(
    "2025-02-01", "2025-12-01", freq="MS"
):
    row_position = (month_start - date_range[0]).days
    plt.axhline(
        row_position,
        color="black",
        linewidth=0.8
    )

# Label months
month_positions = [
    (pd.Timestamp(f"2025-{m:02d}-01") - date_range[0]).days
    for m in range(1, 13)
]

plt.yticks(
    [p + 14 for p in month_positions],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    rotation=0
)

plt.tight_layout()
plt.show()

# ==========================================
# 8. MONTHLY DAILY AQI COMPARISON
# ==========================================

daily_stats["Day"] = daily_stats[
    "Date"
].dt.day

month_order = [
    "January", "February", "March",
    "April", "May", "June",
    "July", "August", "September",
    "October", "November", "December"
]

daily_stats["Month"] = pd.Categorical(
    daily_stats["Month"],
    categories=month_order,
    ordered=True
)

plt.figure(figsize=(13, 6))

sns.lineplot(
    data=daily_stats,
    x="Day",
    y="Daily_Mean",
    hue="Month",
    marker="o"
)

plt.title("Daily Average AQI Comparison Across Months - 2025")
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
# 9. MONTHLY AQI DISTRIBUTION
# ==========================================

plt.figure(figsize=(13, 6))

sns.boxplot(
    data=combined_df,
    x="Month",
    y="AQI",
    order=month_order
)

plt.title("Monthly AQI Distribution - 2025")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 10. HOURLY AQI PROFILE
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

plt.title("Average Hourly AQI Profile - 2025")
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")
plt.xticks(range(24))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 11. SAVE DAILY STATISTICS
# ==========================================

daily_stats.to_excel(
    "AQI_2025_Daily_Statistics.xlsx",
    index=False
)

print("\nDaily statistics exported successfully!")
