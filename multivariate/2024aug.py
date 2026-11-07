
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

# ==========================================
# 1. LOAD ALL 2024 MONTHLY DATASETS
# ==========================================

files = glob.glob(
    "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_*_2024_Hourly*.xlsx"
)

print("Files found:", len(files))

if not files:
    raise FileNotFoundError(
        "No 2024 AQI Excel files found. Check filenames and folder."
    )

month_data = []
loaded_months = set()

for file in sorted(files):

    # Extract month and year from filename
    match = re.search(
        r"DPCC_(\w+)_(\d{4})_Hourly",
        os.path.basename(file)
    )

    if not match:
        print("Skipping:", file)
        continue

    month_name = match.group(1)
    year = int(match.group(2))

    if year != 2024:
        continue

    month_number = pd.to_datetime(
        month_name, format="%B"
    ).month

    # Avoid duplicate files for the same month
    if month_number in loaded_months:
        print("Duplicate month skipped:", file)
        continue

    print("Processing:", month_name, year)

    df = pd.read_excel(file)

    if "Date" not in df.columns:
        print("Skipping file without Date column:", file)
        continue

    # Select hourly AQI columns
    hour_cols = df.columns.drop("Date")

    # Handle missing values
    df[hour_cols] = df[hour_cols].replace(
        ["NA", "N/A", "NaN", "nan", "NULL",
         "null", "None", ""],
        np.nan
    )

    # Convert AQI readings to numeric
    df[hour_cols] = df[hour_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Convert wide format into long format
    temp = df.melt(
        id_vars=["Date"],
        var_name="Hour",
        value_name="AQI"
    )

    # Convert date into numeric
    temp["Date"] = pd.to_numeric(
        temp["Date"], errors="coerce"
    )

    # Extract hour from column headers
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
            month=month_number,
            day=temp["Date"],
            hour=hour
        ),
        errors="coerce"
    )

    # Store processed monthly data
    month_data.append(
        temp[["datetime", "AQI"]]
    )

    loaded_months.add(month_number)


# ==========================================
# 2. COMBINE ALL MONTHS
# ==========================================

if not month_data:
    raise ValueError(
        "No monthly datasets were successfully processed."
    )

combined_df = pd.concat(
    month_data,
    ignore_index=True
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

# Extract date, month and hour
combined_df["Date"] = combined_df[
    "datetime"
].dt.normalize()

combined_df["Month"] = combined_df[
    "datetime"
].dt.month_name()

combined_df["Hour"] = combined_df[
    "datetime"
].dt.hour

print("\nCOMBINED DATASET")
print("----------------")
print("Shape:", combined_df.shape)
print("Months loaded:", sorted(loaded_months))
print("Date range:",
      combined_df["datetime"].min(),
      "to",
      combined_df["datetime"].max())


# ==========================================
# 3. DAILY AQI STATISTICS
# ==========================================

daily_stats = combined_df.groupby("Date").agg(
    Daily_Mean=("AQI", "mean"),
    Daily_Max=("AQI", "max"),
    Daily_Min=("AQI", "min"),
    Daily_Std=("AQI", "std"),
    Valid_Readings=("AQI", "count")
).reset_index()

daily_stats["Month"] = daily_stats[
    "Date"
].dt.month_name()

daily_stats["Day"] = daily_stats[
    "Date"
].dt.day

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
    linewidth=1.8,
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
# 5. DAILY MAXIMUM, MINIMUM AND AVERAGE
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

plt.title("Daily Maximum, Minimum and Average AQI - 2024")
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

plt.title("Daily AQI Variability - 2024")
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

heatmap_data = combined_df.pivot_table(
    index="Date",
    columns="Hour",
    values="AQI",
    aggfunc="mean"
)

# Include all dates; unavailable dates remain blank
date_range = pd.date_range(
    "2024-01-01",
    "2024-12-31",
    freq="D"
)

heatmap_data = heatmap_data.reindex(
    index=date_range,
    columns=range(24)
)

plt.figure(figsize=(16, 11))

sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    vmin=0,
    vmax=500,
    annot=False,
    cbar_kws={"label": "AQI"}
)

plt.title("Hourly AQI Variation Across Days - 2024")
plt.xlabel("Hour of Day")
plt.ylabel("Date")

month_positions = [
    (pd.Timestamp(f"2024-{m:02d}-01") - date_range[0]).days + 14
    for m in range(1, 13)
]

plt.yticks(
    month_positions,
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    rotation=0
)

plt.xticks(np.arange(24) + 0.5, range(24))

plt.tight_layout()
plt.show()


# ==========================================
# 8. MONTHLY DAILY AQI COMPARISON
# ==========================================

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
    hue_order=month_order,
    marker="o"
)

plt.title("Daily Average AQI Comparison Across Months - 2024")
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
# 9. MONTHLY AQI DISTRIBUTION - BOX PLOT
# ==========================================

combined_df["Month"] = pd.Categorical(
    combined_df["Month"],
    categories=month_order,
    ordered=True
)

plt.figure(figsize=(13, 6))

sns.boxplot(
    data=combined_df,
    x="Month",
    y="AQI",
    order=month_order,
    showmeans=True
)

plt.title("Monthly AQI Distribution - 2024")
plt.xlabel("Month")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 10. MONTHLY AVERAGE AQI
# ==========================================

monthly_stats = combined_df.groupby(
    "Month", observed=False
).agg(
    Monthly_Mean=("AQI", "mean"),
    Monthly_Max=("AQI", "max"),
    Monthly_Min=("AQI", "min"),
    Monthly_Std=("AQI", "std"),
    Valid_Readings=("AQI", "count")
).reindex(month_order).dropna(
    subset=["Monthly_Mean"]
).reset_index()

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=monthly_stats,
    x="Month",
    y="Monthly_Mean",
    marker="o",
    linewidth=2,
    label="Monthly Average AQI"
)

plt.title("Monthly Average AQI Variation - 2024")
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 11. HOURLY AQI PROFILE
# ==========================================

hourly_profile = combined_df.groupby(
    "Hour"
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
# 12. EXPORT STATISTICS TO EXCEL
# ==========================================

with pd.ExcelWriter("AQI_2024_Statistics.xlsx") as writer:

    daily_stats.to_excel(
        writer,
        sheet_name="Daily Statistics",
        index=False
    )

    monthly_stats.to_excel(
        writer,
        sheet_name="Monthly Statistics",
        index=False
    )

print("\n2024 AQI analysis completed!")
print("Statistics saved to AQI_2024_Statistics.xlsx")
