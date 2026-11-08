
import pandas as pd
import numpy as np

# 1. Load original dataset
df = pd.read_excel(
    "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_December_2025_Hourly.xlsx"
)

# 2. Identify hourly columns
hour_cols = df.columns.drop("Date")

# 3. Detect and standardize missing values
df[hour_cols] = df[hour_cols].replace(
    ["NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""],
    np.nan
)

# Convert hourly readings to numeric
df[hour_cols] = df[hour_cols].apply(
    pd.to_numeric, errors="coerce"
)

# 4. Calculate missing percentage for each row
df["Missing_Count"] = df[hour_cols].isna().sum(axis=1)

df["Missing_Percentage"] = (
    df["Missing_Count"] / len(hour_cols)
) * 100

# 5. Remove rows with more than 70% missing values
removed_rows = df[df["Missing_Percentage"] > 70].copy()

cleaned_df = df[df["Missing_Percentage"] <= 70].copy()

# Remove temporary columns
cleaned_df = cleaned_df.drop(
    columns=["Missing_Count", "Missing_Percentage"]
)

# 6. Sort data chronologically
cleaned_df = cleaned_df.sort_values(
    "Date"
).reset_index(drop=True)

# 7. Split data into 80:20 ratio
split_index = int(len(cleaned_df) * 0.8)

train = cleaned_df.iloc[:split_index].copy()
test = cleaned_df.iloc[split_index:].copy()


train_medians = train[hour_cols].median()


train[hour_cols] = train[hour_cols].fillna(train_medians)

test[hour_cols] = test[hour_cols].fillna(train_medians)



# 11. Export results
with pd.ExcelWriter("AQI_December_2025_Cleaned.xlsx") as writer:

    train.to_excel(
        writer, sheet_name="Train", index=False
    )

    test.to_excel(
        writer, sheet_name="Test", index=False
    )

    removed_rows.to_excel(
        writer, sheet_name="Removed Rows", index=False
    )

    train_medians.to_frame(
        name="Median AQI"
    ).to_excel(writer, sheet_name="Training Medians")

print("\nData cleaning and splitting completed successfully!")
