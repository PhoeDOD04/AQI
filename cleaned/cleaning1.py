import pandas as pd


july_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_July_2024_Hourly.xlsx")
june_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_June_2024_Hourly.xlsx")
march_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_March_2024_Hourly.xlsx")
may_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_May_2024_Hourly.xlsx")
november_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_November_2024_Hourly.xlsx")
october_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_October_2024_Hourly.xlsx")
december_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_December_2024_Hourly.xlsx")
january_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_January_2024_Hourly.xlsx")
february_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_February_2024_Hourly.xlsx")
april_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_April_2024_Hourly.xlsx")
august_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_August_2024_Hourly.xlsx")
september_2024 = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_September_2024_Hourly.xlsx")
datasets = {
    1: january_2024,
    2: february_2024,
    3: march_2024,
    4: april_2024,
    5: may_2024,
    6: june_2024,
    7: july_2024,
    8: august_2024,
    9: september_2024,
    10: october_2024,
    11: november_2024,
    12: december_2024
}

data = []

for month, df in datasets.items():

    temp = df.melt(
        id_vars=["Date"],
        var_name="Hour",
        value_name="AQI"
    )

    temp["datetime"] = pd.to_datetime(
        "2024-" + str(month) + "-" +
        temp["Date"].astype(str) + " " +
        temp["Hour"].astype(str),
        errors="coerce"
    )

    data.append(temp[["datetime", "AQI"]])

combined_df = pd.concat(data, ignore_index=True)

combined_df["AQI"] = pd.to_numeric(
    combined_df["AQI"], errors="coerce"
)

combined_df = combined_df.dropna()

combined_df = combined_df.sort_values(
    "datetime"
).reset_index(drop=True)
combined_df["AQI_lag1"] = combined_df["AQI"].shift(1)

combined_df["AQI_lag24"] = combined_df["AQI"].shift(24)

combined_df["hour"] = combined_df["datetime"].dt.hour

combined_df["month"] = combined_df["datetime"].dt.month

# Target: AQI of the next hour
combined_df["target"] = combined_df["AQI"].shift(-1)

combined_df = combined_df.dropna()
X = combined_df[
    ["AQI", "AQI_lag1", "AQI_lag24", "hour", "month"]
]

y = combined_df["target"]
split = int(len(X) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

with pd.ExcelWriter("AQI_2024_Train_Test_Split.xlsx") as writer:

    X_train.to_excel(writer, sheet_name="X_train", index=False)

    X_test.to_excel(writer, sheet_name="X_test", index=False)

    y_train.to_frame().to_excel(writer, sheet_name="y_train", index=False)

    y_test.to_frame().to_excel(writer, sheet_name="y_test", index=False)
import numpy as np

missing_values = ["NA", "N/A", "NaN", "nan",
                  "NULL", "null", "None", "none", ""]

for name, df in datasets.items():

    print("\nDataset:", name)

    # Detect actual NaN values and text-based missing values
    missing_mask = (
        df.isna() |
        df.astype(str).apply(
            lambda col: col.str.strip().str.upper().isin(
                ["NA", "N/A", "NAN", "NULL", "NONE", ""]
            )
        )
    )

    missing_mask = (
    combined_df.isna() |
    combined_df.astype(str).apply(
        lambda col: col.str.strip().str.upper().isin(
            ["NA", "N/A", "NAN", "NULL", "NONE", ""]
        )
    )
)

print("Missing values in each column:")
print(missing_mask.sum())

print("\nTotal missing values:")
print(missing_mask.sum().sum())

print("\nRows containing missing values:")
print(combined_df[missing_mask.any(axis=1)])
# Drop rows where all values are missing (NaN or NA)
X_train = X_train.dropna(how="all").reset_index(drop=True)
X_test = X_test.dropna(how="all").reset_index(drop=True)

y_train = y_train.dropna(how="all").reset_index(drop=True)
y_test = y_test.dropna(how="all").reset_index(drop=True)

