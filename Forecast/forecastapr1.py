import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# ---------------------------------------------------------
# 1. Names of ALL cleaned workbooks supplied
# Keep these Excel files in the same folder as this script.
# ---------------------------------------------------------
input_files = [
    "AQI_December_2024_Cleaned.xlsx",
    "AQI_December_2025_Cleaned.xlsx",
    "AQI_February_2024_Cleaned.xlsx",
    "AQI_February_2025_Cleaned.xlsx",
    "AQI_February_2026_Cleaned.xlsx",
    "AQI_January_2024_Cleaned.xlsx",
    "AQI_January_2025_Cleaned.xlsx",
    "AQI_January_2026_Cleaned.xlsx",
    "AQI_March_2024_Cleaned.xlsx",
    "AQI_March_2025_Cleaned.xlsx",
    "AQI_March_2026_Cleaned.xlsx",
    "AQI_April_2026_Cleaned.xlsx",
    "AQI_April_2025_Cleaned.xlsx",
    "AQI_April_2024_Cleaned.xlsx",
    "AQI_May_2026_Cleaned.xlsx",
    "AQI_May_2024_Cleaned.xlsx",
    "AQI_May_2025_Cleaned.xlsx",
    "AQI_June_2026_Cleaned.xlsx",
    "AQI_June_2025_Cleaned.xlsx",
    "AQI_June_2024_Cleaned.xlsx",
    "AQI_July_2024_Cleaned.xlsx",
    "AQI_July_2025_Cleaned.xlsx",
    "AQI_July_2026_Cleaned.xlsx",
    "AQI_August_2024_Cleaned.xlsx",
    "AQI_August_2025_Cleaned.xlsx",
    "AQI_August_2026_Cleaned.xlsx",
    "AQI_September_2024_Cleaned.xlsx",
    "AQI_September_2025_Cleaned.xlsx",
    "AQI_September_2026_Cleaned.xlsx",
    "AQI_October_2024_Cleaned.xlsx",
    "AQI_October_2025_Cleaned.xlsx",
    "AQI_October_2026_Cleaned.xlsx",
    "AQI_November_2025_Cleaned.xlsx",
    "AQI_November_2024_Cleaned.xlsx"
]

# Month names are taken from the workbook filenames.
month_numbers = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April":4,
    "May":5,
    "June":6,
    "July":7,
    "August":8,
    "September":9,
    "October":10,
    "November":11,
    "December": 12
}

def convert_to_prophet_format(wide_df, file_name):
    # Extract month and year from names such as AQI_January_2024_Cleaned.xlsx
    parts = file_name.split("_")
    month_name = parts[1]
    year = int(parts[2])

    month = month_numbers[month_name]

    # All columns other than Date are hourly AQI readings.
    hour_cols = [col for col in wide_df.columns if col != "Date"]

    long_df = wide_df.melt(
        id_vars=["Date"],
        value_vars=hour_cols,
        var_name="Hour",
        value_name="y"
    )

    # Convert values such as 00:00 and 00:00:00 into hour numbers.
    long_df["Hour"] = long_df["Hour"].astype(str).str.slice(0, 2).astype(int)

    # Date column contains day-of-month numbers.
    long_df["ds"] = pd.to_datetime({
        "year": year,
        "month": month,
        "day": pd.to_numeric(long_df["Date"], errors="coerce"),
        "hour": long_df["Hour"]
    }, errors="coerce")

    long_df["y"] = pd.to_numeric(long_df["y"], errors="coerce")

    return (
        long_df[["ds", "y"]]
        .dropna(subset=["ds", "y"])
        .sort_values("ds")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# 2. Read and convert every workbook
# ---------------------------------------------------------
all_months = []

for file_name in input_files:
    print(f"Reading: {file_name}")

    # Combine the Train and Test sheets so all cleaned observations
    # from each workbook are available for one chronological split.
    train_wide = pd.read_excel(file_name, sheet_name="Train")
    test_wide = pd.read_excel(file_name, sheet_name="Test")

    train_long = convert_to_prophet_format(train_wide, file_name)
    test_long = convert_to_prophet_format(test_wide, file_name)

    month_data = pd.concat(
        [train_long, test_long],
        ignore_index=True
    )

    all_months.append(month_data)


# Combine all months and years into one time series.
data = (
    pd.concat(all_months, ignore_index=True)
    .drop_duplicates(subset=["ds"], keep="last")
    .sort_values("ds")
     .reset_index(drop=True)
)

# Save the complete Prophet-ready dataset.
data.to_excel("AQI_Prophet_Ready.xlsx", index=False)

print("\nTotal observations:", len(data))
print("Date range:", data["ds"].min(), "to", data["ds"].max())


# ---------------------------------------------------------
# 3. Chronological 80:20 train-test split
# ---------------------------------------------------------
split_index = int(len(data) * 0.80)

train = data.iloc[:split_index].copy()
test = data.iloc[split_index:].copy()

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))
print("Training ends:", train["ds"].max())
print("Testing starts:", test["ds"].min())


# ---------------------------------------------------------
# 4. Train Prophet
# ---------------------------------------------------------
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    interval_width=0.95,    
    uncertainty_samples=1000
)

model.fit(train[["ds", "y"]])


# ---------------------------------------------------------
# 5. Predict test values and calculate accuracy
# ---------------------------------------------------------
test_forecast = model.predict(test[["ds"]])

results = test.merge(
    test_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
    on="ds",
    how="left"
)

mae = mean_absolute_error(results["y"], results["yhat"])
rmse = np.sqrt(mean_squared_error(results["y"], results["yhat"]))

print("\nTEST RESULTS")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

print("\nActual vs Predicted AQI with bounds:")
print(results[
    ["ds", "y", "yhat", "yhat_lower", "yhat_upper"]
].head(20))

results.to_excel("AQI_Test_Predictions_With_Bounds.xlsx", index=False)


# ---------------------------------------------------------
# 6. Plot actual vs predicted AQI
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(results["ds"], results["y"], label="Actual AQI")
plt.plot(results["ds"], results["yhat"], label="Predicted AQI")
plt.fill_between(
    results["ds"],
    results["yhat_lower"],
    results["yhat_upper"],
    alpha=0.2,
    label="95% uncertainty interval"
)

plt.title("Actual vs Predicted AQI - Prophet")
plt.xlabel("Timestamp")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 7. Plot trend and seasonality components
# ---------------------------------------------------------
model.plot(test_forecast)
plt.title("Prophet Forecast")
plt.tight_layout()
plt.show()

model.plot_components(test_forecast)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 8. Retrain on all available observations and forecast
#    the next 24 hours after the latest timestamp
# ---------------------------------------------------------
final_model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=False,
    interval_width=0.95,
    uncertainty_samples=1000
)

final_model.fit(data[["ds", "y"]])

future = pd.DataFrame({
    "ds": pd.date_range(
        start=data["ds"].max() + pd.Timedelta(hours=1),
        periods=24,
        freq="h"
    )
})

future_forecast = final_model.predict(future)

future_results = future_forecast[
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
]

future_results.to_excel(
    "AQI_Next_24_Hours_Forecast.xlsx",
    index=False
)

print("\nNEXT 24 HOURS FORECAST")
print(future_results)

print("\nFiles generated:")
print("1. AQI_Prophet_Ready_All_Years.xlsx")
print("2. AQI_Test_Predictions_With_Bounds.xlsx")
print("3. AQI_Next_24_Hours_Forecast.xlsx")
