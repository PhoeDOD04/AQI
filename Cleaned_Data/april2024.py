
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_April_2024_Hourly.xlsx")

# Combine all hourly AQI readings
aqi = df.drop(columns=["Date"]).values.flatten()
aqi = pd.Series(aqi).dropna()

# Descriptive statistics
print("Count:", aqi.count())
print("Missing values:", df.drop(columns=["Date"]).isna().sum().sum())
print("Mean:", aqi.mean())
print("Median:", aqi.median())
print("Mode:", aqi.mode().tolist())
print("Standard Deviation:", aqi.std())
print("Variance:", aqi.var())
print("Minimum:", aqi.min())
print("Maximum:", aqi.max())
print("Q1:", aqi.quantile(0.25))
print("Q3:", aqi.quantile(0.75))
print("Range:", aqi.max() - aqi.min())
print("IQR:", aqi.quantile(0.75) - aqi.quantile(0.25))
print("Skewness:", aqi.skew())
print("Kurtosis:", aqi.kurt())
print("CV:", (aqi.std() / aqi.mean()) * 100)

# Outlier detection
Q1 = aqi.quantile(0.25)
Q3 = aqi.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = aqi[(aqi < lower) | (aqi > upper)]

print("Lower Bound:", lower)
print("Upper Bound:", upper)
print("Number of Outliers:", len(outliers))

# Histogram and KDE
plt.figure(figsize=(9, 5))
sns.histplot(aqi, kde=True, bins=20)
plt.axvline(aqi.mean(), linestyle="--", label="Mean")
plt.axvline(aqi.median(), linestyle=":", label="Median")
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.legend()
plt.show()

# Box plot
plt.figure(figsize=(9, 4))
sns.boxplot(x=aqi)
plt.title("AQI Box Plot")
plt.xlabel("AQI")
plt.show()

# Distribution and skewness
plt.figure(figsize=(9, 5))
sns.histplot(aqi, kde=True, stat="density", bins=20)
plt.title("AQI Distribution and Skewness")
plt.xlabel("AQI")
plt.show()
