
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_excel(
    "AQI_Data_Station_Level_Site_301_Anand_Vihar_Delhi_DPCC_April_2026_Hourly.xlsx"
)

# Select hourly AQI columns
hour_cols = df.columns.drop("Date")

# Handle missing values
df[hour_cols] = df[hour_cols].replace(
    ["NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""],
    np.nan
)

# Convert hourly readings to numeric
df[hour_cols] = df[hour_cols].apply(
    pd.to_numeric, errors="coerce"
)

# Combine hourly AQI readings into one series
aqi = pd.Series(df[hour_cols].values.flatten()).dropna()

# ==========================================
# 2. MISSING VALUE ANALYSIS
# ==========================================

total = df[hour_cols].size
missing = df[hour_cols].isna().sum().sum()
valid = aqi.count()

print("\nMISSING VALUE ANALYSIS")
print("----------------------")
print("Total readings:", total)
print("Valid readings:", valid)
print("Missing readings:", missing)
print("Missing percentage:", round(missing / total * 100, 2), "%")

# ==========================================
# 3. DESCRIPTIVE STATISTICS
# ==========================================

Q1 = aqi.quantile(0.25)
Q2 = aqi.quantile(0.50)
Q3 = aqi.quantile(0.75)

IQR = Q3 - Q1

print("\nDESCRIPTIVE STATISTICS")
print("----------------------")

print("Count:", aqi.count())
print("Mean:", round(aqi.mean(), 2))
print("Median:", round(aqi.median(), 2))
print("Mode:", aqi.mode().tolist())
print("Standard Deviation:", round(aqi.std(), 2))
print("Variance:", round(aqi.var(), 2))
print("Minimum:", aqi.min())
print("Maximum:", aqi.max())
print("Q1:", Q1)
print("Q2:", Q2)
print("Q3:", Q3)
print("Range:", aqi.max() - aqi.min())
print("IQR:", IQR)
print("Skewness:", round(aqi.skew(), 3))
print("Kurtosis:", round(aqi.kurt(), 3))
print("Coefficient of Variation:",
      round(aqi.std() / aqi.mean() * 100, 2), "%")

# ==========================================
# 4. OUTLIER DETECTION USING IQR
# ==========================================

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = aqi[
    (aqi < lower_bound) | (aqi > upper_bound)
]

print("\nOUTLIER ANALYSIS")
print("----------------")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Outliers:", len(outliers))
print("Outlier Values:", outliers.tolist())

# ==========================================
# 5. HISTOGRAM WITH KDE
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    aqi,
    bins=20,
    kde=True,
    color="skyblue",
    edgecolor="black",
    label="AQI Frequency Distribution"
)

plt.axvline(
    aqi.mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean = {aqi.mean():.2f}"
)

plt.axvline(
    aqi.median(),
    color="green",
    linestyle=":",
    linewidth=2,
    label=f"Median = {aqi.median():.2f}"
)

plt.title("Histogram of Hourly AQI - April 2026")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ==========================================
# 6. BOX PLOT
# ==========================================

plt.figure(figsize=(10, 4))

sns.boxplot(
    x=aqi,
    color="lightblue",
    showmeans=True,
    meanprops={
        "marker": "D",
        "markerfacecolor": "red",
        "markeredgecolor": "black"
    }
)

plt.axvline(
    lower_bound,
    color="red",
    linestyle="--",
    label=f"Lower Bound = {lower_bound:.2f}"
)

plt.axvline(
    upper_bound,
    color="green",
    linestyle="--",
    label=f"Upper Bound = {upper_bound:.2f}"
)

plt.title("Box Plot of Hourly AQI - April 2026")
plt.xlabel("AQI")
plt.legend()
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()
plt.show()

# ==========================================
# 7. DISTRIBUTION AND SKEWNESS
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    aqi,
    bins=20,
    stat="density",
    color="skyblue",
    alpha=0.4,
    edgecolor="black",
    label="Observed AQI"
)

sns.kdeplot(
    aqi,
    color="blue",
    linewidth=2,
    label="KDE Curve"
)

# Normal distribution curve using mean and SD
x = np.linspace(aqi.min(), aqi.max(), 400)

normal_curve = (
    1 / (aqi.std() * np.sqrt(2 * np.pi))
    * np.exp(-0.5 * ((x - aqi.mean()) / aqi.std()) ** 2)
)

plt.plot(
    x,
    normal_curve,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Normal Distribution"
)

plt.axvline(
    aqi.mean(),
    color="green",
    linestyle=":",
    label="Mean"
)

plt.axvline(
    aqi.median(),
    color="orange",
    linestyle="-.",
    label="Median"
)

plt.title(
    f"AQI Distribution and Skewness = {aqi.skew():.3f}"
)

plt.xlabel("AQI")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ==========================================
# 8. NORMAL Q-Q PLOT
# ==========================================

sorted_aqi = np.sort(aqi)
n = len(sorted_aqi)

# Theoretical normal quantiles (no SciPy required)
from statistics import NormalDist

normal_dist = NormalDist()

theoretical_quantiles = [
    normal_dist.inv_cdf((i - 0.5) / n)
    for i in range(1, n + 1)
]

# Standardize observed AQI values
standardized_aqi = (
    sorted_aqi - aqi.mean()
) / aqi.std()

# Reference line
slope = np.std(standardized_aqi, ddof=1)
intercept = np.mean(standardized_aqi)

plt.figure(figsize=(7, 7))

plt.scatter(
    theoretical_quantiles,
    standardized_aqi,
    color="blue",
    alpha=0.7,
    label="Observed AQI Quantiles"
)

plt.plot(
    theoretical_quantiles,
    intercept + slope * np.array(theoretical_quantiles),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Normal Reference Line"
)

plt.title("Normal Q-Q Plot - April 2026")
plt.xlabel("Theoretical Normal Quantiles")
plt.ylabel("Standardized AQI Quantiles")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\nFINAL SUMMARY")
print("-------------")
print("Mean:", round(aqi.mean(), 2))
print("Median:", round(aqi.median(), 2))
print("Skewness:", round(aqi.skew(), 3))
print("Kurtosis:", round(aqi.kurt(), 3))
print("IQR:", round(IQR, 2))
print("Outliers:", len(outliers))
