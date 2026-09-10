import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Create plots folder
# -----------------------
os.makedirs("plots", exist_ok=True)

print("Loading Global Weather Dataset...")

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("GlobalWeatherRepository.csv")

print("\nDataset Loaded Successfully!")

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

# -----------------------
# Convert Date Column
# -----------------------
df["last_updated"] = pd.to_datetime(df["last_updated"])

# -----------------------
# Missing Values
# -----------------------
print("\nMissing Values")
print(df.isnull().sum())

# -----------------------
# Temperature Trend
# -----------------------
plt.figure(figsize=(12,5))
plt.plot(df["last_updated"].head(5000), df["temperature_celsius"].head(5000), color="red")
plt.title("Temperature Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("plots/temperature_trend.png")
plt.show()

# -----------------------
# Temperature Distribution
# -----------------------
plt.figure(figsize=(8,5))
sns.histplot(df["temperature_celsius"], bins=30, kde=True, color="orange")
plt.title("Temperature Distribution")
plt.tight_layout()
plt.savefig("plots/temperature_distribution.png")
plt.show()

# -----------------------
# Humidity Distribution
# -----------------------
plt.figure(figsize=(8,5))
sns.histplot(df["humidity"], bins=30, kde=True, color="blue")
plt.title("Humidity Distribution")
plt.tight_layout()
plt.savefig("plots/humidity_distribution.png")
plt.show()

# -----------------------
# Wind Speed Distribution
# -----------------------
plt.figure(figsize=(8,5))
sns.histplot(df["wind_kph"], bins=30, kde=True, color="green")
plt.title("Wind Speed Distribution")
plt.tight_layout()
plt.savefig("plots/wind_speed_distribution.png")
plt.show()

# -----------------------
# Temperature vs Humidity
# -----------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df.sample(5000, random_state=42),
    x="temperature_celsius",
    y="humidity",
    alpha=0.5
)
plt.title("Temperature vs Humidity")
plt.tight_layout()
plt.savefig("plots/temp_vs_humidity.png")
plt.show()

# -----------------------
# Correlation Heatmap
# -----------------------
cols = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "precip_mm",
    "uv_index"
]

plt.figure(figsize=(10,6))
sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# -----------------------
# Top 10 Hottest Countries
# -----------------------
top_hot = (
    df.groupby("country")["temperature_celsius"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_hot.plot(kind="bar", color="red")
plt.title("Top 10 Hottest Countries")
plt.ylabel("Average Temperature")
plt.tight_layout()
plt.savefig("plots/top10_hottest_countries.png")
plt.show()

# -----------------------
# Top 10 Humid Countries
# -----------------------
top_humid = (
    df.groupby("country")["humidity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_humid.plot(kind="bar", color="blue")
plt.title("Top 10 Most Humid Countries")
plt.ylabel("Average Humidity")
plt.tight_layout()
plt.savefig("plots/top10_humid_countries.png")
plt.show()

# -----------------------
# Monthly Temperature
# -----------------------
df["Month"] = df["last_updated"].dt.month

monthly = df.groupby("Month")["temperature_celsius"].mean()

plt.figure(figsize=(10,5))
monthly.plot(marker="o", color="purple")
plt.title("Monthly Average Temperature")
plt.xlabel("Month")
plt.ylabel("Temperature")
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/monthly_temperature.png")
plt.show()

# -----------------------
# Weather Condition Count
# -----------------------
top_conditions = df["condition_text"].value_counts().head(10)

plt.figure(figsize=(10,5))
top_conditions.plot(kind="bar", color="teal")
plt.title("Top Weather Conditions")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/weather_conditions.png")
plt.show()

print("\nProject Completed Successfully!")
print("All charts are saved inside the 'plots' folder.")