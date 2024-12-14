import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

# 1. Generate Synthetic Data
np.random.seed(42)
n = 120
dates = pd.date_range("2020-01-01", periods=n, freq="ME")
trend = np.arange(n) * 0.5
seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.normal(0, 2, n)
sales = 50 + trend + seasonality + noise

data = pd.DataFrame({
    "date": dates,
    "sales": sales,
})
data["integer_index"] = np.arange(n)

# Plot the time series
plt.figure(figsize=(10, 5))
plt.plot(dates, sales, label="Synthetic Sales Data")
plt.title("Synthetic Time Series (Trend + Seasonality + Noise)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()


# Model 1: Raw datetime index (bad approach)
X_raw_date = pd.DataFrame({"raw_date": dates.astype("int64") // 10**9})
model_raw_date = LinearRegression()
model_raw_date.fit(X_raw_date, sales)

# Model 2: Integer index (slightly better but still naive)
X_integer = data[["integer_index"]]
model_integer = LinearRegression()
model_integer.fit(X_integer, sales)

# Model 3: Deterministic Process (best approach)
dp = DeterministicProcess(
    index=dates,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[CalendarFourier(freq="YE", order=4)],
)
X_deterministic = dp.in_sample()
model_deterministic = LinearRegression()
model_deterministic.fit(X_deterministic, sales)

pred_raw_date = model_raw_date.predict(X_raw_date)
pred_integer = model_integer.predict(X_integer)
pred_deterministic = model_deterministic.predict(X_deterministic)

plt.figure(figsize=(12, 6))
plt.plot(dates, sales, label="True Sales", color="black", alpha=0.7)
plt.plot(dates, pred_raw_date, label="Raw Date Index Model", linestyle="--")
plt.plot(dates, pred_integer, label="Integer Index Model", linestyle="--")
plt.plot(dates, pred_deterministic, label="Deterministic Features Model", linestyle="-")
plt.title("Comparison of Models")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()
