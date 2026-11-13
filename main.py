
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. DATA PREPROCESSING
# ==========================================

data = pd.read_csv("co2.csv")

# Convert time to datetime
data["time"] = pd.to_datetime(data["time"])

# Sort data by time
data = data.sort_values("time").reset_index(drop=True)

# Handle missing values
data["co2"] = data["co2"].interpolate()

print("Dataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nStatistics:\n", data["co2"].describe())


# ==========================================
# 2. DATA VISUALIZATION
# ==========================================

# fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Time Series Plot
# ax[0].plot(data["time"], data["co2"])
# ax[0].set_title("CO2 Concentration Over Time")
# ax[0].set_xlabel("Time")
# ax[0].set_ylabel("CO2")
# ax[0].grid(True)
#
# # Histogram
# ax[1].hist(data["co2"], bins=20, edgecolor="black")
# ax[1].set_title("CO2 Distribution")
# ax[1].set_xlabel("CO2")
# ax[1].set_ylabel("Frequency")
# ax[1].grid(axis="y", alpha=0.3)
#
# plt.tight_layout()
# plt.show()


# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================

def create_ts_data(data, window_size):

    data = data.copy()

    # Create lag features
    for i in range(1, window_size + 1):
        data[f"co2_lag_{i}"] = data["co2"].shift(i)

    # Predict next time step
    data["target"] = data["co2"].shift(-1)

    data = data.dropna()

    return data


data_ts = create_ts_data(data, 5)


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

x = data_ts.drop(["time", "target"], axis=1)
y = data_ts["target"]

train_ratio = 0.8
split_index = int(len(x) * train_ratio)

# Split data chronologically
x_train = x.iloc[:split_index]
y_train = y.iloc[:split_index]

x_test = x.iloc[split_index:]
y_test = y.iloc[split_index:]

test_time = data_ts.loc[x_test.index, "time"]


# ==========================================
# 5. MODEL TRAINING & EVALUATION
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = []
predictions = {}

for name, model in models.items():

    # Train
    model.fit(x_train, y_train)

    # Predict
    y_predict = model.predict(x_test)

    predictions[name] = y_predict

    # Evaluate
    mae = mean_absolute_error(y_test, y_predict)
    rmse = mean_squared_error(y_test, y_predict) ** 0.5
    r2 = r2_score(y_test, y_predict)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ==========================================
# 6. MODEL COMPARISON
# ==========================================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\nModel Comparison:")
print(result_df.to_string(index=False))


# ==========================================
# 7. ACTUAL VS PREDICTED
# ==========================================

best_model_name = result_df.iloc[0]["Model"]

best_predictions = predictions[best_model_name]

plt.figure(figsize=(12, 5))

plt.plot(
    test_time,
    y_test.values,
    label="Actual"
)

plt.plot(
    test_time,
    best_predictions,
    label="Predicted"
)

plt.title(f"Actual vs Predicted - {best_model_name}")
plt.xlabel("Time")
plt.ylabel("CO2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Model Comparison:
#             Model      MAE     RMSE        R2
# Linear Regression 0.360164 0.465279  0.990917
#     Random Forest 5.628664 7.051154 -1.086053
#     Decision Tree 6.299342 7.711379 -1.494991
