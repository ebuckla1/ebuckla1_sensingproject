import pandas as pd
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

data_dir = "training_data"

all_window_features = []
file_count = 0
row_count = 0

for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        file_count += 1

        all_window_features.append(df)
        row_count += len(df)


window_df = pd.concat(all_window_features, ignore_index=True)

X = window_df.drop(columns=["sleep_score"])
y = window_df["sleep_score"]

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")

with open("sleep_score_windowed.pkl", "wb") as f:
    pickle.dump(model, f)

with open("feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)