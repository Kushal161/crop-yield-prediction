from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load dataset (replace with your actual dataset)
file_path = 'yield_df.csv'  # Path for Crop Yield data
yield_data = pd.read_csv(file_path)

# Features for yield prediction (e.g., rainfall and avg_temp)
X = yield_data[['average_rain_fall_mm_per_year', 'avg_temp']]  # Features
y = yield_data['hg/ha_yield']  # Target variable (yield in hg/ha)

# Standardize the features (optional but often beneficial)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² score: {r2}")

# Alternatively, you can use the model's built-in score method (equivalent to r2_score)
r2_builtin = rf_model.score(X_test, y_test)
print(f"R² score (built-in): {r2_builtin}")
