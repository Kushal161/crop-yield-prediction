import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load datasets
file_path = 'Crop_recommendation.csv'  # Path for Crop Recommendation data
data = pd.read_csv(file_path)

yield_file_path = 'yield_df.csv'  # Path for Crop Yield data
yield_data = pd.read_csv(yield_file_path)

# ------------------- CNN for Crop Recommendation -------------------

# Prepare data for the CNN model
X = data.drop('label', axis=1)  # Features
y = data['label']  # Crop labels

# Encode the target labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features (helpful for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for crop recommendation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Build the CNN model
recommend_model = Sequential()
recommend_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
recommend_model.add(Dense(64, activation='relu'))
recommend_model.add(Dense(32, activation='relu'))
recommend_model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

# Compile the model
recommend_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the recommendation model
recommend_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate CNN model for crop recommendation
print("---- CNN Crop Recommendation Model Evaluation ----")
cnn_loss, cnn_accuracy = recommend_model.evaluate(X_test, y_test)
print(f"Accuracy: {cnn_accuracy * 100:.2f}%")

# Get predictions and classification report
y_pred_cnn = recommend_model.predict(X_test)
y_pred_cnn_classes = np.argmax(y_pred_cnn, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnn_classes))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnn_classes))

# ------------------- DNN for Crop Yield Prediction -------------------

# Prepare data for yield prediction
X_yield = yield_data[['average_rain_fall_mm_per_year', 'avg_temp']]  # Features for yield prediction
y_yield = yield_data['hg/ha_yield']  # Target variable (yield in hg/ha)

# Standardize the features
X_yield_scaled = scaler.fit_transform(X_yield)

# Train-test split for yield prediction
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(X_yield_scaled, y_yield, test_size=0.3, random_state=42)

# Build the DNN model for yield prediction
yield_model = Sequential()
yield_model.add(Dense(128, input_dim=X_yield_train.shape[1], activation='relu'))
yield_model.add(Dense(64, activation='relu'))
yield_model.add(Dense(32, activation='relu'))
yield_model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
yield_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the yield prediction model
yield_model.fit(X_yield_train, y_yield_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate DNN model for crop yield prediction
print("\n---- DNN Crop Yield Prediction Model Evaluation ----")
y_yield_pred = yield_model.predict(X_yield_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_yield_test, y_yield_pred)
mae = mean_absolute_error(y_yield_test, y_yield_pred)
r2 = r2_score(y_yield_test, y_yield_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")
