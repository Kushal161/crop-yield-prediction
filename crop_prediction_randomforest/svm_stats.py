from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVC, SVR  # Import SVC for classification, SVR for regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Load datasets
file_path = 'Crop_recommendation.csv'  # Path for Crop Recommendation data
data = pd.read_csv(file_path)

yield_file_path = 'yield_df.csv'  # Path for Crop Yield data
yield_data = pd.read_csv(yield_file_path)

# Prepare data for the SVM model (Crop Recommendation)
X = data.drop('label', axis=1)  # Features for crop recommendation
y = data['label']  # Crop labels

# Train-test split and train the SVM model for Crop Recommendation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the SVM model for classification
svm_model = SVC(kernel='rbf', random_state=42)  # RBF kernel for non-linear classification
svm_model.fit(X_train, y_train)

# Prepare data for SVR model (Yield Prediction)
yield_X = yield_data[['average_rain_fall_mm_per_year', 'avg_temp']]  # Features for yield prediction
yield_y = yield_data['hg/ha_yield']  # Target variable (yield in hg/ha)

# Split the data into training and testing sets
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(yield_X, yield_y, test_size=0.3, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_yield_train_scaled = scaler.fit_transform(X_yield_train)
X_yield_test_scaled = scaler.transform(X_yield_test)

# Initialize and train the SVR model for regression
svr_model = SVR(kernel='rbf')  # RBF kernel for regression
svr_model.fit(X_yield_train_scaled, y_yield_train)

# Get unique elements in the dataset
crop_names = yield_data['Item'].unique()
print(crop_names)


# Evaluate SVM Model (Crop Recommendation)
y_pred_svm = svm_model.predict(X_test)

# Generate classification report
print("Classification Report for SVM (Crop Recommendation):")
print(classification_report(y_test, y_pred_svm))

# Confusion matrix
print("Confusion Matrix for SVM (Crop Recommendation):")
print(confusion_matrix(y_test, y_pred_svm))


# Predict on test data
y_yield_pred_svr = svr_model.predict(X_yield_test_scaled)

# Calculate Mean Squared Error
mse_svr = mean_squared_error(y_yield_test, y_yield_pred_svr)

# Calculate Mean Absolute Error
mae_svr = mean_absolute_error(y_yield_test, y_yield_pred_svr)

# Calculate R-squared score
r2_svr = r2_score(y_yield_test, y_yield_pred_svr)

print(f"SVR Regression Evaluation for Crop Yield Prediction:")
print(f"MSE: {mse_svr}")
print(f"MAE: {mae_svr}")
print(f"R²: {r2_svr}")
