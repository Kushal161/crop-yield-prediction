from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
import webbrowser

# Load datasets
file_path = 'Crop_recommendation.csv'  # Path for Crop Recommendation data
data = pd.read_csv(file_path)

yield_file_path = 'yield_df.csv'  # Path for Crop Yield data
yield_data = pd.read_csv(yield_file_path)

# Prepare data for the CNN model (Crop Recommendation)
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

# Build a feedforward neural network for crop recommendation
recommend_model = Sequential()
recommend_model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
recommend_model.add(Dense(64, activation='relu'))
recommend_model.add(Dense(32, activation='relu'))
recommend_model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

# Compile the model for crop recommendation
recommend_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the recommendation model
recommend_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Prepare data for the CNN-style DNN (Crop Yield Prediction)
# We'll use the yield data and train a model to predict the yield

# Features for yield prediction (rainfall and avg_temp)
X_yield = yield_data[['average_rain_fall_mm_per_year', 'avg_temp']]

# Target variable (yield in hg/ha)
y_yield = yield_data['hg/ha_yield']

# Standardize the features for yield prediction
X_yield_scaled = scaler.fit_transform(X_yield)

# Train-test split for yield prediction
X_yield_train, X_yield_test, y_yield_train, y_yield_test = train_test_split(X_yield_scaled, y_yield, test_size=0.3, random_state=42)

# Build a neural network for crop yield prediction
yield_model = Sequential()
yield_model.add(Dense(128, input_dim=X_yield_train.shape[1], activation='relu'))
yield_model.add(Dense(64, activation='relu'))
yield_model.add(Dense(32, activation='relu'))
yield_model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model for crop yield prediction
yield_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the yield prediction model
yield_model.fit(X_yield_train, y_yield_train, epochs=50, batch_size=32, validation_split=0.2)

# Initialize Flask app
app = Flask(__name__)

# Home route with two separate options for Crop Recommendation and Crop Yield Prediction
@app.route('/')
def home():
    wordpress_url = "http://localhost/wordpress-6.6.2/wordpress/"
    webbrowser.open(wordpress_url)
    return "Opening WordPress home page in the default browser..."

# Block 1: Crop Recommendation Route
@app.route('/recommend_crop', methods=['GET', 'POST'])
def recommend_crop():
    if request.method == 'POST':
        # Get user input for environmental factors
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create a DataFrame for the user input and scale it
        user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        user_input_scaled = scaler.transform(user_input)

        # Make crop recommendation using the neural network model
        predicted_crop_index = recommend_model.predict(user_input_scaled).argmax(axis=-1)[0]
        predicted_crop = label_encoder.inverse_transform([predicted_crop_index])[0]

        # Render the recommendation result
        return render_template('recommend_result.html', crop=predicted_crop)
    return render_template('recommend_crop.html')

# Block 2: Crop Yield Prediction Route
@app.route('/predict_yield', methods=['GET', 'POST'])
def predict_yield():
    if request.method == 'POST':
        # Get user input for selected crop, rainfall, and average temperature
        selected_crop = request.form['crop']
        rainfall = float(request.form['rainfall'])
        avg_temp = float(request.form['avg_temp'])

        # Standardize the user input for yield prediction
        user_input_yield = pd.DataFrame([[rainfall, avg_temp]], columns=['average_rain_fall_mm_per_year', 'avg_temp'])
        user_input_yield_scaled = scaler.transform(user_input_yield)

        # Predict crop yield using the yield prediction model
        predicted_yield_hg = yield_model.predict(user_input_yield_scaled)[0][0]
        predicted_yield_kg = predicted_yield_hg / 10  # Convert hg to kg

        # Approximate price ranges per kg for each crop
        price_dict = {
            'maize': (25, 35),
            'Potatoes': (15, 25),
            'rice': (30, 45),
            'Sorghum': (20, 30),
            'Soybeans': (40, 50),
            'Wheat': (25, 35),
            'Cassava': (10, 15),
            'Sweet potatoes': (15, 20),
            'banana': (10, 15),
            'Yams': (20, 30)
        }

        if selected_crop in price_dict:
            price_min, price_max = price_dict[selected_crop]
            min_cost = predicted_yield_kg * price_min
            max_cost = predicted_yield_kg * price_max
        else:
            min_cost, max_cost = "N/A", "N/A"

        # Render the crop yield and cost result
        return render_template('yield_result.html',
                               yield_prediction=predicted_yield_hg,
                               crop=selected_crop,
                               min_cost=min_cost,
                               max_cost=max_cost)

    # Get unique crop names from yield_data for dropdown
    crops = yield_data['Item'].unique()
    return render_template('predict_yield.html', crops=crops)

@app.route('/predict_pesticide', methods=['GET', 'POST'])
def predict_pesticide():
    if request.method == 'POST':
        # Get user input for selected crop
        selected_crop = request.form['crop']

        # Fetch the pesticide requirement from yield_data for the selected crop
        filtered_pesticide_data = yield_data[yield_data['Item'] == selected_crop]

        if not filtered_pesticide_data.empty:
            # Get the pesticide required from the 'pesticide_required' column
            pesticide_required = filtered_pesticide_data['pesticides_tonnes'].values[0]  # Replace with actual column name if needed
        else:
            pesticide_required = "No data available for the selected crop."

        # Render the pesticide prediction result
        return render_template('pesticide_result.html', pesticide_required=pesticide_required, crop=selected_crop)

    # Get unique crop names from yield_data for dropdown
    crops = yield_data['Item'].unique()
    return render_template('predict_pesticide.html', crops=crops)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
