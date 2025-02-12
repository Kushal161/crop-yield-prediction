from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, redirect, url_for
import webbrowser


# Load datasets
file_path = 'Crop_recommendation.csv'  # Path for Crop Recommendation data
data = pd.read_csv(file_path)

yield_file_path = 'yield_df.csv'  # Path for Crop Yield data
yield_data = pd.read_csv(yield_file_path)

# Prepare data for the Random Forest model (Crop Recommendation)
X = data.drop('label', axis=1)  # Features
y = data['label']  # Crop labels

# Train-test split and train the model for Crop Recommendation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#unique elements in the dataset
crop_names = yield_data['Item'].unique()
print(crop_names)

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

        # Create a DataFrame for the user input
        user_input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # Make crop recommendation
        predicted_crop = rf_model.predict(user_input)[0]

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
        avg_temp = float(request.form['avg_temp'])  # Get avg_temp from user input

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

        # Fetch the yield data for the selected crop
        filtered_yield_data = yield_data[yield_data['Item'] == selected_crop]

        if not filtered_yield_data.empty:
            # Get the closest match for average rainfall and average temperature
            closest_yield_data = filtered_yield_data.iloc[
                ((filtered_yield_data['average_rain_fall_mm_per_year'] - rainfall).abs() +
                 (filtered_yield_data['avg_temp'] - avg_temp).abs()).argsort()[:1]
            ]

            if not closest_yield_data.empty:
                predicted_yield_hg = closest_yield_data['hg/ha_yield'].values[0]  # Predicted yield in hg/ha
                predicted_yield_kg = predicted_yield_hg / 10  # Convert hg to kg (1 kg = 10 hg)

                # Get the price range for the selected crop
                price_min, price_max = price_dict[selected_crop]

                # Calculate the cost range
                min_cost = predicted_yield_kg * price_min
                max_cost = predicted_yield_kg * price_max
            else:
                predicted_yield_hg = "No yield data available for this rainfall and temperature range."
                min_cost, max_cost = "N/A", "N/A"
        else:
            predicted_yield_hg = "No yield data available for the selected crop."
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
