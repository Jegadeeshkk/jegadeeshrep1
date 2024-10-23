from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the data
file_path = 'college_joining_probabilities_5_years.csv'
data = pd.read_csv(file_path)

# Define the feature columns and target columns
feature_columns = ['Job_Market_Level', 'Education_Level', 'Technologies', 'Political_Stability',
                   'Natural_Disasters', 'Infrastructures', 'Demographical_Data', 'Feedbacks',
                   'Core_Industry_Data', 'Trade_Association_Reports']
target_columns = ['Joining_Probability', 'Not_Joining_Probability']

# Separate the features and targets
X = data[feature_columns]
y = data[target_columns]

# Standardize the feature columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the RandomForestRegressor on all data
model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)


def get_admission_probability(state, district, college, course):
    # Filter data for the given state, district, college, and course
    filtered_data = data[(data['state'] == state) & (data['district'] == district) &
                         (data['college'] == college) & (data['stream'] == course)]
    if filtered_data.empty:
        return None, None, "No data available for the specified input."

    # Get the feature values
    features = filtered_data[feature_columns]

    # Standardize the features
    features_scaled = scaler.transform(features)

    # Predict probabilities
    probabilities = model.predict(features_scaled)

    # Add predictions to the filtered data
    filtered_data['Joining_Probability'] = probabilities[:, 0]
    filtered_data['Not_Joining_Probability'] = probabilities[:, 1]

    # Plotting the factors
    factor_values = filtered_data[feature_columns].iloc[0].values
    factor_names = feature_columns
    factor_values = [float(value) for value in factor_values]  # Convert to float
    plt.figure(figsize=(12, 6))
    plt.bar(factor_names, factor_values, color='skyblue')
    plt.xlabel('Factor Value')
    plt.title(f'Factors Influencing Admission for {course} in {college}, {district}, {state}')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels to avoid overlap
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make sure everything fits
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    factor_graph = base64.b64encode(img.getvalue()).decode()

    # Plotting the joining and not joining probabilities
    probabilities_df = filtered_data[['Joining_Probability', 'Not_Joining_Probability']].iloc[0].astype(float)
    plt.figure(figsize=(8, 6))
    probabilities_df.plot(kind='bar', color=['green', 'red'])
    plt.ylabel('Probability (%)')
    plt.title(f'Joining vs Not Joining Probability for {course} in {college}, {district}, {state}')
    plt.xticks(rotation=0)
    plt.grid(True)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    probability_graph = base64.b64encode(img.getvalue()).decode()

    return factor_graph, probability_graph, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        state = request.form['state']
        district = request.form['district']
        college = request.form['college']
        course = request.form['course']
        factor_graph, probability_graph, error = get_admission_probability(state, district, college, course)
        return render_template('index.html', factor_graph=factor_graph, probability_graph=probability_graph,
                               error=error)
    return render_template('index.html', factor_graph=None, probability_graph=None, error=None)


if __name__ == '__main__':
    app.run(debug=True)