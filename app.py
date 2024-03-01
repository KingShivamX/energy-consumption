from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load the data from the CSV file
file_path = 'zzz_deployment\energy-consumption\my_data.csv'
df = pd.read_csv(file_path)

# Split the dataset into features (X) and target variable (y)
X = df[['BuildingSize', 'Occupancy', 'Insulation', 'HeatingCoolingSystems']]
y = df['EnergyConsumption']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

def generate_plot(size, predicted_energy, mse):
    # Actual data from the CSV file
    building_sizes = df['BuildingSize']
    actual_energy_consumption = df['EnergyConsumption']

    # Plot the graph
    plt.figure(figsize=(8, 6))

    # Scatter plot for data points from CSV file
    plt.scatter(building_sizes, actual_energy_consumption, color='orange', marker='o', label='Data from CSV file')

    # Scatter plot for predicted energy consumption
    plt.scatter(size, predicted_energy, color='blue', label='Predicted Energy Consumption')

    # Add lines and annotations
    plt.axhline(y=predicted_energy, color='red', linestyle='--', label='Predicted Energy Consumption')
    plt.axvline(x=size, color='green', linestyle='--', label='Input Building Size')

    # Annotate the plot with MSE, predicted energy consumption, and input building size
    plt.annotate(f'MSE: {mse:.2f}', xy=(0.05, 0.85), xycoords='axes fraction', color='black', fontsize=10)
    plt.annotate(f'Predicted Energy: {predicted_energy:.2f} kWh', xy=(0.05, 0.80), xycoords='axes fraction', color='black', fontsize=10)
    plt.annotate(f'Input Building Size: {size} sqft', xy=(0.05, 0.75), xycoords='axes fraction', color='black', fontsize=10)

    # Set labels and title
    plt.xlabel('Building Size (Square Feet)')
    plt.ylabel('Energy Consumption (Kilowatts)')
    plt.title('Energy Consumption vs. Building Size')

    # Add legend
    plt.legend()

    # Save the plot to a BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close()

    # Encode the BytesIO object to base64 for HTML embedding
    img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

    return img_base64

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    size = int(request.form['size'])
    people = int(request.form['people'])
    insulation = int(request.form['insulation'])
    heating_cooling_systems = int(request.form['heating_cooling_systems'])

    # Predict future energy consumption for a new building
    new_building = np.array([[size, people, insulation, heating_cooling_systems]])
    predicted_energy = model.predict(new_building)[0]

    # Calculate Mean Squared Error (MSE)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # Generate plot and convert to base64
    img_base64 = generate_plot(size, predicted_energy, mse)

    return render_template('index.html', size=size, people=people, insulation=insulation,
                           heating_cooling_systems=heating_cooling_systems, predicted_energy=predicted_energy,
                           mse=mse, plot=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
