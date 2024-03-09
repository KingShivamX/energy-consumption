from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

file_path = 'energy-consumption\my_data.csv'
df = pd.read_csv(file_path)

X = df[['NoOfRooms', 'Occupancy', 'HeavyAppliances', 'HeatingCoolingSystems']]
y = df['ElectricityBill']

model = LinearRegression()
model.fit(X, y)


def generate_plot(size, predicted_energy):


    building_sizes = df['NoOfRooms']
    actual_energy_consumption = df['ElectricityBill']


    plt.figure(figsize=(8, 6))


    plt.scatter(building_sizes, actual_energy_consumption, color='orange', marker='o', label='Data from CSV file')


    plt.scatter(size, predicted_energy, color='blue', label='Predicted Energy Consumption')


    plt.axhline(y=predicted_energy, color='red', linestyle='--', label='Predicted Electricity Bill')
    plt.axvline(x=size, color='green', linestyle='--', label='Input No of Rooms')


    plt.annotate(f'Predicted Electricity Bill: {predicted_energy:.2f} Rupees', xy=(0.05, 0.80), xycoords='axes fraction', color='black', fontsize=10)
    plt.annotate(f'Input No of Rooms: {size} rooms', xy=(0.05, 0.75), xycoords='axes fraction', color='black', fontsize=10)


    plt.xlabel('No. of Rooms')
    plt.ylabel('Electricity Bill (Rupees)')
    plt.title('Energy Consumption Prediction')


    plt.legend()


    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close()


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


    new_building = np.array([[size, people, insulation, heating_cooling_systems]])
    predicted_energy = model.predict(new_building)[0]

    predicted_energy_divided = predicted_energy / 8

    img_base64 = generate_plot(size, predicted_energy)

    return render_template('index.html', size=size, people=people, insulation=insulation,
                           heating_cooling_systems=heating_cooling_systems, predicted_energy=predicted_energy,
                           predicted_energy_divided=predicted_energy_divided,
                           plot=img_base64,)


if __name__ == '__main__':
    app.run(debug=True)



# /////////////////////////////////////////////////////////////////////////////////