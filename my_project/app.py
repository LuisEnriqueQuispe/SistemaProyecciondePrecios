from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelos
knn = joblib.load('models/knn_model.pkl')
log_reg = joblib.load('models/log_reg_model.pkl')
nn = joblib.load('models/nn_model.pkl')

# Cargar datos
file_path = 'IshopPeru2024.csv'
data = pd.read_csv(file_path)

# Página de inicio
@app.route('/')
def home():
    # Obtener lista de productos únicos
    products = data['Product Name'].unique()
    return render_template('index.html', products=products)

# Endpoint para predecir
@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form['model']
    product_name = request.form['product']
    target_year = int(request.form['year'])

    # Verificar que el año no sea menor a 2024
    if target_year < 2024:
        return jsonify({'error': 'El año no puede ser menor a 2024.'}), 400

    # Obtener características del producto seleccionado
    product_data = data[data['Product Name'] == product_name].iloc[0]
    ratings = product_data['Number Of Ratings']
    star_rating = product_data['Star Rating']
    ram = int(product_data['Ram'].replace(' GB', ''))
    base_price = product_data['Sale Price']

    # Ajustar el precio basado en el año
    depreciation_rate = 0.1  # Ejemplo de tasa de depreciación anual del 10%
    years_diff = target_year - 2024
    adjusted_price = base_price * ((1 - depreciation_rate) ** years_diff)

    features = pd.DataFrame([[ratings, star_rating, ram, target_year]], columns=['Number Of Ratings', 'Star Rating', 'Ram', 'Year'])

    # Cargar porcentajes de entrenamiento y validación
    with open('models/train_validation_percentages.txt', 'r') as f:
        percentages = f.readlines()
        train_percent = float(percentages[0].strip().split(': ')[1])
        validation_percent = float(percentages[1].strip().split(': ')[1])

    if model_choice == 'knn':
        predicted_relative_price = knn.predict(features)[0]
    elif model_choice == 'log_reg':
        predicted_relative_price = log_reg.predict(features)[0]
    elif model_choice == 'nn':
        predicted_relative_price = nn.predict(features)[0]

    # Ajustar la predicción para la devaluación
    final_prediction = adjusted_price

    return jsonify({'prediction': final_prediction, 'train_percent': train_percent, 'validation_percent': validation_percent})

if __name__ == '__main__':
    app.run(debug=True)
