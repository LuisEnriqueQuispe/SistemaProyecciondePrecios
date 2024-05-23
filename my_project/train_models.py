import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import joblib
import matplotlib.pyplot as plt

# Crear la carpeta 'static' si no existe
if not os.path.exists('static'):
    os.makedirs('static')

# Cargar los datos
file_path = 'IshopPeru2024.csv'
data = pd.read_csv(file_path)

# Agregar una columna de año (asumiendo que todos los datos son de 2024)
data['Year'] = 2024

# Crear una función para ajustar el precio basado en el año
def adjust_price(row, target_year):
    depreciation_rate = 0.1  # Ejemplo de tasa de depreciación anual del 10%
    years_diff = target_year - row['Year']
    return row['Sale Price'] * ((1 - depreciation_rate) ** years_diff)

# Crear datos ajustados para diferentes años (2024-2030)
years = list(range(2024, 2031))
adjusted_data = []

for year in years:
    adjusted_year_data = data.copy()
    adjusted_year_data['Year'] = year
    adjusted_year_data['Sale Price'] = adjusted_year_data.apply(lambda row: adjust_price(row, year), axis=1)
    adjusted_data.append(adjusted_year_data)

# Combinar todos los datos ajustados
all_data = pd.concat(adjusted_data, ignore_index=True)

# Selección de características y etiqueta
X = all_data[['Number Of Ratings', 'Star Rating', 'Ram', 'Year']]
y = all_data['Sale Price']

# Convertir 'Ram' de texto a número
X['Ram'] = X['Ram'].str.replace(' GB', '').astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular porcentajes de entrenamiento y validación
train_percent = (len(X_train) / len(X)) * 100
validation_percent = (len(X_test) / len(X)) * 100

# Entrenamiento de modelos
# KNN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
joblib.dump(knn, 'models/knn_model.pkl')

# Regresión Logística (adaptada para regresión)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train.astype('int'))
joblib.dump(log_reg, 'models/log_reg_model.pkl')

# Redes Neuronales
nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
nn.fit(X_train, y_train)
joblib.dump(nn, 'models/nn_model.pkl')

# Guardar los porcentajes de entrenamiento y validación
with open('models/train_validation_percentages.txt', 'w') as f:
    f.write(f'Train Percent: {train_percent}\n')
    f.write(f'Validation Percent: {validation_percent}\n')

# Crear y guardar gráfico de porcentaje de validación
fig, ax = plt.subplots()
ax.pie([train_percent, validation_percent], labels=['Train', 'Validation'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Para un gráfico de pie circular
plt.title('Train vs Validation Data Distribution')
plt.savefig('static/validation_percentage.png')
plt.close()

print("Modelos entrenados y guardados en la carpeta 'models'.")
