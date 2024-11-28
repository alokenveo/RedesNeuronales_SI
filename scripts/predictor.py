#Evitar avisos de TensorFlow sobre la utilización de OneDNN
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Importar las librerías necesarias
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 2. Cargar el dataset
print("Cargando el dataset California Housing...")
housing = fetch_california_housing()

# Convertir el dataset en un DataFrame de pandas
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target  # Añadir la columna de precios de las casas

# 3. Exploración del Dataset
print("\nPrimeras filas del dataset:")
print(df.head())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Verificar valores nulos
print("\nVerificación de valores nulos:")
print(df.isnull().sum())

# 4. Preprocesamiento de los Datos
# Normalizar las características numéricas
scaler = StandardScaler()
X = df.drop(columns=['PRICE'])
y = df['PRICE']

# Normalizar las características de entrada
X_scaled = scaler.fit_transform(X)

# Dividir el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Diseñar la Red Neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),  # Capa de entrada
    tf.keras.layers.Dense(units=32, activation='relu'),  # Capa oculta 1
    tf.keras.layers.Dense(units=16, activation='relu'),  # Capa oculta 2
    tf.keras.layers.Dense(units=1)  # Capa de salida (regresión)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Entrenar el Modelo
# Dividir el conjunto de entrenamiento en datos de validación (20%)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# 7. Evaluar el Modelo
print("\nEvaluando el modelo...")
test_loss = model.evaluate(X_test, y_test)
print(f"\nPérdida en el conjunto de prueba: {test_loss:.4f}")

# Hacer predicciones
predictions = model.predict(X_test)

# Comparar las predicciones con los valores reales
print("\nComparación de los valores reales con las predicciones:")
for i in range(10):
    print(f"Real: {y_test.iloc[i]:.2f}, Predicho: {predictions[i][0]:.2f}")

# 8. Generar Métricas
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nError cuadrático medio (MSE): {mse:.4f}")
print(f"Coeficiente de determinación (R²): {r2:.4f}")

# 9. Generar Gráficas
# Gráfica de la pérdida durante el entrenamiento
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Gráfica de dispersión comparando valores reales vs predicciones
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions)
plt.title('Valores Reales vs Predicciones')
plt.xlabel('Valor Real')
plt.ylabel('Valor Predicho')
plt.show()
