# Importar TensorFlow
import tensorflow as tf
import numpy as np
import time

from sklearn.metrics import r2_score


# Datos de entrada y salida
# Queremos aprender la relación: y = 3x^2 + 2x + 1
x_train = np.array([-2, -1, 0, 1, 2, 3], dtype=float)  # Entradas
y_train = np.array([9, 2, 1, 6, 17, 34], dtype=float) # Salidas esperadas

# Crear el modelo secuencial con varias capas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[1]),  # Primera capa oculta
    tf.keras.layers.Dense(units=32, activation='relu'),                  # Segunda capa oculta
    tf.keras.layers.Dense(units=16, activation='relu'),                  # Tercera capa oculta
    tf.keras.layers.Dense(units=1)                                       # Capa de salida
])

# Compilar el modelo
model.compile(
    optimizer='adam',                          # Optimizador Adam
    loss='mean_squared_error'                  # Pérdida: Error cuadrático medio
)

# Entrenar el modelo
print("Entrenando el modelo...")
start_time = time.time()
model.fit(x_train, y_train, epochs=1000, verbose=0)  # Entrenar por 1000 épocas
end_time = time.time()
print("Entrenamiento completado.")
print(f"Tiempo total: {end_time - start_time:.2f} segundos\n")



# Hacer predicciones
print("Predicción con el modelo entrenado:")
test_input = np.array([4, 5, 6], dtype=float)  # Nuevas entradas
predictions = model.predict(test_input)
for i, val in enumerate(test_input):
    print(f"Para x = {val}, y predicho = {predictions[i][0]:.2f}")

r2 = r2_score(y_train, model.predict(x_train))
print(f"R-squared: {r2:.4f}")

