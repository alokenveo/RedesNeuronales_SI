#Evitar avisos de Keras y TensorFlow sobre la utilización de OneDNN
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Importar TensorFlow
import tensorflow as tf
import numpy as np
import time

# Datos simples de entrada y salida
# Queremos aprender la relación: y = 2x - 1
x_train = np.array([0, 1, 2, 3, 4], dtype=float)  # Entrada
y_train = np.array([-1, 1, 3, 5, 7], dtype=float) # Salida esperada

# Crear el modelo secuencial
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  # Una capa con una neurona y una entrada
])

# Compilar el modelo
model.compile(
    optimizer='sgd',                    # Descenso de gradiente estocástico
    loss='mean_squared_error'           # Error cuadrático medio
)

# Entrenar el modelo
print("Entrenando el modelo...")
start_time = time.time()
model.fit(x_train, y_train, epochs=500, verbose=0)  # Entrenar por 500 épocas
end_time = time.time()
print("Entrenamiento completado.")
print(f"Tiempo total: {end_time - start_time:.2f} segundos.\n")

# Hacer predicciones
print("Predicción con el modelo entrenado:")
test_input = np.array([5, 10, 15], dtype=float)  # Nuevas entradas
predictions = model.predict(test_input)
for i, val in enumerate(test_input):
    print(f"Para x = {val}, y predicho = {predictions[i][0]:.2f}")
