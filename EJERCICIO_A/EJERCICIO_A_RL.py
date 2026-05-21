import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'area': [95, 123, 118, 96, 182, 86, 63, 193, 155, 128, 195, 115, 105, 140],
    'habitaciones': [4, 2, 2, 4, 4, 4, 2, 2, 2, 2, 5, 5, 3, 3],
    'antiguedad': [13, 14, 22, 8, 19, 5, 7, 3, 29, 3, 18, 22, 10, 6],
    'distancia': [12.5, 8.0, 15.0, 5.5, 9.0, 3.5, 18.0, 4.0, 22.0, 6.5, 7.5, 20.0, 2.0, 14.0],
    'precio': [171600, 216200, 196900, 190200, 305700, 180100, 133600, 320000, 242700, 216800, 357400, 219800, 198500, 241300]
}

df = pd.DataFrame(data)

# ===========================================================================
# FUNCIONES MATEMÁTICAS LOGÍSTICAS
# ===========================================================================

def normalizar_datos(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def regresion_logistica_gd(X, y, lr=0.01, epocas=2000):
    n_muestras, n_features = X.shape
    w = np.zeros(n_features)
    
    for i in range(epocas):
        z = np.dot(X, w)
        y_pred_prob = sigmoide(z)
        
        error = y_pred_prob - y
        gradiente = (1 / n_muestras) * np.dot(X.T, error)
        
        w = w - lr * gradiente
        
    return w

# ===========================================================================
# ETAPA 2.1
# ===========================================================================

print("\n--- ETAPA 2.1: VARIABLE OBJETIVO BINARIA ---")

df['y_bin'] = (df['precio'] > 200000).astype(int)

print(df['y_bin'].value_counts())

# ===========================================================================
# ETAPA 2.2 - MODELO UNIVARIADO "SOLO ÁREA"
# ===========================================================================

# 1. Definir feature y target
X_area = df[['area']].values.astype(float)
y = df['y_bin'].values

# 2. Normalizar y agregar columna de Bias (x0 = 1)
X_area_norm, mu_area, sigma_area = normalizar_datos(X_area)
X_gd_area = np.c_[np.ones(X_area_norm.shape[0]), X_area_norm]

# 3. Entrenar el modelo (Parámetros del PDF: lr=0.01, 2000 épocas)
W_area = regresion_logistica_gd(X_gd_area, y, lr=0.01, epocas=2000)

# 4. Calcular el Accuracy
probabilidades = sigmoide(np.dot(X_gd_area, W_area))
predicciones = (probabilidades >= 0.5).astype(int)
accuracy_area = np.mean(predicciones == y) * 100

print("\n--- ETAPA 2.2: MODELO SOLO ÁREA (x1) ---")
print(f"w0 (Bias): {W_area[0]:.4f}")
print(f"w1 (Peso Área): {W_area[1]:.4f}")
print(f"Accuracy: {accuracy_area:.2f}%")

# ===========================================================================
# ETAPA 2.2 - MODELO UNIVARIADO "SOLO HABITACIONES"
# ===========================================================================

# 1. Definir feature y target
X_hab = df[['habitaciones']].values.astype(float)
y = df['y_bin'].values

# 2. Normalizar y agregar columna de Bias (x0 = 1)
X_hab_norm, mu_hab, sigma_hab = normalizar_datos(X_hab)
X_gd_hab = np.c_[np.ones(X_hab_norm.shape[0]), X_hab_norm]

# 3. Entrenar el modelo (Parámetros del PDF: lr=0.01, 2000 épocas)
W_hab = regresion_logistica_gd(X_gd_hab, y, lr=0.01, epocas=2000)

# 4. Calcular el Accuracy
probabilidades_hab = sigmoide(np.dot(X_gd_hab, W_hab))
predicciones_hab = (probabilidades_hab >= 0.5).astype(int)
accuracy_hab = np.mean(predicciones_hab == y) * 100

print("\n--- ETAPA 2.2: MODELO SOLO HABITACIONES (x2) ---")
print(f"w0 (Bias): {W_hab[0]:.4f}")
print(f"w2 (Peso Habitaciones): {W_hab[1]:.4f}")
print(f"Accuracy: {accuracy_hab:.2f}%")

# ===========================================================================
# ETAPA 2.2 - MODELO UNIVARIADO "SOLO ANTIGUEDAD"
# ===========================================================================

X_ant = df[['antiguedad']].values.astype(float)

X_ant_norm, mu_ant, sigma_ant = normalizar_datos(X_ant)
X_gd_ant = np.c_[np.ones(X_ant_norm.shape[0]), X_ant_norm]

W_ant = regresion_logistica_gd(X_gd_ant, y, lr=0.01, epocas=2000)

probabilidades_ant = sigmoide(np.dot(X_gd_ant, W_ant))
predicciones_ant = (probabilidades_ant >= 0.5).astype(int)
accuracy_ant = np.mean(predicciones_ant == y) * 100

print("\n--- ETAPA 2.2: MODELO SOLO ANTIGÜEDAD (x3) ---")
print(f"w0 (Bias): {W_ant[0]:.4f}")
print(f"w3 (Peso Antigüedad): {W_ant[1]:.4f}")
print(f"Accuracy: {accuracy_ant:.2f}%")

# ===========================================================================
# ETAPA 2.2 - MODELO UNIVARIADO "SOLO DISTANCIA"
# ===========================================================================

X_dist = df[['distancia']].values.astype(float)

X_dist_norm, mu_dist, sigma_dist = normalizar_datos(X_dist)
X_gd_dist = np.c_[np.ones(X_dist_norm.shape[0]), X_dist_norm]

W_dist = regresion_logistica_gd(X_gd_dist, y, lr=0.01, epocas=2000)

probabilidades_dist = sigmoide(np.dot(X_gd_dist, W_dist))
predicciones_dist = (probabilidades_dist >= 0.5).astype(int)
accuracy_dist = np.mean(predicciones_dist == y) * 100

print("\n--- ETAPA 2.2: MODELO SOLO DISTANCIA (x4) ---")
print(f"w0 (Bias): {W_dist[0]:.4f}")
print(f"w4 (Peso Distancia): {W_dist[1]:.4f}")
print(f"Accuracy: {accuracy_dist:.2f}%")

# ===========================================================================
# ETAPA 2.3 - MODELO LOGÍSTICO COMPLETO
# ===========================================================================

# 1. Definir TODAS las features y el target
columnas_features = ['area', 'habitaciones', 'antiguedad', 'distancia']
X_completa = df[columnas_features].values.astype(float)
y = df['y_bin'].values

# 2. Normalizar la matriz completa y agregar columna de Bias (x0 = 1)
# guardamos mu_comp y sigma_comp para usarlas en el Ítem 8
X_comp_norm, mu_comp, sigma_comp = normalizar_datos(X_completa)
X_gd_comp = np.c_[np.ones(X_comp_norm.shape[0]), X_comp_norm]

# 3. Entrenar el modelo
W_comp = regresion_logistica_gd(X_gd_comp, y, lr=0.01, epocas=3000)

# 4. Calcular el Accuracy del modelo completo
probabilidades_comp = sigmoide(np.dot(X_gd_comp, W_comp))
predicciones_comp = (probabilidades_comp >= 0.5).astype(int)
accuracy_comp = np.mean(predicciones_comp == y) * 100

print("\n--- ETAPA 2.3: MODELO COMPLETO (4 CARACTERÍSTICAS) ---")
print(f"w0 (Bias): {W_comp[0]:.4f}")
print(f"w1 (Peso Área): {W_comp[1]:.4f}")
print(f"w2 (Peso Habitaciones): {W_comp[2]:.4f}")
print(f"w3 (Peso Antigüedad): {W_comp[3]:.4f}")
print(f"w4 (Peso Distancia): {W_comp[4]:.4f}")
print(f"Accuracy Modelo Completo: {accuracy_comp:.2f}%")

# ===========================================================================
# ETAPA 2.3 - MATRIZ DE CONFUSIÓN
# ===========================================================================

# Calcular TP, TN, FP, FN
TP = np.sum((predicciones_comp == 1) & (y == 1))
TN = np.sum((predicciones_comp == 0) & (y == 0))
FP = np.sum((predicciones_comp == 1) & (y == 0))
FN = np.sum((predicciones_comp == 0) & (y == 1))

matriz_confusion = np.array([[TN, FP], 
                             [FN, TP]])

plt.figure(figsize=(6, 4))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Pred. ESTÁNDAR (0)', 'Pred. PREMIUM (1)'], 
            yticklabels=['Real ESTÁNDAR (0)', 'Real PREMIUM (1)'],
            annot_kws={"size": 16})
plt.title('Matriz de Confusión - Modelo Completo', fontsize=14)
plt.show()

# ===========================================================================
# ETAPA 2.3 - PREDICCIÓN DE UNA NUEVA PROPIEDAD
# ===========================================================================

# Propiedad a predecir: 175 m2, 4 hab, 8 años, 6 km
x_nueva = np.array([175.0, 4.0, 8.0, 6.0])

# 1. Normalizamos con los mu y sigma del modelo completo
x_nueva_norm = (x_nueva - mu_comp) / sigma_comp

# 2. Agregamos el Bias Trick (x0 = 1)
x_nueva_gd = np.insert(x_nueva_norm, 0, 1.0)

# 3. Calculamos la combinación lineal (z)
z_nueva = np.dot(x_nueva_gd, W_comp)

# 4. Calculamos la probabilidad final pasando por la Sigmoide
probabilidad_premium = sigmoide(z_nueva)

print("\n--- ETAPA 2.3: PREDICCIÓN NUEVA PROPIEDAD ---")
print(f"Combinación lineal (z): {z_nueva:.4f}")
print(f"Probabilidad de ser PREMIUM P(y=1): {probabilidad_premium * 100:.2f}%")
