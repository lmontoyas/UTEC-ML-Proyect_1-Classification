"""
Funciones de evaluación de negocio y análisis de costos.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


# Parámetros del modelo de costos
COSTO_FP = 20    # Campaña de retención innecesaria (cliente que no iba a fugarse). Costo = $20. 
COSTO_FN = 300   # Pérdida por cliente fugado no detectado  (15× el costo de retención). Costo = $300


def calcular_costo_negocio(
    y_true,                                             # etiquetas reales
    y_pred,                                             # predicciones binarias del modelo
    costo_retencion_fp: float = COSTO_FP,               # costo unitario de un FP
    costo_perdida_fn: float = COSTO_FN) -> dict:        # costo unitario de un FN

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    costo_total_fp = fp * costo_retencion_fp
    costo_total_fn = fn * costo_perdida_fn
    costo_total    = costo_total_fp + costo_total_fn

    # Ahorro vs. modelo naive (predecir siempre No-Churn)
    costo_naive = int(y_true.sum()) * costo_perdida_fn
    ahorro = costo_naive - costo_total

    return {
        'Verdaderos Positivos (Fugas Detectadas)': int(tp),
        'Verdaderos Negativos (Leales Correctos)': int(tn),
        'Falsos Positivos (Gastos Innecesarios)' : int(fp),
        'Falsos Negativos (Fugas No Detectadas)' : int(fn),
        'Costo Falsos Positivos ($)'             : round(costo_total_fp, 2),
        'Costo Falsos Negativos ($)'             : round(costo_total_fn, 2),
        'COSTO TOTAL ($)'                        : round(costo_total, 2),
        'Costo Modelo Naive ($)'                 : round(costo_naive, 2),
        'Ahorro vs. Naive ($)'                   : round(ahorro, 2),
    }


# Barrer umbrales de 0.05 a 0.8 y devolver el óptimo.
def optimizar_umbral(y_true, y_prob, criterio: str = 'costo') -> tuple:
    umbrales  = np.linspace(0.05, 0.80, 76)
    costos    = []
    f1_scores = []

    for u in umbrales:
        y_pred_u = (y_prob >= u).astype(int)
        resultado = calcular_costo_negocio(y_true, y_pred_u)
        costos.append(resultado['COSTO TOTAL ($)'])
        f1_scores.append(f1_score(y_true, y_pred_u, zero_division=0))

    if criterio == 'f1':
        idx_optimo = int(np.argmax(f1_scores))

    elif criterio == 'costo_f1_min':
        # Minimizar costo pero exigir que el F1 no caiga más del 5% respecto al mejor F1
        f1_min = max(f1_scores) * 0.95
        costos_filtrados = [c if f >= f1_min else np.inf
                            for c, f in zip(costos, f1_scores)]
        idx_optimo = int(np.argmin(costos_filtrados))

    else:
        idx_optimo = int(np.argmin(costos))

    return umbrales[idx_optimo], umbrales, costos, f1_scores
