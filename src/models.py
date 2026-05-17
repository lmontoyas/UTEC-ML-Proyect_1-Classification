# src/models.py
from sklearn.metrics import confusion_matrix

def calcular_costo_negocio(y_true, y_pred, costo_retencion_fp=20):
    """
    Calcula el impacto financiero basado en los errores del modelo.
    Costo de Retención (FP) = $20.
    Costo de Adquisición/Pérdida (FN) = 15x el costo de retención ($300).
    """
    multiplicador_adquisicion = 15 
    costo_perdida_fn = costo_retencion_fp * multiplicador_adquisicion
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    perdida_por_fp = fp * costo_retencion_fp
    perdida_por_fn = fn * costo_perdida_fn
    costo_total = perdida_por_fp + perdida_por_fn
    
    return {
        'Falsos Positivos (Gastos Innecesarios)': fp,
        'Falsos Negativos (Fugas No Detectadas)': fn,
        'Costo Falsos Positivos ($)': perdida_por_fp,
        'Costo Falsos Negativos ($)': perdida_por_fn,
        'COSTO TOTAL ($)': costo_total
    }