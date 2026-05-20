from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# PASO 1 — LIMPIEZA Y FEATURE ENGINEERING

def clean_and_engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Paso 1: Limpieza básica, homologación y creac|ión de nuevas variables.
    """
    df = df.copy()
    
    # 1. Limpiar 'TotalCharges' (espacios a nulos, luego a números, luego rellenar ceros)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')


    # Imputar los valores nulos (los clientes nuevos con tenure=0 no tienen TotalCharges aún)
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 1.2 Homologar categorías: 'No internet service' y 'No phone service' -> 'No'
    cols_to_fix = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_to_fix:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    # Mapear las variables booleanas como 1 o 0
    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    """
    Paso 2: Feature engineering: Creación de nuevas variables predictoras.
    """
    
    # 4.1  Ratio Tenure / MonthlyCharges  (clientes baratos con larga permanencia)
    df['Tenure_to_Charge_Ratio'] = np.where(df['tenure'] == 0, 0.0, df['tenure'] / (df['MonthlyCharges'] + 1e-5))

    # 4.2  ¿Carga mensual es mayor que antes?
    monthly_avg_hist = df['TotalCharges'] / (df['tenure'] + 1e-5)
    df['Charge_Growth'] = np.where(
        df['tenure'] == 0,
        1.0,                                                        # sin historial: ratio neutro
        df['MonthlyCharges'] / (monthly_avg_hist + 1e-5)
    )

    # 4.3  Número de servicios adicionales contratados (0–6)
    addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Total_Addons'] = df[addon_cols].sum(axis=1)

    # 4.4  Riesgo fibra sin soporte: señal de insatisfacción
    df['Risk_Fiber_NoSupport'] = (
        (df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 0)
    ).astype(int)

    # 4.5  Pago manual (cheque físico o electrónico no-automático)
    df['Is_ManualPayment'] = (~df['PaymentMethod'].str.contains('automatic', case=False, na=False)).astype(int)

    # 4.6  Cliente solitario sin apoyo familiar
    df['Solo_User'] = ((df['Partner'] == 0) & (df['Dependents'] == 0)).astype(int)

    # 4.7  Antigüedad en segmentos (bins) — a mayor tiempo, mayor lealtad
    df['Tenure_Bin'] = pd.cut(
        df['tenure'],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # 4.8  Contrato codificado como ordinal (mayor plazo = menor riesgo)
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['Contract_Ordinal'] = df['Contract'].map(contract_map)
    
    """
    Paso 3: Mapear la variable objetivo.
    """

    # Mapear la variable objetivo a binario (Solo en el set de entrenamiento)
    if is_train and 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    return df


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Aplica One-Hot Encoding y transformación Yeo-Johnson.
    El scaler se ajusta ÚNICAMENTE sobre train_df para evitar Data Leakage.

    Retorna:
        X_train, X_val, X_test, y_train, y_val, test_ids
    """

    # 1. Guardar IDs para el envío final (solo para Kaggle)
    test_ids = test_df['customerID']
    
    # 2. Separar la variable objetivo
    y_train = train_df['Churn'].reset_index(drop=True)
    y_val   = val_df['Churn'].reset_index(drop=True)

    drop_cols_train = ['customerID', 'Churn']
    drop_cols_test  = ['customerID']

    X_train_raw = train_df.drop(columns=drop_cols_train)
    X_val_raw   = val_df.drop(columns=drop_cols_train)
    X_test_raw  = test_df.drop(columns=drop_cols_test)
    
    # 4. One-Hot Encoding (InternetService, Contract, PaymentMethod, gender)
    X_train_enc = pd.get_dummies(X_train_raw, drop_first=True)
    X_val_enc   = pd.get_dummies(X_val_raw,   drop_first=True)
    X_test_enc  = pd.get_dummies(X_test_raw,  drop_first=True)
    
    # 5. Alinear columnas (Obligamos a Val y Test a tener las mismas columnas que Train)
    X_train_enc, X_val_enc  = X_train_enc.align(X_val_enc,  join='left', axis=1, fill_value=0)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)
    
    # 6. Transformación Yeo-Johnson y Escalado
    num_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Tenure_to_Charge_Ratio', 'Charge_Growth',
        'Total_Addons', 'Tenure_Bin', 'Contract_Ordinal',
    ]
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    
    # Ajustamos (fit) y transformamos SOLO en Train
    X_train_enc[num_cols] = pt.fit_transform(X_train_enc[num_cols])   # fit + transform
    
    # Solo transformamos (usando las reglas del Train) en Val y Test
    X_val_enc[num_cols]   = pt.transform(X_val_enc[num_cols])          # solo transform
    X_test_enc[num_cols]  = pt.transform(X_test_enc[num_cols])         # solo transform
    
    return (
        X_train_enc.reset_index(drop=True),
        X_val_enc.reset_index(drop=True),
        X_test_enc.reset_index(drop=True),
        y_train,
        y_val,
        test_ids,
        pt            # retornamos el scaler para poder reutilizarlo
    )