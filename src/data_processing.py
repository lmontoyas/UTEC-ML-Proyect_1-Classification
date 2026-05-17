from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def clean_and_engineer_features(df, is_train=True):
    """
    Paso 1: Limpieza básica, homologación y creac|ión de nuevas variables.
    """
    df_clean = df.copy()
    
    # 1. Limpiar 'TotalCharges' (espacios a nulos, luego a números, luego rellenar ceros)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(" ", np.nan)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])

    # Imputar los valores nulos (los clientes nuevos con tenure=0 no tienen TotalCharges aún)
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
    
    # 1.2 Homologar categorías: 'No internet service' y 'No phone service' -> 'No'
    cols_to_fix = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in cols_to_fix:
        df_clean[col] = df_clean[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    # Mapear las variables booleanas como 1 o 0
    yes_no_columns = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
    ]
    for col in yes_no_columns:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
    
    """
    Paso 2: Feature engineering: Creación de nuevas variables predictoras.
    """
    
    # 1. Ratio de costo por tiempo (sumamos 1e-5 para evitar divisiones por cero)
    df_clean['Tenure_to_Charge_Ratio'] = df_clean['tenure'] / (df_clean['MonthlyCharges'] + 1e-5)
    
    # 2. Interacción de riesgo: Tiene Fibra pero no tiene Soporte Técnico
    df_clean['Risk_Fiber_NoSupport'] = np.where(
        (df_clean['InternetService'] == 'Fiber optic') & (df_clean['TechSupport'] == 'No'), 1, 0
    )

    # 3. Fidelización (Total_Addons): Suma de servicios adicionales contratados (0 a 6)
    extra_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_clean['Total_Addons'] = (df_clean[extra_services] == 'Yes').sum(axis=1) 

    # 4. Dolor de Pago (Is_ManualPayment): 1 si el pago es manual (cheque), 0 si es automático
    df_clean['Is_ManualPayment'] = np.where(
        df_clean['PaymentMethod'].str.contains('automatic', case=False), 0, 1)

    # 5. Usuario Solitario (Solo_User): 1 si no tiene pareja (Partner='No') ni dependientes ('No')
    df_clean['Solo_User'] = np.where(
        (df_clean['Partner'] == 'No') & (df_clean['Dependents'] == 'No'), 1, 0)
    
    """
    Paso 3: Mapear la variable objetivo.
    """

    # Mapear la variable objetivo a binario (Solo en el set de entrenamiento)
    if is_train and 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    return df_clean


def preprocess_data(train_df, val_df, test_df):
    """
    Procesa los 3 conjuntos de datos asegurando que no haya Data Leakage.
    El escalador y las columnas se ajustan SOLO con train_df.
    """
    # 1. Guardar IDs para el envío final (solo Kaggle Test lo necesita)
    test_ids = test_df['customerID']
    
    # 2. Separar la variable objetivo (Test Kaggle no tiene Churn)
    y_train = train_df['Churn']
    y_val = val_df['Churn']
    
    # 3. Quitar columnas que no deben entrar al modelo
    X_train = train_df.drop(columns=['customerID', 'Churn'])
    X_val = val_df.drop(columns=['customerID', 'Churn'])
    X_test = test_df.drop(columns=['customerID'])
    
    # 4. One-Hot Encoding
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_val_encoded = pd.get_dummies(X_val, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    
    # 5. Alinear columnas (Obligamos a Val y Test a tener las mismas columnas que Train)
    X_train_encoded, X_val_encoded = X_train_encoded.align(X_val_encoded, join='left', axis=1, fill_value=0)
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    # 6. Transformación Yeo-Johnson y Escalado
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Tenure_to_Charge_Ratio']
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    
    # Ajustamos (fit) y transformamos SOLO en Train
    X_train_encoded[num_cols] = pt.fit_transform(X_train_encoded[num_cols])
    
    # Solo transformamos (usando las reglas del Train) en Val y Test
    X_val_encoded[num_cols] = pt.transform(X_val_encoded[num_cols])
    X_test_encoded[num_cols] = pt.transform(X_test_encoded[num_cols])
    
    return X_train_encoded, X_val_encoded, X_test_encoded, y_train, y_val, test_ids