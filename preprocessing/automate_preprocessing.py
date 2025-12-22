import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Fungsi Load Data
def load_data(path):
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

# Fungsi Preprocessing (Sesuai Eksperimen Telco Churn)
def preprocess_data(df):
    print("Starting preprocessing...")
    df_clean = df.copy()

    # Bersihkan TotalCharges (String kosong -> NaN -> 0)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'].fillna(0, inplace=True)

    # Hapus CustomerID
    if 'customerID' in df_clean.columns:
        df_clean.drop('customerID', axis=1, inplace=True)

    # Label Encoding (Otomatis deteksi kolom 2 nilai unik)
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' and df_clean[col].nunique() == 2:
            df_clean[col] = le.fit_transform(df_clean[col])
            
    # Pastikan Target 'Churn' ter-encode (Yes=1, No=0)
    if df_clean['Churn'].dtype == 'object':
        df_clean['Churn'] = le.fit_transform(df_clean['Churn'])

    # 4. One-Hot Encoding (Untuk kategori > 2 nilai)
    # Kolom Telco yang perlu OHE
    ohe_candidates = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                      'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    # Filter hanya kolom yang benar-benar ada di dataset
    valid_ohe = [col for col in ohe_candidates if col in df_clean.columns]
    df_clean = pd.get_dummies(df_clean, columns=valid_ohe)

    # Scaling (Standarisasi Angka)
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    print(f"Preprocessing selesai. Dimensi akhir: {df_clean.shape}")
    return df_clean

# Fungsi Simpan
def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # Path Relatif dalam Repository GitHub
    # Input: Folder raw
    raw_path = 'telco_churn_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
    
    # Output: Folder preprocessing
    output_path = 'preprocessing/telco_churn_preprocessing/telco_churn_clean.csv'
    
    # Eksekusi
    try:
        df = load_data(raw_path)
        df_clean = preprocess_data(df)
        save_data(df_clean, output_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Pastikan file raw ada di folder yang benar.")