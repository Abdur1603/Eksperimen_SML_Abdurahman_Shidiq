import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def clean_and_encode_data(df):
    print("Cleaning and Encoding data...")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    df['TotalCharges'].fillna(0, inplace=True) 

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) <= 2:
                df[col] = le.fit_transform(df[col])
    
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def scale_and_split_data(df, target_col='Churn'):
    """Melakukan scaling dan split data train/test."""
    print("Scaling and Splitting data...")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Dimensi X_train: {X_train.shape}")
    print(f"Dimensi X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, output_dir):
    """Menyimpan data yang sudah diproses."""
    print(f"Saving processed data to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print("Data processing complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, '../telco_customer_churn_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    output_path = os.path.join(base_dir, 'telco_customer_churn_preprocessing')
    
    df = load_data(input_path)
    df_clean = clean_and_encode_data(df)
    X_train, X_test, y_train, y_test = scale_and_split_data(df_clean)
    save_data(X_train, X_test, y_train, y_test, output_path)