import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mlflowe

# Wczytanie danych z pliku CSV
df = pd.read_csv('failures_data.csv')

# Sprawdzenie brakujących wartości i usunięcie ich
df = df.dropna()

# Usunięcie pola NAME
df = df.drop(['NAME'], axis=1)

# Konwersja dat na format numeryczny
df['DATE'] = pd.to_datetime(df['DATE'])
df['POTENTIAL_DATA'] = pd.to_datetime(df['POTENTIAL_DATA'])
df['DAYS_TO_POTENTIAL'] = (df['POTENTIAL_DATA'] - df['DATE']).dt.days

# Zakodowanie wartości kategorycznych
df['FAILURE_TYPE'] = df['FAILURE_TYPE'].astype('category').cat.codes
df['STATUS'] = df['STATUS'].astype('category').cat.codes

# Usunięcie oryginalnych kolumn dat
df = df.drop(['DATE', 'POTENTIAL_DATA'], axis=1)

# Definiowanie funkcji augmentacji danych
def augment_data(data, num_records):
    augmented_data = []
    for _ in range(num_records):
        record = data.sample(1).copy()
        
        # Dodanie szumu do cech numerycznych
        record['POTENTIAL_PRICE'] += np.random.normal(0, 100)  # Dodanie szumu do ceny
        record['DAYS_TO_POTENTIAL'] += np.random.normal(0, 5)  # Dodanie szumu do liczby dni
        
        # Upewnij się, że wartości pozostają w rozsądnych granicach
        record['POTENTIAL_PRICE'] = max(0, record['POTENTIAL_PRICE'].values[0])
        record['DAYS_TO_POTENTIAL'] = max(0, record['DAYS_TO_POTENTIAL'].values[0])
        
        augmented_data.append(record)
    
    augmented_df = pd.concat(augmented_data, ignore_index=True)
    return augmented_df

# Generowanie 100 dodatkowych rekordów
augmented_df = augment_data(df, 100)

# Normalizacja danych
scaler = MinMaxScaler()
df[['POTENTIAL_PRICE', 'DAYS_TO_POTENTIAL']] = scaler.fit_transform(df[['POTENTIAL_PRICE', 'DAYS_TO_POTENTIAL']])
augmented_df[['POTENTIAL_PRICE', 'DAYS_TO_POTENTIAL']] = scaler.fit_transform(augmented_df[['POTENTIAL_PRICE', 'DAYS_TO_POTENTIAL']])

# Zapisanie sformatowanych danych do nowych plików CSV
df.to_csv('failures_data_formatted.csv', index=False)
augmented_df.to_csv('failures_data_augmented.csv', index=False)