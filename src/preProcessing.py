import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import process, fuzz
from unidecode import unidecode
import re
import requests

def maskPrecio(df):
    resp = requests.get("https://api.bluelytics.com.ar/v2/latest")
    resp.raise_for_status()
    data = resp.json()

    blue_sell = data["blue"]["value_sell"]
    mask = df['Moneda'] == '$'
    df.loc[mask, 'Precio'] = df.loc[mask, 'Precio'] / blue_sell 
    df.drop(columns=['Moneda'], inplace=True)
    return df

def eliminarOutliers(df, column, training=True, predet_mu=0, predet_sigma=1, log=False):
    if log:
        df[column] = np.log(df[column])
    if training:
        mu = df[column].mean()
        sigma = df[column].std()
    else:
        mu = predet_mu
        sigma = predet_sigma
    df = df[np.abs(df[column] - mu) <= 3 * sigma].copy()
    if log:
        df[column] = np.exp(df[column])
    return df, mu, sigma

def limpiarKm(df):
    df['Kilómetros'] = (
        df['Kilómetros']
        .str.replace(r'\s*km$', '', regex=True)
        .str.replace(r'\.', '', regex=True)
        .astype(float)
    )
    return df

def filtrar_rango(df, column, min_val, max_val):
    return df[(df[column] >= min_val) & (df[column] <= max_val)].copy()

def code_decode(df, mappings, codificar=True):
    """
    Si codificar=True: transforma columnas categóricas a códigos numéricos.
    Si codificar=False: convierte códigos numéricos de vuelta a categorías originales.
    """
    for col, mapping in mappings.items():
        code_col = col.replace(' ', '_') + '_code'
        if codificar:
            df[code_col] = df[col].map(mapping)
        else:
            inv_map = {v: k for k, v in mapping.items()}
            df[code_col] = np.round(df[code_col]).astype(int)
            df[col] = df[code_col].map(inv_map)
    return df


def imputar_columnas(train_df, val_df, test_df, cols, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=5)
    train_df[cols] = imputer.fit_transform(train_df[cols])
    val_df[cols] = imputer.transform(val_df[cols])
    test_df[cols] = imputer.transform(test_df[cols])

    return train_df, val_df, test_df


def graficar_histograma(data, columna, bins=100, escala='linear', log=False,
    titulo=None, xlabel=None, ylabel='Frecuencia', figsize=(10, 6), edgecolor='black'):

    valores = data[columna]
    if log:
        valores = np.log(valores)
        xlabel = xlabel or f"log_{columna}"
    else:
        xlabel = xlabel or columna

    plt.figure(figsize=figsize)
    plt.hist(valores, bins=bins, edgecolor=edgecolor)
    if escala != 'linear':
        plt.xscale(escala)
    plt.title(titulo or f'Distribución de la columna "{xlabel}"')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def normalize(text):
    return unidecode(str(text)).lower().strip()

def mapping(df, col, overrides, canonical=None):
    if col == 'Color':
        df['color_norm'] = df['Color'].apply(normalize)
        df['color_estandar'] = df['color_norm'].apply(lambda x: map_color(x, overrides, canonical))
        df['Color'] = df['color_estandar']
        df.drop(columns=['color_norm', 'color_estandar'], inplace=True)
        df.dropna(subset=['Color'], inplace=True)
    if col == 'Marca':
        df['Marca'] = df['Marca'].apply(lambda x: normalize_brand(x, overrides, canonical))
    if col == 'Modelo':
        df['Modelo'] = df['Modelo'].apply(lambda x: normalize_model(x, overrides))
    return df 

def map_color(name, overrides, canonical):
    canonical_lower = [c.lower() for c in canonical]
    if name in overrides:
        return overrides[name]
    result = process.extractOne(name, canonical_lower, scorer=fuzz.token_sort_ratio)
    if result is not None:
        match, score = result[0], result[1]
        if score >= 40:
            return canonical[canonical_lower.index(match)]
    return None

def normalize_brand(name: str, overrides=None, canonical=None) -> str:
    canonical_lower = [c.lower() for c in canonical]
    if pd.isna(name):
        return 'otro'
    n = normalize(name)
    if n in overrides:
        return overrides[n]
    res = process.extractOne(n, canonical_lower, scorer=fuzz.token_sort_ratio)
    if res:
        match, score = res[0], res[1]
        if score >= 80:
            return canonical[canonical_lower.index(match)]
    return 'otro'

def normalize_model(name: str, overrides) -> str:
    if pd.isna(name):
        return name
    n = normalize(name)
    return overrides.get(n, n)

def extraer_litros(s):
    """Extrae litros de un string buscando ‘número + L’; si no, toma el primer número."""
    if pd.isna(s):
        return np.nan
    text = str(s).lower()
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*[lL]', text)
    if m:
        return float(m.group(1).replace(',', '.'))
    m2 = re.match(r'(\d+(?:[\.,]\d+)?)', text)
    return float(m2.group(1).replace(',', '.')) if m2 else np.nan

def tipo_motor_extendido(motor, version):
    """
    Clasifica el tipo de motor buscando palabras clave en Motor y, si no encuentra,
    en Versión. Devuelve una de:
      'eléctrico', 'híbrido', 'diésel', 'turbo', 'gas / gnc', 'atmosférico', 'desconocido'
    """
    m = str(motor).lower() if pd.notna(motor) else ""
    v = str(version).lower() if pd.notna(version) else ""
    
    if re.search(r'eléctr|plug', m):
        return 'eléctrico'
    if 'hibrid' in m:
        return 'híbrido'
    if 'diesel' in m or 'diésel' in m:
        return 'diésel'
    if 'turbo' in m:
        return 'turbo'
    if 'gnc' in m or 'gpl' in m:
        return 'gas / gnc'
    
    if re.search(r'eléctr|plug', v):
        return 'eléctrico'
    if 'hibrid' in v:
        return 'híbrido'
    if 'diesel' in v or 'diésel' in v:
        return 'diésel'
    if 'turbo' in v:
        return 'turbo'
    if 'gnc' in v or 'gpl' in v:
        return 'gas / gnc'
    
    return 'atmosférico' if (m or v) else 'desconocido'

def feature_engineering(df):

    df['Edad'] = 2025 - df['Año']
    df['log_km'] = np.log1p(df['Kilómetros'])
    df['Edad_x_logkm'] = df['Edad'] * df['log_km']
    df.drop(columns=['Año', 'Kilómetros'], inplace=True)
    df.drop(columns=['Título', 'Descripción'], inplace=True)

    return df