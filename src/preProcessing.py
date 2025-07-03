import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
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
    imputer = KNNImputer(n_neighbors=n_neighbors)
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
