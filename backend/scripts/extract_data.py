# airflow/scripts/extract_data.py

import pandas as pd
import os

def load_raw_data(data_dir):
    """
    Carga los archivos principales desde el directorio indicado.
    Retorna tres DataFrames: transacciones, clientes, productos.
    """
    transacciones = pd.read_parquet(os.path.join(data_dir, "transacciones.parquet"))
    clientes      = pd.read_parquet(os.path.join(data_dir, "clientes.parquet"))
    productos     = pd.read_parquet(os.path.join(data_dir, "productos.parquet"))
    return transacciones, clientes, productos
