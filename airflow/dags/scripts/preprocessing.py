import pandas as pd
import numpy as np
import itertools

def preprocess_data(transacciones, clientes, productos):
    transacciones = transacciones.drop_duplicates()
    clientes = clientes.drop_duplicates()
    productos = productos.drop_duplicates()

    transacciones["purchase_date"] = pd.to_datetime(transacciones["purchase_date"])
    df = transacciones.merge(clientes, on="customer_id", how="left")
    df = df.merge(productos, on="product_id", how="left")

    df['week'] = df['purchase_date'].dt.to_period('W').apply(lambda r: r.start_time)

    compras_semanales = (
        df.groupby(['customer_id', 'product_id', 'week'])
          .agg(comprado=('order_id', 'count'))
          .reset_index()
    )
    compras_semanales['comprado'] = compras_semanales['comprado'].apply(lambda x: 1 if x > 0 else 0)

    rango_semanas = pd.date_range(df['week'].min(), df['week'].max(), freq='W-MON')
    clientes_unicos = df['customer_id'].unique()
    productos_unicos = df['product_id'].unique()

    base = pd.DataFrame(
        list(itertools.product(clientes_unicos, productos_unicos, rango_semanas)),
        columns=['customer_id', 'product_id', 'week']
    )
    dataset = base.merge(compras_semanales, on=['customer_id', 'product_id', 'week'], how='left')
    dataset['comprado'] = dataset['comprado'].fillna(0).astype(int)

    dataset = dataset.merge(clientes, on='customer_id', how='left')
    dataset = dataset.merge(productos, on='product_id', how='left')

    constant_cols = [col for col in dataset.columns if dataset[col].nunique() == 1]
    dataset = dataset.drop(columns=constant_cols)

    return dataset
