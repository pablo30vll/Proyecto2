import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import itertools

def predict_next_week(model_path, dataset_path, output_path):
    print("Iniciando generación de predicciones...")

    model = joblib.load(model_path)
    dataset = pd.read_parquet(dataset_path)

    print(f"Modelo cargado: {model_path}")
    print(f"Dataset cargado: {dataset.shape}")
    print(f"Columnas en dataset: {list(dataset.columns)}")

    last_week = dataset['week'].max()
    next_week = pd.Timestamp(last_week) + timedelta(days=7)

    print(f"Última semana en datos: {last_week}")
    print(f"Predicciones para: {next_week}")

    clientes_unicos = dataset['customer_id'].unique()
    productos_unicos = dataset['product_id'].unique()

    print(f"Clientes únicos: {len(clientes_unicos)}")
    print(f"Productos únicos: {len(productos_unicos)}")

    df_pred = pd.DataFrame(
        list(itertools.product(clientes_unicos, productos_unicos)),
        columns=['customer_id', 'product_id']
    )
    df_pred['week'] = next_week

    print(f"Combinaciones creadas: {df_pred.shape}")

    print("Extrayendo información de clientes...")
    clientes_info = (
        dataset.sort_values('week')
        .groupby('customer_id')
        .last()
        .reset_index()
    )

    cliente_cols = ['customer_id']
    for col in dataset.columns:
        if col in ['customer_type', 'region_id', 'zone_id', 'Y', 'X', 
                   'num_deliver_per_week', 'num_visit_per_week'] or col.startswith('customer_'):
            if col in clientes_info.columns:
                cliente_cols.append(col)

    clientes_df = clientes_info[cliente_cols].copy()
    clientes_df = clientes_df.loc[:, ~clientes_df.columns.duplicated()]
    print(f"Info clientes extraída: {clientes_df.shape}, columnas: {list(clientes_df.columns)}")

    print("Extrayendo información de productos...")
    productos_info = (
        dataset.sort_values('week')
        .groupby('product_id')
        .last()
        .reset_index()
    )

    producto_cols = ['product_id']
    for col in dataset.columns:
        if col in ['brand', 'category', 'sub_category', 'segment', 'package', 'size'] or col.startswith('product_'):
            if col in productos_info.columns:
                producto_cols.append(col)

    productos_df = productos_info[producto_cols].copy()
    productos_df = productos_df.loc[:, ~productos_df.columns.duplicated()]
    print(f"Info productos extraída: {productos_df.shape}, columnas: {list(productos_df.columns)}")

    print("Combinando información...")
    df_pred = df_pred.merge(clientes_df, on='customer_id', how='left')
    df_pred = df_pred.merge(productos_df, on='product_id', how='left')

    print(f"Dataset de predicción después de merge: {df_pred.shape}")
    print(f"Columnas finales: {list(df_pred.columns)}")

    drop_cols = ['comprado', 'customer_id', 'product_id']
    expected_features = [col for col in df_pred.columns if col not in drop_cols]

    print(f"Features para predicción: {len(expected_features)}")
    X_pred = df_pred[expected_features].copy()

    print(f"Shape final para predicción: {X_pred.shape}")

    missing_info = X_pred.isnull().sum()
    if missing_info.sum() > 0:
        print("Valores faltantes detectados:")
        for col, missing in missing_info[missing_info > 0].items():
            print(f"{col}: {missing}")
        for col in X_pred.columns:
            if X_pred[col].dtype in ['object', 'category']:
                X_pred[col] = X_pred[col].fillna('Unknown')
            else:
                X_pred[col] = X_pred[col].fillna(X_pred[col].median())

    print("Realizando predicciones...")

    try:
        y_prob = model.predict_proba(X_pred)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        print("Predicciones completadas")
        print(f"Min: {y_prob.min():.3f}")
        print(f"Max: {y_prob.max():.3f}")
        print(f"Mean: {y_prob.mean():.3f}")
        print(f"Predicciones positivas: {y_pred.sum():,}")

    except Exception as e:
        print(f"Error en predicción: {e}")
        print("Información de debug:")
        print(f"Columnas en X_pred: {list(X_pred.columns)}")
        print(f"Shape de X_pred: {X_pred.shape}")
        print(f"Tipos de datos: {X_pred.dtypes.value_counts()}")

        try:
            if hasattr(model, 'named_steps'):
                preprocessor = model.named_steps.get('preproc')
                if hasattr(preprocessor, 'transformers_'):
                    print("Transformers del modelo:")
                    for name, transformer, cols in preprocessor.transformers_:
                        print(f"{name}: {cols}")
        except Exception as e2:
            print(f"No se pudo obtener info del modelo: {e2}")

        raise e

    df_pred['probabilidad_compra'] = y_prob
    df_pred['prediccion_compra'] = y_pred

    df_final = df_pred[df_pred['prediccion_compra'] == 1][
        ['customer_id', 'product_id', 'week', 'probabilidad_compra']
    ].copy()

    df_final = df_final.sort_values('probabilidad_compra', ascending=False)
    df_final.to_csv(output_path, index=False)

    print(f"Predicciones exportadas a: {output_path}")
    print("Resumen final:")
    print(f"Total combinaciones evaluadas: {len(df_pred):,}")
    print(f"Predicciones positivas: {len(df_final):,}")
    print(f"Tasa de predicciones positivas: {len(df_final)/len(df_pred)*100:.2f}%")
    print(f"Probabilidad promedio (positivas): {df_final['probabilidad_compra'].mean():.3f}")

    return {
        "total_combinations": len(df_pred),
        "positive_predictions": len(df_final),
        "positive_rate": len(df_final) / len(df_pred),
        "avg_probability": df_final['probabilidad_compra'].mean() if len(df_final) > 0 else 0
    }
