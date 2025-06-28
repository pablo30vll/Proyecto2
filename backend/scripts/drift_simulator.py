import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_drift_scenario(
    original_data_path,
    output_new_data_dir, 
    drift_type="strong",
    sample_fraction=0.8
):
    print(f"CREANDO ESCENARIO DE DRIFT: {drift_type.upper()}")
    
    if os.path.exists(original_data_path):
        original_df = pd.read_parquet(original_data_path)
        print(f"Dataset original cargado: {original_df.shape}")
    else:
        raise FileNotFoundError(f"No se encuentra: {original_data_path}")
    
    sample_size = int(len(original_df) * sample_fraction)
    new_df = original_df.sample(n=sample_size, random_state=42).copy()
    
    print(f"Muestra base: {new_df.shape}")
    
    if drift_type == "none":
        print("Sin drift - datos idénticos")
        
    elif drift_type == "mild":
        print("Drift suave - cambios menores")
        
        if 'Y' in new_df.columns:
            new_df['Y'] = new_df['Y'] + np.random.normal(5, 2, len(new_df))
        if 'X' in new_df.columns:
            new_df['X'] = new_df['X'] + np.random.normal(3, 1.5, len(new_df))
        if 'num_deliver_per_week' in new_df.columns:
            new_df['num_deliver_per_week'] = np.clip(
                new_df['num_deliver_per_week'] + np.random.normal(0.5, 0.3, len(new_df)),
                1, 10
            )
        if 'customer_type' in new_df.columns:
            mask = np.random.random(len(new_df)) < 0.1
            current_types = new_df['customer_type'].unique()
            new_df.loc[mask, 'customer_type'] = np.random.choice(
                current_types, size=mask.sum()
            )
            
    elif drift_type == "strong":
        print("Drift fuerte - cambios significativos")
        
        if 'Y' in new_df.columns:
            new_df['Y'] = new_df['Y'] * np.random.normal(1.3, 0.2, len(new_df))
        if 'X' in new_df.columns:
            new_df['X'] = new_df['X'] * np.random.normal(0.7, 0.15, len(new_df))
        if 'size' in new_df.columns:
            new_df['size'] = new_df['size'] * np.random.normal(1.5, 0.3, len(new_df))
        if 'num_deliver_per_week' in new_df.columns:
            new_df['num_deliver_per_week'] = np.clip(
                new_df['num_deliver_per_week'] * np.random.normal(1.4, 0.4, len(new_df)),
                1, 15
            )
        if 'customer_type' in new_df.columns:
            mask = np.random.random(len(new_df)) < 0.3
            new_df.loc[mask, 'customer_type'] = 'A'
        if 'brand' in new_df.columns:
            mask = np.random.random(len(new_df)) < 0.25
            available_brands = new_df['brand'].unique()
            if len(available_brands) > 0:
                new_df.loc[mask, 'brand'] = available_brands[0]
    
    if 'week' in new_df.columns:
        new_df['week'] = pd.to_datetime(new_df['week']) + timedelta(weeks=1)
    
    os.makedirs(output_new_data_dir, exist_ok=True)
    
    output_path = os.path.join(output_new_data_dir, "dataset_with_drift.parquet")
    new_df.to_parquet(output_path)
    
    try:
        if all(col in new_df.columns for col in ['customer_id', 'product_id', 'week']):
            transactions = new_df[new_df['comprado'] == 1][['customer_id', 'product_id', 'week']].copy()
            transactions['order_id'] = range(1, len(transactions) + 1)
            transactions['purchase_date'] = transactions['week']
            transactions = transactions[['order_id', 'customer_id', 'product_id', 'purchase_date']]
            transactions.to_parquet(os.path.join(output_new_data_dir, "transacciones.parquet"))
        
        client_cols = ['customer_id'] + [col for col in new_df.columns 
                                         if col.startswith(('customer_', 'region_', 'zone_', 'Y', 'X', 'num_'))]
        if client_cols:
            clients = new_df[client_cols].drop_duplicates('customer_id')
            clients.to_parquet(os.path.join(output_new_data_dir, "clientes.parquet"))
        
        product_cols = ['product_id'] + [col for col in new_df.columns 
                                         if col in ['brand', 'category', 'sub_category', 'segment', 'package', 'size']]
        if product_cols:
            products = new_df[product_cols].drop_duplicates('product_id')
            products.to_parquet(os.path.join(output_new_data_dir, "productos.parquet"))
    
    except Exception as e:
        print(f"Error creando archivos auxiliares: {e}")
        print("Solo se guardó el dataset principal")
    
    stats = {
        "drift_type": drift_type,
        "original_size": len(original_df),
        "new_size": len(new_df),
        "sample_fraction": sample_fraction,
        "output_path": output_path,
        "files_created": os.listdir(output_new_data_dir),
        "timestamp": datetime.now().isoformat()
    }
    
    if drift_type != "none":
        print("VERIFICACIÓN DE DRIFT CREADO:")
        for col in ['Y', 'X', 'size']:
            if col in original_df.columns and col in new_df.columns:
                orig_mean = original_df[col].mean()
                new_mean = new_df[col].mean()
                change_pct = ((new_mean - orig_mean) / orig_mean) * 100
                print(f"{col}: {orig_mean:.2f} → {new_mean:.2f} ({change_pct:+.1f}%)")
    
    print(f"Datos con drift guardados en: {output_new_data_dir}")
    print(f"Archivos creados: {stats['files_created']}")
    
    return stats

def test_drift_detection_locally(dataset_path, new_data_path):
    print("PROBANDO DRIFT DETECTION LOCALMENTE...")
    try:
        from scripts import detect_drift
        result = detect_drift(
            reference_dataset_path=dataset_path,
            new_dataset_path=new_data_path,
            threshold=0.05
        )
        print("Resultado del test:")
        print(f"¿Drift detectado? {'SÍ' if result['drift_detected'] else 'NO'}")
        if 'summary' in result:
            print(f"Columnas analizadas: {result['summary']['total_columns_analyzed']}")
            print(f"Columnas con drift: {result['summary']['columns_with_drift']}")
        return result
    except Exception as e:
        print(f"Error en test local: {e}")
        return None

def create_no_drift_scenario():
    return create_drift_scenario(
        original_data_path="/tmp/sodai/tmp/dataset_limpio.parquet",
        output_new_data_dir="/tmp/sodai/data/new_no_drift",
        drift_type="none"
    )

def create_mild_drift_scenario():
    return create_drift_scenario(
        original_data_path="/tmp/sodai/tmp/dataset_limpio.parquet",
        output_new_data_dir="/tmp/sodai/data/new_mild_drift", 
        drift_type="mild"
    )

def create_strong_drift_scenario():
    return create_drift_scenario(
        original_data_path="/tmp/sodai/tmp/dataset_limpio.parquet",
        output_new_data_dir="/tmp/sodai/data/new_strong_drift",
        drift_type="strong"
    )
