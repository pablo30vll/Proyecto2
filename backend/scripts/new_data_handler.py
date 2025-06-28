import pandas as pd
import os
import shutil
from datetime import datetime, timedelta

def prepare_incremental_data(
    new_data_dir,
    historical_data_path,
    output_path,
    weeks_to_include=4
):
    print("PREPARANDO DATOS INCREMENTALES...")
    
    if os.path.exists(historical_data_path):
        historical_df = pd.read_parquet(historical_data_path)
        print(f"Datos históricos cargados: {historical_df.shape}")
        print(f"Rango histórico: {historical_df['week'].min()} a {historical_df['week'].max()}")
    else:
        print("No hay datos históricos, usando solo datos nuevos")
        historical_df = pd.DataFrame()
    
    new_transacciones_path = os.path.join(new_data_dir, "transacciones.parquet")
    if not os.path.exists(new_transacciones_path):
        raise FileNotFoundError(f"No se encuentran nuevas transacciones en {new_transacciones_path}")
    
    clientes_path = os.path.join(new_data_dir, "clientes.parquet")
    productos_path = os.path.join(new_data_dir, "productos.parquet")
    
    if not os.path.exists(clientes_path):
        print("No hay clientes.parquet en datos nuevos, usando datos históricos")
    
    if not os.path.exists(productos_path):
        print("No hay productos.parquet en datos nuevos, usando datos históricos")
    
    from scripts.extract_data import load_raw_data
    from scripts.preprocessing import preprocess_data
    
    try:
        transacciones, clientes, productos = load_raw_data(new_data_dir)
        new_dataset = preprocess_data(transacciones, clientes, productos)
        
        print(f"Nuevos datos procesados: {new_dataset.shape}")
        print(f"Rango nuevos datos: {new_dataset['week'].min()} a {new_dataset['week'].max()}")
        
        if not historical_df.empty:
            latest_historical_weeks = sorted(historical_df['week'].unique())[-weeks_to_include:]
            historical_recent = historical_df[historical_df['week'].isin(latest_historical_weeks)]
            
            print(f"Semanas históricas incluidas: {len(latest_historical_weeks)}")
            print(f"Datos históricos recientes: {historical_recent.shape}")
            
            combined_df = pd.concat([historical_recent, new_dataset], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=['customer_id', 'product_id', 'week'], 
                keep='last'
            )
            
            print(f"Dataset combinado: {combined_df.shape}")
        else:
            combined_df = new_dataset
            print(f"Usando solo datos nuevos: {combined_df.shape}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_parquet(output_path)
        
        stats = {
            "success": True,
            "historical_samples": len(historical_df) if not historical_df.empty else 0,
            "new_samples": len(new_dataset),
            "combined_samples": len(combined_df),
            "historical_weeks": len(historical_df['week'].unique()) if not historical_df.empty else 0,
            "new_weeks": len(new_dataset['week'].unique()),
            "combined_weeks": len(combined_df['week'].unique()),
            "weeks_included_from_historical": weeks_to_include,
            "date_range": {
                "min_week": str(combined_df['week'].min()),
                "max_week": str(combined_df['week'].max())
            }
        }
        
        print("Datos incrementales preparados exitosamente")
        return stats
        
    except Exception as e:
        print(f"Error procesando nuevos datos: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def simulate_new_week_data(
    original_data_dir,
    output_new_data_dir,
    weeks_ahead=1,
    sample_fraction=0.8
):
    print(f"SIMULANDO DATOS NUEVOS ({weeks_ahead} semanas adelante)...")
    
    from scripts.extract_data import load_raw_data
    
    transacciones, clientes, productos = load_raw_data(original_data_dir)
    
    transacciones["purchase_date"] = pd.to_datetime(transacciones["purchase_date"])
    
    last_date = transacciones["purchase_date"].max()
    target_start = last_date + timedelta(weeks=weeks_ahead-1)
    target_end = target_start + timedelta(days=6)
    
    print(f"Última fecha original: {last_date}")
    print(f"Simulando semana: {target_start} a {target_end}")
    
    simulation_start = last_date - timedelta(weeks=4)
    simulation_end = simulation_start + timedelta(days=6)
    
    simulation_data = transacciones[
        (transacciones["purchase_date"] >= simulation_start) & 
        (transacciones["purchase_date"] <= simulation_end)
    ].copy()
    
    if len(simulation_data) == 0:
        print("No hay datos para simular, usando muestra aleatoria")
        simulation_data = transacciones.sample(n=min(1000, len(transacciones)))
    
    date_diff = target_start - simulation_start
    simulation_data["purchase_date"] = simulation_data["purchase_date"] + date_diff
    
    if sample_fraction < 1.0:
        simulation_data = simulation_data.sample(frac=sample_fraction, random_state=42)
    
    print(f"Datos simulados: {len(simulation_data)} transacciones")
    
    os.makedirs(output_new_data_dir, exist_ok=True)
    
    simulation_data.to_parquet(os.path.join(output_new_data_dir, "transacciones.parquet"))
    
    shutil.copy2(
        os.path.join(original_data_dir, "clientes.parquet"),
        os.path.join(output_new_data_dir, "clientes.parquet")
    )
    shutil.copy2(
        os.path.join(original_data_dir, "productos.parquet"),
        os.path.join(output_new_data_dir, "productos.parquet")
    )
    
    print(f"Datos nuevos simulados guardados en: {output_new_data_dir}")
    
    return {
        "simulated_transactions": len(simulation_data),
        "target_week_start": str(target_start),
        "target_week_end": str(target_end),
        "sample_fraction": sample_fraction
    }

def backup_historical_data(current_dataset_path, backup_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"dataset_backup_{timestamp}.parquet"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copy2(current_dataset_path, backup_path)
    
    print(f"Backup creado: {backup_path}")
    return backup_path
