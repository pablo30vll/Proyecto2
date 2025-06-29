import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import os

def detect_drift(
    reference_dataset_path, 
    new_dataset_path, 
    numeric_columns=None, 
    categorical_columns=None,
    threshold=0.05,
    min_samples=30
):
    print(f"DETECTANDO DRIFT...")
    print(f"Dataset de referencia: {reference_dataset_path}")
    print(f"Dataset nuevo: {new_dataset_path}")
    
    if not os.path.exists(reference_dataset_path):
        print(f"No existe dataset de referencia: {reference_dataset_path}")
        return {"drift_detected": True, "reason": "no_reference_dataset"}
    
    if not os.path.exists(new_dataset_path):
        print(f"No existe dataset nuevo: {new_dataset_path}")
        return {"drift_detected": True, "reason": "no_new_dataset"}
    
    try:
        reference_data = pd.read_parquet(reference_dataset_path)
        new_data = pd.read_parquet(new_dataset_path)
        
        print(f"Datos de referencia: {reference_data.shape}")
        print(f"Datos nuevos: {new_data.shape}")
        
        if numeric_columns is None:
            numeric_columns = reference_data.select_dtypes(include=np.number).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in ['customer_id', 'product_id', 'comprado']]
        
        if categorical_columns is None:
            categorical_columns = reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_columns = [col for col in categorical_columns if col not in ['customer_id', 'product_id', 'week', 'purchase_date']]
        
        print(f"Analizando columnas numéricas: {numeric_columns}")
        print(f"Analizando columnas categóricas: {categorical_columns}")
        
        drift_results = {
            "drift_detected": False,
            "numeric_drift": [],
            "categorical_drift": [],
            "test_details": {},
            "summary": {}
        }
        
        for col in numeric_columns:
            if col in reference_data.columns and col in new_data.columns:
                ref_values = reference_data[col].dropna()
                new_values = new_data[col].dropna()
                
                if len(ref_values) >= min_samples and len(new_values) >= min_samples:
                    stat, pvalue = ks_2samp(ref_values, new_values)
                    
                    drift_results["test_details"][col] = {
                        "test": "kolmogorov_smirnov",
                        "statistic": float(stat),
                        "pvalue": float(pvalue),
                        "threshold": threshold,
                        "drift": pvalue < threshold,
                        "ref_samples": len(ref_values),
                        "new_samples": len(new_values)
                    }
                    
                    if pvalue < threshold:
                        drift_results["numeric_drift"].append(col)
                        print(f"DRIFT DETECTADO en {col}: p-value={pvalue:.4f}")
                    else:
                        print(f"Sin drift en {col}: p-value={pvalue:.4f}")
                else:
                    print(f"Insuficientes muestras para {col}: ref={len(ref_values)}, new={len(new_values)}")
        
        for col in categorical_columns:
            if col in reference_data.columns and col in new_data.columns:
                ref_values = reference_data[col].dropna()
                new_values = new_data[col].dropna()
                
                if len(ref_values) >= min_samples and len(new_values) >= min_samples:
                    ref_counts = ref_values.value_counts()
                    new_counts = new_values.value_counts()
                    
                    all_categories = sorted(set(ref_counts.index) | set(new_counts.index))
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    new_aligned = [new_counts.get(cat, 0) for cat in all_categories]
                    
                    if len(all_categories) > 1 and sum(ref_aligned) > 0 and sum(new_aligned) > 0:
                        contingency_table = np.array([ref_aligned, new_aligned])
                        
                        try:
                            chi2, pvalue, dof, expected = chi2_contingency(contingency_table)
                            
                            drift_results["test_details"][col] = {
                                "test": "chi_square",
                                "statistic": float(chi2),
                                "pvalue": float(pvalue),
                                "threshold": threshold,
                                "drift": pvalue < threshold,
                                "ref_samples": len(ref_values),
                                "new_samples": len(new_values),
                                "categories": len(all_categories)
                            }
                            
                            if pvalue < threshold:
                                drift_results["categorical_drift"].append(col)
                                print(f"DRIFT DETECTADO en {col}: p-value={pvalue:.4f}")
                            else:
                                print(f"Sin drift en {col}: p-value={pvalue:.4f}")
                        except Exception as e:
                            print(f"Error en test chi-cuadrado para {col}: {e}")
                    else:
                        print(f"Insuficiente variabilidad para {col}")
                else:
                    print(f"Insuficientes muestras para {col}: ref={len(ref_values)}, new={len(new_values)}")
        
        total_drift_columns = len(drift_results["numeric_drift"]) + len(drift_results["categorical_drift"])
        drift_results["drift_detected"] = total_drift_columns > 0
        
        drift_results["summary"] = {
            "total_columns_analyzed": len(numeric_columns) + len(categorical_columns),
            "numeric_columns_analyzed": len(numeric_columns),
            "categorical_columns_analyzed": len(categorical_columns),
            "columns_with_drift": total_drift_columns,
            "drift_percentage": total_drift_columns / max(1, len(numeric_columns) + len(categorical_columns)) * 100
        }
        
        print(f"\nRESUMEN DE DRIFT:")
        print(f"   Columnas analizadas: {drift_results['summary']['total_columns_analyzed']}")
        print(f"   Columnas con drift: {total_drift_columns}")
        print(f"   Porcentaje de drift: {drift_results['summary']['drift_percentage']:.1f}%")
        print(f"   ¿Drift detectado?: {'SÍ' if drift_results['drift_detected'] else 'NO'}")
        
        if drift_results["numeric_drift"]:
            print(f"   Numéricas con drift: {drift_results['numeric_drift']}")
        if drift_results["categorical_drift"]:
            print(f"   Categóricas con drift: {drift_results['categorical_drift']}")
        
        if drift_results["drift_detected"]:
            print(f"[Drift Detection] DRIFT DETECTADO en {len(drift_results['numeric_drift'])} columnas numéricas y {len(drift_results['categorical_drift'])} columnas categóricas. Se debe reentrenar el modelo.")
        else:
            print(f"[Drift Detection] Sin drift detectado. Se reutiliza el modelo actual.")
        
        return drift_results
        
    except Exception as e:
        print(f"Error al detectar drift: {e}")
        return {
            "drift_detected": True, 
            "reason": "error_in_detection",
            "error": str(e)
        }

def save_drift_report(drift_results, output_path):
    import json
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(drift_results, f, indent=2, default=str)
    
    print(f"Reporte de drift guardado en: {output_path}")

def detect_drift_simple(old_dataset_path, new_dataset_path, threshold=0.05):
    result = detect_drift(old_dataset_path, new_dataset_path, threshold=threshold)
    return result.get("drift_detected", True)
