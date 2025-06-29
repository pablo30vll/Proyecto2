import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from collections import Counter
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json

def get_ct_feature_names(ct: ColumnTransformer, input_cols):
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if transformer == 'drop' or (isinstance(transformer, str) and transformer == 'drop'):
            continue
        est = transformer.steps[-1][1] if isinstance(transformer, Pipeline) else transformer
        if hasattr(est, "get_feature_names_out"):
            try:
                names = est.get_feature_names_out(cols)
            except:
                names = est.get_feature_names_out(np.array(cols, dtype=object))
        else:
            names = cols
        feature_names.extend(list(names))
    return feature_names

def run_shap_analysis(model_path, val_path, output_dir="outputs/shap/"):
    print("EJECUTANDO ANÁLISIS DE INTERPRETABILIDAD...")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        final_pipe = joblib.load(model_path)
        val_df = pd.read_parquet(val_path)
        
        print(f"Modelo cargado desde: {model_path}")
        print(f"Datos de validación: {val_df.shape}")
        
        drop_cols = ['comprado', 'customer_id', 'product_id']
        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df['comprado']
        
        sample_size = min(1000, len(X_val))
        X_val_sub = X_val.sample(n=sample_size, random_state=42)
        y_val_sub = y_val.loc[X_val_sub.index]
        
        print(f"Muestra para análisis: {X_val_sub.shape}")
        
        preproc = final_pipe.named_steps['preproc']
        clf = final_pipe.named_steps['clf']
        input_columns = X_val.columns.tolist()
        feat_names = get_ct_feature_names(preproc, input_columns)
        
        counter = Counter()
        feat_names_unique = []
        for name in feat_names:
            if counter[name] == 0:
                feat_names_unique.append(name)
            else:
                feat_names_unique.append(f"{name}_{counter[name]}")
            counter[name] += 1
        
        print(f"Features después del preprocesamiento: {len(feat_names_unique)}")
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            idx_sorted = np.argsort(importances)[::-1][:15]
            plt.figure(figsize=(10, 8))
            plt.barh(
                np.array(feat_names_unique)[idx_sorted][::-1],
                importances[idx_sorted][::-1]
            )
            plt.xlabel("Importancia")
            plt.title("Top 15 Características más Importantes (Random Forest)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("Feature Importance guardado")
        
        print("Calculando Permutation Importance...")
        try:
            r = permutation_importance(
                final_pipe,
                X_val_sub,
                y_val_sub,
                n_repeats=5,
                random_state=42,
                n_jobs=1
            )
            perm_idx = r.importances_mean.argsort()[::-1][:15]
            plt.figure(figsize=(10, 8))
            plt.boxplot(
                r.importances[perm_idx].T,
                vert=False,
                labels=np.array(feat_names_unique)[perm_idx]
            )
            plt.xlabel("Importancia")
            plt.title("Permutation Importance (Top 15)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "permutation_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print("Permutation Importance guardado")
        except Exception as e:
            print(f"Error en Permutation Importance: {e}")
        
        print("Generando Partial Dependence Plots...")
        try:
            if hasattr(clf, 'feature_importances_'):
                top_features_idx = np.argsort(clf.feature_importances_)[::-1][:4]
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.ravel()
                X_val_prep = preproc.transform(X_val_sub)
                
                for i, feat_idx in enumerate(top_features_idx):
                    unique_vals = np.unique(X_val_prep[:, feat_idx])
                    if len(unique_vals) > 1:
                        try:
                            PartialDependenceDisplay.from_estimator(
                                clf,
                                X_val_prep,
                                features=[feat_idx],
                                ax=axes[i],
                                grid_resolution=20
                            )
                            axes[i].set_title(f"PDP: {feat_names_unique[feat_idx]}")
                        except Exception as e:
                            axes[i].text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                                       transform=axes[i].transAxes, ha='center')
                            axes[i].set_title(f"PDP: {feat_names_unique[feat_idx]} (Error)")
                    else:
                        axes[i].text(0.5, 0.5, "Sin variabilidad", 
                                   transform=axes[i].transAxes, ha='center')
                        axes[i].set_title(f"PDP: {feat_names_unique[feat_idx]} (Constante)")
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "partial_dependence_plots.png"), dpi=300, bbox_inches='tight')
                plt.close()
                print("Partial Dependence Plots guardados")
        except Exception as e:
            print(f"Error en Partial Dependence Plots: {e}")
        
        print("Generando estadísticas detalladas...")
        try:
            stats = {
                "model_type": "RandomForest",
                "sample_size": sample_size,
                "n_features_original": len(X_val.columns),
                "n_features_transformed": len(feat_names_unique),
                "feature_names_original": X_val.columns.tolist(),
                "feature_names_transformed": feat_names_unique,
                "top_features_by_importance": []
            }
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                top_indices = np.argsort(importances)[::-1][:20]
                for i, idx in enumerate(top_indices):
                    stats["top_features_by_importance"].append({
                        "rank": i + 1,
                        "feature_name": feat_names_unique[idx],
                        "importance": float(importances[idx])
                    })
            with open(os.path.join(output_dir, "interpretability_stats.json"), 'w') as f:
                json.dump(stats, f, indent=2)
            print("Estadísticas guardadas")
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
        
        plt.close('all')
        gc.collect()
        
        files_generated = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.json'))]
        print("Análisis de interpretabilidad completado")
        print(f"Archivos generados en {output_dir}:")
        for file in files_generated:
            print(f"- {file}")
        
        return True
    except Exception as e:
        print(f"Error en análisis de interpretabilidad: {e}")
        try:
            create_basic_analysis(model_path, val_path, output_dir)
            return True
        except Exception as e2:
            print(f"Error en análisis básico: {e2}")
            return False

def create_basic_analysis(model_path, val_path, output_dir):
    print("Creando análisis básico de fallback...")
    model = joblib.load(model_path)
    
    if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('clf'), 'feature_importances_'):
        clf = model.named_steps['clf']
        importances = clf.feature_importances_
        
        plt.figure(figsize=(10, 6))
        n_features = min(15, len(importances))
        indices = np.argsort(importances)[-n_features:]
        plt.barh(range(n_features), importances[indices])
        plt.yticks(range(n_features), [f"feature_{i}" for i in indices])
        plt.xlabel('Importancia')
        plt.title('Feature Importance (Análisis Básico)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "basic_feature_importance.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Análisis básico completado")
    else:
        print("No se puede extraer importancia del modelo")
