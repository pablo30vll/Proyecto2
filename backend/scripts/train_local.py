#!/usr/bin/env python3
"""
Entrenador para PRODUCCIÓN con mejores hiperparámetros
Compatible 100% con Docker - Solo sklearn estándar
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
import joblib
import os
import json
from datetime import datetime

def train_production_model(train_path, val_path, model_path, best_params):
    """
    Entrena modelo de PRODUCCIÓN con mejores hiperparámetros
    Compatible 100% con Docker
    """
    print("🚀 ENTRENANDO MODELO DE PRODUCCIÓN")
    print("=" * 50)
    print(f"🎯 Hiperparámetros optimizados: {best_params}")
    
    # Cargar datos
    print("📊 Cargando datos...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    print(f"✅ Datos cargados:")
    print(f"   Train: {train_df.shape}")
    print(f"   Val: {val_df.shape}")
    
    # Preparar features y target
    drop_cols = ['comprado', 'customer_id', 'product_id']
    X_train = train_df.drop(columns=[col for col in drop_cols if col in train_df])
    y_train = train_df['comprado']
    X_val = val_df.drop(columns=[col for col in drop_cols if col in val_df])
    y_val = val_df['comprado']
    
    print(f"📈 Distribución del target:")
    print(f"   Train: {np.bincount(y_train)} ({y_train.mean():.3f})")
    print(f"   Val: {np.bincount(y_val)} ({y_val.mean():.3f})")
    
    # PROCESAR WEEK FEATURE ANTES DEL PIPELINE (sin FunctionTransformer)
    if 'week' in X_train.columns:
        print("📅 Procesando feature 'week' sin FunctionTransformer...")
        X_train['week_of_year'] = pd.to_datetime(X_train['week']).dt.isocalendar().week
        X_val['week_of_year'] = pd.to_datetime(X_val['week']).dt.isocalendar().week
        X_train = X_train.drop('week', axis=1)
        X_val = X_val.drop('week', axis=1)
        print("✅ Week procesado correctamente")
    
    # Detectar tipos de features automáticamente
    numeric_feats = []
    categorical_feats = []
    
    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64']:
            numeric_feats.append(col)
        else:
            categorical_feats.append(col)
    
    print(f"🔍 Features detectadas:")
    print(f"   Numéricas ({len(numeric_feats)}): {numeric_feats}")
    print(f"   Categóricas ({len(categorical_feats)}): {categorical_feats}")
    
    # Pipeline de PRODUCCIÓN - Solo sklearn estándar
    transformers = []
    
    # Pipeline numérico: RobustScaler (equivalente a winsorización + escalado)
    if numeric_feats:
        print("🔢 Configurando pipeline numérico con RobustScaler...")
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # Maneja outliers automáticamente
        ])
        transformers.append(('num', num_pipeline, numeric_feats))
    
    # Pipeline categórico: Estándar
    if categorical_feats:
        print("📝 Configurando pipeline categórico...")
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, categorical_feats))
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Clasificador con TUS mejores hiperparámetros
    print("🤖 Configurando RandomForest con hiperparámetros optimizados...")
    classifier = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],      # 94
        max_depth=best_params['max_depth'],            # 17
        max_features=best_params['max_features'],      # None
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1  # Para ver progreso
    )
    
    # Pipeline completo
    model = Pipeline([
        ('preproc', preprocessor),
        ('clf', classifier)
    ])
    
    print("🔄 Entrenando modelo de producción...")
    print("⏳ Esto puede tomar varios minutos...")
    
    # Entrenar
    model.fit(X_train, y_train)
    
    print("✅ Entrenamiento completado!")
    
    # Evaluar
    print("📊 Evaluando modelo...")
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_val = model.predict_proba(X_val)[:, 1]
    
    # Usar threshold optimizado (0.2 según análisis anterior)
    threshold = 0.2
    y_pred_train = (y_prob_train > threshold).astype(int)
    y_pred_val = (y_prob_val > threshold).astype(int)
    
    # Métricas en train
    ap_train = average_precision_score(y_train, y_prob_train)
    precision_train = precision_score(y_train, y_pred_train, zero_division=0)
    recall_train = recall_score(y_train, y_pred_train, zero_division=0)
    f1_train = f1_score(y_train, y_pred_train, zero_division=0)
    
    # Métricas en validación
    ap_val = average_precision_score(y_val, y_prob_val)
    precision_val = precision_score(y_val, y_pred_val, zero_division=0)
    recall_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    
    print(f"🎯 RESULTADOS FINALES:")
    print(f"=" * 40)
    print(f"📈 TRAIN:")
    print(f"   Average Precision: {ap_train:.4f}")
    print(f"   Precision: {precision_train:.4f}")
    print(f"   Recall: {recall_train:.4f}")
    print(f"   F1 Score: {f1_train:.4f}")
    
    print(f"📊 VALIDATION:")
    print(f"   Average Precision: {ap_val:.4f}")
    print(f"   Precision: {precision_val:.4f}")
    print(f"   Recall: {recall_val:.4f}")
    print(f"   F1 Score: {f1_val:.4f}")
    
    print(f"🎲 PROBABILIDADES:")
    print(f"   Val Min: {y_prob_val.min():.4f}")
    print(f"   Val Max: {y_prob_val.max():.4f}")
    print(f"   Val Mean: {y_prob_val.mean():.4f}")
    print(f"   Val Median: {np.median(y_prob_val):.4f}")
    
    print(f"🎯 PREDICCIONES CON THRESHOLD {threshold}:")
    print(f"   Train positivas: {y_pred_train.sum():,}/{len(y_pred_train):,} ({y_pred_train.mean():.1%})")
    print(f"   Val positivas: {y_pred_val.sum():,}/{len(y_pred_val):,} ({y_pred_val.mean():.1%})")
    
    # Guardar modelo
    print(f"💾 Guardando modelo en: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Verificar que se puede cargar
    print("🔍 Verificando modelo guardado...")
    try:
        test_model = joblib.load(model_path)
        test_pred = test_model.predict_proba(X_val[:5])
        print(f"✅ Modelo verificado - Predicciones test: {test_pred[:, 1]}")
    except Exception as e:
        print(f"❌ Error verificando modelo: {e}")
        return None
    
    # Guardar métricas y metadatos
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": "sodai_production_model",
        "hyperparameters": best_params,
        "train_metrics": {
            "average_precision": float(ap_train),
            "precision": float(precision_train),
            "recall": float(recall_train),
            "f1_score": float(f1_train)
        },
        "validation_metrics": {
            "average_precision": float(ap_val),
            "precision": float(precision_val),
            "recall": float(recall_val),
            "f1_score": float(f1_val)
        },
        "model_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "numeric_features": numeric_feats,
            "categorical_features": categorical_feats,
            "threshold_used": threshold,
            "positive_rate_train": float(y_pred_train.mean()),
            "positive_rate_val": float(y_pred_val.mean()),
            "probability_stats": {
                "min": float(y_prob_val.min()),
                "max": float(y_prob_val.max()),
                "mean": float(y_prob_val.mean()),
                "median": float(np.median(y_prob_val))
            },
            "production_ready": True,
            "docker_compatible": True
        }
    }
    
    results_path = os.path.join(os.path.dirname(model_path), "production_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📋 Resultados guardados en: {results_path}")
    print("🎉 ¡MODELO DE PRODUCCIÓN LISTO!")
    
    return model_path, ap_val

if __name__ == "__main__":
    # Hiperparámetros optimizados
    BEST_PARAMS = {
        'n_bins': 3,
        'winsor_lower': 3,
        'winsor_upper': 98,
        'n_estimators': 94,
        'max_depth': 17,
        'max_features': None
    }
    
    # Rutas - AJUSTA SEGÚN TU ESTRUCTURA
    train_path = "/home/pvergara/Escritorio/laboratorio/dags/scripts/tmp/train.parquet"  # Cambia por tu ruta
    val_path = "/home/pvergara/Escritorio/laboratorio/dags/scripts/tmp/val.parquet"      # Cambia por tu ruta
    model_path = "outputs/models/sodai_production_model.pkl"
    
    print("🏭 ENTRENADOR DE PRODUCCIÓN SODAI")
    print("=" * 50)
    print("✅ Hiperparámetros optimizados")
    print("✅ Compatible con Docker")
    print("✅ Solo sklearn estándar")
    print("✅ RobustScaler (equivale a winsorización)")
    print("✅ Threshold optimizado (0.2)")
    
    # Verificar archivos
    if not os.path.exists(train_path):
        print(f"❌ No se encuentra: {train_path}")
        print("💡 Ajusta las rutas en el script")
        exit(1)
    
    if not os.path.exists(val_path):
        print(f"❌ No se encuentra: {val_path}")
        print("💡 Ajusta las rutas en el script")
        exit(1)
    
    # Entrenar
    result = train_production_model(train_path, val_path, model_path, BEST_PARAMS)
    
    if result:
        model_path, ap_val = result
        print(f"\n🎯 MODELO DE PRODUCCIÓN COMPLETADO:")
        print(f"   📁 Archivo: {model_path}")
        print(f"   📊 Average Precision: {ap_val:.4f}")
        print(f"   ✅ Compatible con Docker")
        print(f"   ✅ Listo para despliegue")
        
        print(f"\n📝 PRÓXIMOS PASOS:")
        print(f"   1. cp {model_path} /ruta/app/models/model.pkl")
        print(f"   2. docker-compose restart backend")
        print(f"   3. python3 verify_model.py")
        print(f"   4. ¡Probar con datos reales!")
        
    else:
        print("\n❌ Error en el entrenamiento")