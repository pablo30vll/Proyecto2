import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, KBinsDiscretizer, FunctionTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import json
import numpy as np
from datetime import datetime

# Transformer personalizado para winsorización que es serializable
class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Transformer personalizado para winsorización que es completamente serializable"""
    
    def __init__(self, lower_percentile=3, upper_percentile=98):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Aplica winsorización de forma segura"""
        df = pd.DataFrame(X)
        lower_q = df.quantile(self.lower_percentile/100, axis=0)
        upper_q = df.quantile(self.upper_percentile/100, axis=0)
        upper_q = upper_q.where(upper_q > lower_q, lower_q + 1e-6)
        return df.clip(lower=lower_q, upper=upper_q, axis=1).values

# Función global para extraer semana del año (serializable)
def extract_week_of_year(X):
    """Extrae la semana del año de fechas"""
    arr = X.values.ravel() if hasattr(X, "values") else X.ravel()
    weeks = pd.to_datetime(arr).isocalendar().week.to_numpy().reshape(-1,1).astype(float)
    return weeks

def train_model_with_best_params(train_path, val_path, mlflow_uri, model_path, 
                                best_params=None, fast_mode=True):
    """
    Entrena el modelo usando los mejores hiperparámetros encontrados previamente
    """
    print("🚀 Iniciando entrenamiento con MEJORES HIPERPARÁMETROS...")
    
    # Usar los mejores parámetros que encontraste
    if best_params is None:
        best_params = {
            'n_bins': 3,
            'winsor_lower': 3,
            'winsor_upper': 98,
            'n_estimators': 94,
            'max_depth': 17,
            'max_features': None
        }
    
    print(f"📊 Usando hiperparámetros optimizados: {best_params}")

    # Cargar datos
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    print(f"Datos originales - Train: {train_df.shape}, Val: {val_df.shape}")
    
    # Reducir datos si es modo rápido
    if fast_mode:
        if len(train_df) > 50000:
            train_df = train_df.sample(n=50000, random_state=42)
            print(f"Datos reducidos para demo - Train: {train_df.shape}")
        if len(val_df) > 20000:
            val_df = val_df.sample(n=20000, random_state=42)
            print(f"Datos reducidos para demo - Val: {val_df.shape}")
    
    # Preparar features
    drop_cols = ['comprado', 'customer_id', 'product_id']
    X_train = train_df.drop(columns=[col for col in drop_cols if col in train_df])
    y_train = train_df['comprado']
    X_val = val_df.drop(columns=[col for col in drop_cols if col in val_df])
    y_val = val_df['comprado']
    
    # Definir features
    numeric_feats = ['region_id', 'Y', 'X', 'num_deliver_per_week', 'num_visit_per_week', 'size']
    categorical_feats = ['customer_type', 'brand', 'category', 'sub_category', 'segment', 'package']
    discretize_feats = ['num_deliver_per_week']
    week_feat = ['week'] if 'week' in X_train.columns else []
    
    # Filtrar features que existen
    numeric_feats = [c for c in numeric_feats if c in X_train]
    categorical_feats = [c for c in categorical_feats if c in X_train]
    discretize_feats = [c for c in discretize_feats if c in X_train]
    
    print(f"✅ Features identificados:")
    print(f"   Numéricas: {numeric_feats}")
    print(f"   Categóricas: {categorical_feats}")
    print(f"   Para discretizar: {discretize_feats}")
    print(f"   Week feature: {len(week_feat) > 0}")

    # Construir transformers con mejores hiperparámetros
    transformers = []
    
    if numeric_feats:
        # Usar transformer personalizado serializable
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('winsor', WinsorizerTransformer(
                lower_percentile=best_params["winsor_lower"],
                upper_percentile=best_params["winsor_upper"]
            )),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, numeric_feats))

    if discretize_feats:
        disc_pipe = Pipeline([
            ('kbins', KBinsDiscretizer(
                n_bins=best_params["n_bins"], 
                encode='ordinal', 
                strategy='quantile'
            ))
        ])
        transformers.append(('disc', disc_pipe, discretize_feats))

    if categorical_feats:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, categorical_feats))

    if week_feat:
        week_pipe = Pipeline([
            ('extract', FunctionTransformer(extract_week_of_year, validate=False)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('week', week_pipe, week_feat))

    # Crear pipeline completo
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    # Crear clasificador con mejores hiperparámetros
    clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # Usar todos los cores disponibles
    )
    
    # Pipeline final
    pipe = Pipeline([
        ('preproc', preprocessor),
        ('clf', clf)
    ])
    
    print("🤖 Entrenando modelo final con mejores hiperparámetros...")
    pipe.fit(X_train, y_train)
    
    # Evaluación
    print("📊 Evaluando modelo...")
    y_prob = pipe.predict_proba(X_val)[:,1]
    y_pred = (y_prob > 0.5).astype(int)
    
    # Métricas
    ap = average_precision_score(y_val, y_prob)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"🎯 Métricas finales:")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    # Guardar modelo
    print(f"💾 Guardando modelo en: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        # Intentar con joblib primero
        joblib.dump(pipe, model_path.replace('.pkl', '.joblib'))
        print("✅ Modelo guardado exitosamente con joblib")
        
        # También guardar con pickle como backup
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(pipe, f)
        print("✅ Modelo también guardado con pickle")
        
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")
        # Guardar componentes por separado como fallback
        try:
            joblib.dump(pipe.named_steps['clf'], model_path.replace('.pkl', '_classifier.joblib'))
            joblib.dump(pipe.named_steps['preproc'], model_path.replace('.pkl', '_preprocessor.joblib'))
            print("✅ Componentes guardados por separado")
        except Exception as e2:
            print(f"❌ Error guardando componentes: {e2}")

    # Guardar resultados
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": "sodai_purchase_prediction_optimized",
        "final_metrics": {
            "average_precision": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "best_params": best_params,
        "model_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "numeric_features": numeric_feats,
            "categorical_features": categorical_feats,
            "has_week_feature": len(week_feat) > 0,
            "fast_mode": fast_mode,
            "optimized": True
        }
    }
    
    results_dir = os.path.dirname(model_path)
    results_path = os.path.join(results_dir, "training_results_optimized.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📋 Resultados guardados en: {results_path}")
    
    print("🎉 Entrenamiento OPTIMIZADO completado!")
    return ap

# Función original modificada para ser serializable
def train_model(train_path, val_path, mlflow_uri, model_path, tune_hparams=True, n_trials=5):
    """Función original pero con correcciones de serialización"""
    
    if not tune_hparams:
        # Si no se quiere optimizar, usar directamente los mejores parámetros conocidos
        best_params = {
            'n_bins': 3,
            'winsor_lower': 3,
            'winsor_upper': 98,
            'n_estimators': 94,
            'max_depth': 17,
            'max_features': None
        }
        return train_model_with_best_params(train_path, val_path, mlflow_uri, model_path, best_params)
    
    print("Iniciando entrenamiento RÁPIDO para demostración...")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    print(f"Datos originales - Train: {train_df.shape}, Val: {val_df.shape}")
    
    if len(train_df) > 50000:
        train_df = train_df.sample(n=50000, random_state=42)
        print(f"Datos reducidos para demo - Train: {train_df.shape}")
    if len(val_df) > 20000:
        val_df = val_df.sample(n=20000, random_state=42)
        print(f"Datos reducidos para demo - Val: {val_df.shape}")
    
    drop_cols = ['comprado', 'customer_id', 'product_id']
    X_train = train_df.drop(columns=[col for col in drop_cols if col in train_df])
    y_train = train_df['comprado']
    X_val = val_df.drop(columns=[col for col in drop_cols if col in val_df])
    y_val = val_df['comprado']
    
    # Features
    numeric_feats = ['region_id', 'Y', 'X', 'num_deliver_per_week', 'num_visit_per_week', 'size']
    categorical_feats = ['customer_type', 'brand', 'category', 'sub_category', 'segment', 'package']
    discretize_feats = ['num_deliver_per_week']
    week_feat = ['week'] if 'week' in X_train.columns else []
    
    # Filtrar
    numeric_feats = [c for c in numeric_feats if c in X_train]
    categorical_feats = [c for c in categorical_feats if c in X_train]
    discretize_feats = [c for c in discretize_feats if c in X_train]
    
    print(f"Numéricas: {numeric_feats}")
    print(f"Categóricas: {categorical_feats}")
    print(f"Para discretizar: {discretize_feats}")
    print(f"¿Week?: {len(week_feat) > 0}")

    trial_history = []

    def objective(trial):
        n_bins = trial.suggest_categorical("n_bins", [5, 7, 9])
        winsor_lower = trial.suggest_categorical("winsor_lower", [1, 3, 5])
        winsor_upper = trial.suggest_categorical("winsor_upper", [95, 97, 99])
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
        max_depth = trial.suggest_categorical("max_depth", [10, 15, 20])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

        transformers = []
        
        if numeric_feats:
            # Usar transformer serializable
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('winsor', WinsorizerTransformer(winsor_lower, winsor_upper)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, numeric_feats))

        if discretize_feats:
            disc_pipe = Pipeline([
                ('kbins', KBinsDiscretizer(
                    n_bins=n_bins, encode='ordinal', strategy='quantile'
                ))
            ])
            transformers.append(('disc', disc_pipe, discretize_feats))

        if categorical_feats:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipe, categorical_feats))

        if week_feat:
            week_pipe = Pipeline([
                ('extract', FunctionTransformer(extract_week_of_year, validate=False)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('week', week_pipe, week_feat))

        preprocessor = ColumnTransformer(transformers, remainder='drop')
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            class_weight='balanced',
            random_state=42,
            n_jobs=2
        )
        pipe = Pipeline([
            ('preproc', preprocessor),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_val)[:,1]
        y_pred = (y_prob > 0.5).astype(int)
        ap = average_precision_score(y_val, y_prob)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        trial_info = {
            "trial_number": trial.number,
            "params": trial.params,
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        trial_history.append(trial_info)
        print(f"Trial {trial.number}: AP={ap:.4f}")
        return ap

    if tune_hparams:
        print(f"Optimizando con {n_trials} trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"Mejores parámetros: {best_params}")
    else:
        best_params = {
            'n_bins': 3,
            'winsor_lower': 3,
            'winsor_upper': 98,
            'n_estimators': 94,
            'max_depth': 17,
            'max_features': None
        }
        print(f"Parámetros fijos: {best_params}")

    # Entrenar modelo final con transformer serializable
    print("Entrenando modelo final...")
    transformers = []
    
    if numeric_feats:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('winsor', WinsorizerTransformer(
                best_params["winsor_lower"], 
                best_params["winsor_upper"]
            )),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, numeric_feats))

    if discretize_feats:
        disc_pipe = Pipeline([
            ('kbins', KBinsDiscretizer(
                n_bins=best_params["n_bins"], encode='ordinal', strategy='quantile'
            ))
        ])
        transformers.append(('disc', disc_pipe, discretize_feats))

    if categorical_feats:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, categorical_feats))

    if week_feat:
        week_pipe = Pipeline([
            ('extract', FunctionTransformer(extract_week_of_year, validate=False)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('week', week_pipe, week_feat))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        class_weight='balanced',
        random_state=42,
        n_jobs=2
    )
    pipe = Pipeline([
        ('preproc', preprocessor),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)[:,1]
    y_pred = (y_prob > 0.5).astype(int)
    ap = average_precision_score(y_val, y_prob)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"Métricas finales:")
    print(f"   Average Precision: {ap:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    print(f"Guardando modelo en: {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Guardar con joblib (más seguro para sklearn)
    try:
        joblib.dump(pipe, model_path.replace('.pkl', '.joblib'))
        print("✅ Modelo guardado con joblib")
    except Exception as e:
        print(f"❌ Error con joblib: {e}")
        
    # También intentar con pickle
    try:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(pipe, f)
        print("✅ Modelo guardado con pickle")
    except Exception as e:
        print(f"❌ Error con pickle: {e}")

    # Guardar resultados
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": "sodai_purchase_prediction_fast",
        "final_metrics": {
            "average_precision": float(ap),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "best_params": best_params,
        "model_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "numeric_features": numeric_feats,
            "categorical_features": categorical_feats,
            "has_week_feature": len(week_feat) > 0,
            "fast_mode": True
        },
        "hyperparameter_search": {
            "tune_hparams": tune_hparams,
            "n_trials": n_trials if tune_hparams else 0,
            "trials_history": trial_history
        }
    }
    results_dir = os.path.dirname(model_path)
    results_path = os.path.join(results_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en: {results_path}")
    print("Entrenamiento RÁPIDO completado!")

    return ap

if __name__ == "__main__":
    # Para entrenar directamente con los mejores hiperparámetros sin optimización
    best_params = {
        'n_bins': 3,
        'winsor_lower': 3,
        'winsor_upper': 98,
        'n_estimators': 94,
        'max_depth': 17,
        'max_features': None
    }
    
    ap_score = train_model_with_best_params(
        train_path="tmp/train.parquet",
        val_path="tmp/val.parquet",
        mlflow_uri="file://outputs/mlruns",
        model_path="outputs/models/sodai_model_optimized.pkl",
        best_params=best_params,
        fast_mode=False  # Usar todos los datos
    )