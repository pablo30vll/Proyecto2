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
import joblib
import os
import json
from datetime import datetime

def train_model(train_path, val_path, mlflow_uri, model_path, tune_hparams=True, n_trials=20):
    print(" Iniciando entrenamiento")
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    drop_cols = ['comprado', 'customer_id', 'product_id']
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['comprado']
    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df['comprado']
    
    print(f" Datos de entrenamiento: {X_train.shape}")
    print(f" Datos de validación: {X_val.shape}")
    print(f" Columnas disponibles: {list(X_train.columns)}")
    
    available_cols = set(X_train.columns)
    expected_numeric = ['Y','X','num_deliver_per_week','num_visit_per_week','size']
    expected_categorical = ['customer_type','brand','category','sub_category','segment','package','region_id','zone_id']
    expected_discretize = ['num_deliver_per_week']
    
    numeric_feats = [col for col in expected_numeric if col in available_cols]
    categorical_feats = [col for col in expected_categorical if col in available_cols]
    discretize_feats = [col for col in expected_discretize if col in available_cols]
    
    print(f"Columnas numéricas encontradas: {numeric_feats}")
    print(f"Columnas categóricas encontradas: {categorical_feats}")
    print(f"Columnas para discretizar: {discretize_feats}")
    
    if len(numeric_feats) < 2:
        numeric_feats = X_train.select_dtypes(include=['number']).columns.tolist()
        numeric_feats = [col for col in numeric_feats if col != 'week']
        print(f"Usando todas las columnas numéricas: {numeric_feats}")
    
    if len(categorical_feats) < 2:
        categorical_feats = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_feats = [col for col in categorical_feats if col != 'week']
        print(f"Usando todas las columnas categóricas: {categorical_feats}")
    
    has_week_col = 'week' in available_cols
    print(f"¿Existe columna 'week'? {has_week_col}")
    
    def extract_week_of_year(X):
        arr = X.values.ravel() if hasattr(X, "values") else X.ravel()
        weeks = pd.to_datetime(arr).isocalendar().week.to_numpy().reshape(-1,1).astype(float)
        return weeks
    
    trial_history = []

    def objective(trial):
        n_bins = trial.suggest_int("n_bins", 3, 10)
        winsor_lower = trial.suggest_int("winsor_lower", 0, 5)
        winsor_upper = trial.suggest_int("winsor_upper", 95, 100)
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 5, 30)
        max_features = trial.suggest_categorical("max_features", ["sqrt","log2",None])
        
        transformers = []
        
        if numeric_feats:
            def safe_winsorize(Z):
                df = pd.DataFrame(Z)
                lower_q = df.quantile(winsor_lower/100, axis=0)
                upper_q = df.quantile(winsor_upper/100, axis=0)
                upper_q = upper_q.where(upper_q > lower_q, lower_q + 1e-6)
                return df.clip(lower=lower_q, upper=upper_q, axis=1).values
            
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('winsor', FunctionTransformer(safe_winsorize, validate=False)),
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
        
        if has_week_col:
            week_pipe = Pipeline([
                ('extract', FunctionTransformer(extract_week_of_year, validate=False)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('week', week_pipe, ['week']))
        
        if not transformers:
            all_numeric = X_train.select_dtypes(include=['number']).columns.tolist()
            if all_numeric:
                simple_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('all_numeric', simple_pipe, all_numeric))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
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
        
        print(f"  Trial {trial.number}: AP={ap:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        return ap

    if tune_hparams:
        print("Iniciando optimización de hiperparámetros con Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Mejores parámetros encontrados: {best_params}")
    else:
        best_params = {
            "n_bins": 9,
            "winsor_lower": 2,
            "winsor_upper": 100,
            "n_estimators": 178,
            "max_depth": 24,
            "max_features": None
        }
        print(f"Usando parámetros fijos: {best_params}")
    
    print("Entrenando modelo final...")
    
    transformers = []
    
    if numeric_feats:
        def safe_winsorize_final(Z):
            df = pd.DataFrame(Z)
            lower_q = df.quantile(best_params["winsor_lower"]/100, axis=0)
            upper_q = df.quantile(best_params["winsor_upper"]/100, axis=0)
            upper_q = upper_q.where(upper_q > lower_q, lower_q + 1e-6)
            return df.clip(lower=lower_q, upper=upper_q, axis=1).values
        
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('winsor', FunctionTransformer(safe_winsorize_final, validate=False)),
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
    
    if has_week_col:
        week_pipe = Pipeline([
            ('extract', FunctionTransformer(extract_week_of_year, validate=False)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('week', week_pipe, ['week']))
    
    if not transformers:
        all_numeric = X_train.select_dtypes(include=['number']).columns.tolist()
        if all_numeric:
            simple_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('all_numeric', simple_pipe, all_numeric))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    clf = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        max_features=best_params["max_features"],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
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
    joblib.dump(pipe, model_path)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": "sodai_purchase_prediction",
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
            "has_week_feature": has_week_col
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
    
    return ap
