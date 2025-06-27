import sys
import os
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import pandas as pd
from airflow.utils.trigger_rule import TriggerRule

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from scripts.extract_data import load_raw_data
from scripts.preprocessing import preprocess_data
from scripts.split_data import split_temporal
from scripts.drift import detect_drift, save_drift_report
from scripts.training_fast import train_model    
from scripts.shap_analysis import run_shap_analysis
from scripts.predict import predict_next_week

BASE_DIR = "/tmp/airflow_tmp"
DATA_DIR = f"{BASE_DIR}/data"
OUTPUTS_DIR = f"{BASE_DIR}/outputs"
TMP_DIR = f"{BASE_DIR}/tmp"
MLRUNS_DIR = f"{BASE_DIR}/mlruns"
MODELS_DIR = f"{OUTPUTS_DIR}/models"
HISTORIC_DIR = DATA_DIR   


default_args = {
    'owner': 'sodai_team',
    'start_date': datetime(2024, 6, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

def extract_task(**context):
    """Extrae y deja los datos nuevos listos en DATA_DIR"""
    
    pass 

def preprocess_task():
    """
    Extrae, limpia y transforma los datos para dejarlos listos para el modelado.
    Guarda el resultado como parquet para los siguientes pasos del pipeline.
    """
    transacciones = pd.read_parquet(f"{DATA_DIR}/transacciones.parquet")
    clientes      = pd.read_parquet(f"{DATA_DIR}/clientes.parquet")
    productos     = pd.read_parquet(f"{DATA_DIR}/productos.parquet")

    df_final = preprocess_data(transacciones, clientes, productos)
    df_final.to_parquet(f"{DATA_DIR}/dataset_limpio.parquet")
    print(f" Datos procesados guardados: {df_final.shape}")


def split_task(**context):
    """Realiza el split temporal"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    dataset = pd.read_parquet(f"{DATA_DIR}/dataset_limpio.parquet")
    train_df, val_df, test_df = split_temporal(dataset)
    train_df.to_parquet(f"{DATA_DIR}/train.parquet")
    val_df.to_parquet(f"{DATA_DIR}/val.parquet")
    test_df.to_parquet(f"{DATA_DIR}/test.parquet")


def drift_check_task(**context):
    """Chequea drift contra el dataset anterior"""
    old_path = f"{DATA_DIR}/dataset_hist.parquet"
    new_path = f"{DATA_DIR}/dataset_limpio.parquet"
    if not os.path.exists(old_path):
        return 'train_model'
    drift_result = detect_drift(old_path, new_path, threshold=0.05)
    save_drift_report(drift_result, f"{OUTPUTS_DIR}/drift_report.json")
    if drift_result["drift_detected"]:
        return 'train_model'
    else:
        return 'skip_training'

def train_task(**context):
    """Entrenamiento con contexto temporal"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    ap = train_model(
    train_path=f"{DATA_DIR}/train.parquet",
    val_path=f"{DATA_DIR}/val.parquet",
    mlflow_uri=f"file://{MLRUNS_DIR}",
    model_path=f"{MODELS_DIR}/model.pkl",
    tune_hparams=True,
    n_trials=30
    )
    import shutil
    shutil.copy2(f"{DATA_DIR}/dataset_limpio.parquet", f"{DATA_DIR}/dataset_hist.parquet")


    print(f" Entrenamiento completado - AP: {ap:.4f}")

def skip_training_task(**context):
    """No hace nada, solo avanza"""
    pass

def shap_task(**context):
    run_shap_analysis(
        model_path=f"{MODELS_DIR}/model.pkl",
        val_path=f"{DATA_DIR}/val.parquet",
        output_dir=f"{OUTPUTS_DIR}/shap/"
    )

def predict_task(**context):
    predict_next_week(
        model_path=f"{MODELS_DIR}/model.pkl",
        dataset_path=f"{DATA_DIR}/dataset_limpio.parquet",
        output_path=f"{OUTPUTS_DIR}/predicciones.csv"
    )

with DAG(
    'sodai_mlops',
    default_args=default_args,
    schedule_interval='0 9 * * 1',  
    catchup=False,
    description='Pipeline MLOps semanal con drift y retrain automático',
    tags=['mlops', 'drift']
) as dag:

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_task,
        doc_md="Carga los archivos parquet nuevos y los deja listos para el pipeline"
    )
    preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_task,
        doc_md="Limpieza y transformación de los datos"
    )
    split = PythonOperator(
        task_id='split_data',
        python_callable=split_task,
        doc_md="Split temporal en train/val/test"
    )
    drift_check = BranchPythonOperator(
        task_id='drift_check',
        python_callable=drift_check_task,
        doc_md="Detección de drift entre datos históricos y nuevos"
    )
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_task,
        doc_md="Entrenamiento (solo si hay drift)"
    )
    skip_training = PythonOperator(
        task_id='skip_training',
        python_callable=skip_training_task,
        doc_md="Avanza si NO hay drift"
    )
    shap_analysis = PythonOperator(
        task_id='shap_analysis',
        python_callable=shap_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,  # <--- AQUÍ EL CAMBIO
        doc_md="Interpretabilidad SHAP del modelo"
    )
    predict = PythonOperator(
        task_id='predict',
        python_callable=predict_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,  # <--- AQUÍ EL CAMBIO
        doc_md="Predicción para la próxima semana"
    )

    extract >> preprocess >> split >> drift_check
    drift_check >> [train, skip_training]
    [train, skip_training] >> shap_analysis >> predict
