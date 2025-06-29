FROM apache/airflow:2.8.3-python3.10

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Copia tu DAG y scripts
COPY dag_mlops.py /opt/airflow/dags/dag_mlops.py
COPY scripts/ /opt/airflow/dags/scripts/
