# airflow/scripts/split_data.py

import pandas as pd

def split_temporal(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Realiza la partición temporal en train, val y test usando la columna 'week' del dataset.
    """
    semanas = sorted(dataset['week'].unique())
    n_total = len(semanas)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)

    semanas_train = semanas[:n_train]
    semanas_val   = semanas[n_train : n_train + n_val]
    semanas_test  = semanas[n_train + n_val : ]

    train_df = dataset[dataset['week'].isin(semanas_train)].copy()
    val_df   = dataset[dataset['week'].isin(semanas_val)].copy()
    test_df  = dataset[dataset['week'].isin(semanas_test)].copy()

    return train_df, val_df, test_df
