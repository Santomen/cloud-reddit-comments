import os
from kfp.dsl import Dataset, Output, component
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.11-slim"),
    packages_to_install=[
        "pandas",
        "google-cloud-bigquery",
        "scikit-learn",
        "db-dtypes",
        "numpy",
        # Restricciones de seguridad para evitar conflictos de KFP
        "protobuf<5.0.0",
        "urllib3<2.0.0"
    ],
)
def load_data(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample

    # 1. Carga Eficiente desde BigQuery
    client = bigquery.Client(project=project_id)
    
    # Seleccionamos solo las columnas que nos importan
    query = f"""
        SELECT corpus, quartile_label 
        FROM `{project_id}.{bq_dataset}.{bq_table}`
    """
    
    print("Iniciando descarga de datos de BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"Datos descargados. Total filas originales: {len(df)}")

    # 2. Limpieza básica
    df = df.dropna(subset=['corpus', 'quartile_label'])
    
    # 3. Mapeo de Etiquetas (S, M, L -> 0, 1, 2)
    # Asumimos: S=Small (0), M=Medium (1), L=Large (2)
    label_map = {'S': 0, 'M': 1, 'L': 2}
    
    # Validar que solo existan esas etiquetas o filtrar basura
    df = df[df['quartile_label'].isin(label_map.keys())]
    df['label'] = df['quartile_label'].map(label_map)

    # 4. Oversampling Automático
    # Estrategia: Encontrar la clase con más ejemplos y subir las otras a ese nivel.
    
    # Contar ocurrencias por clase
    conteo_clases = df['label'].value_counts()
    clase_mayoritaria = conteo_clases.idxmax()
    n_maximo = conteo_clases.max()
    
    print(f"Distribución antes del balanceo: \n{conteo_clases}")
    
    df_balanced_list = []
    
    # Iterar sobre cada clase (0, 1, 2)
    for clase in df['label'].unique():
        df_clase = df[df['label'] == clase]
        
        if clase == clase_mayoritaria:
            df_balanced_list.append(df_clase)
        else:
            # Si es minoritaria, hacemos Upsampling (con reemplazo)
            df_clase_upsampled = resample(
                df_clase,
                replace=True,     # Sample with replacement
                n_samples=n_maximo, # Match majority class
                random_state=42
            )
            df_balanced_list.append(df_clase_upsampled)
            
    df_balanced = pd.concat(df_balanced_list)
    
    # Mezclar filas (Shuffle) para que no queden ordenadas por clase
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total filas después del oversampling: {len(df_balanced)}")
    print(f"Distribución balanceada: \n{df_balanced['label'].value_counts()}")

    # 5. Dividir en Train y Test
    # Usamos stratify para asegurar que la distribución se mantenga en el split
    X_train, X_test = train_test_split(
        df_balanced,
        test_size=0.2,
        random_state=42,
        stratify=df_balanced['label']
    )

    # 6. Guardar CSVs para el siguiente paso
    # Guardamos 'corpus' (texto) y 'label' (número ya mapeado)
    output_columns = ['corpus', 'label']
    
    X_train[output_columns].to_csv(train_dataset.path, index=False)
    X_test[output_columns].to_csv(test_dataset.path, index=False)
    
    print("Datos procesados y guardados exitosamente.")
