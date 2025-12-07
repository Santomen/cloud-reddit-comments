import os
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
# Si usas dotenv aquí, descomenta la siguiente línea, si no, puedes quitarla
# from dotenv import load_dotenv

@component(
    base_image="python:3.10-slim",  # O usa os.getenv si tienes load_dotenv funcionando
    packages_to_install=[
        "pandas",
        "numpy",
        "tensorflow",
        "scikit-learn",
        "protobuf<5.0.0", 
        "urllib3<2.0.0"
    ],
)
def evaluate_lstm_model(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],      
    tokenizer_artifact: Input[Model],  
    metrics: Output[Metrics],
):
    # --- ¡ESTA ES LA CLAVE! ---
    import os  # <--- ESTE IMPORT ES EL QUE ARREGLA TU ERROR ACTUAL
    # --------------------------
    import pandas as pd
    import numpy as np
    import pickle
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.metrics import accuracy_score, confusion_matrix

    # --- 1. CONFIGURACIÓN NLP ---
    MAX_LENGTH = 100
    TRUNC_TYPE = 'post'
    PADDING_TYPE = 'post'

    # --- 2. CARGAR ARTEFACTOS ---
    print("Cargando modelo y tokenizer...")
    
    # Cargar Tokenizer
    with open(tokenizer_artifact.path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    # Cargar Modelo con verificación segura
    model_path = model_artifact.path
    if not os.path.exists(model_path) and os.path.exists(model_path + ".keras"):
        model_path += ".keras"
        
    model = load_model(model_path)

    # --- 3. CARGAR Y PROCESAR DATOS ---
    print("Cargando datos de prueba...")
    df_test = pd.read_csv(test_dataset.path)
    
    sentences = df_test['corpus'].astype(str).tolist()
    labels = df_test['label'].values
    
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # --- 4. EVALUAR ---
    print("Evaluando modelo...")
    loss, accuracy = model.evaluate(padded, labels, verbose=0)
    print(f"Test Accuracy: {accuracy}")

    metrics.log_metric("Test Accuracy", accuracy)
    metrics.log_metric("Test Loss", loss)
