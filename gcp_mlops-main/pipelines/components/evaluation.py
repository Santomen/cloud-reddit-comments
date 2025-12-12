import os
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(
    base_image="python:3.10-slim",
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
    import os
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
    print("Cargando tokenizer...")
    with open(tokenizer_artifact.path, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    print("Cargando modelo...")
    model_path = model_artifact.path
    print(f"Path original recibido de Vertex AI: {model_path}")


    if not model_path.endswith(".keras"):
        new_path = model_path + ".keras"
        print(f"Renombrando archivo a: {new_path}")
        os.rename(model_path, new_path)
        model_path = new_path
        
    model = load_model(model_path)
    print("Modelo cargado exitosamente.")

    # --- 3. CARGAR Y PROCESAR DATOS DE TEST ---
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

    # Loguear métrica principal
    metrics.log_metric("Test Accuracy", accuracy)
    metrics.log_metric("Test Loss", loss)
    
    # (Opcional) Generar matriz de confusión para logs
    predictions = model.predict(padded)
    pred_classes = np.argmax(predictions, axis=1)
    
    cm = confusion_matrix(labels, pred_classes)
    print("Matriz de Confusión:")
    print(cm)
