import os
from kfp.dsl import Dataset, Input, Metrics, Model, Output, component
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.10-slim"),
    packages_to_install=[
        "pandas",
        "numpy",
        "tensorflow", 
        "scikit-learn",
        "google-cloud-storage",
        "protobuf<5.0.0", # Seguro de vida contra errores de KFP
        "urllib3<2.0.0"
    ],
)
def train_lstm_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],      # El modelo .keras
    tokenizer_artifact: Output[Model],  # El diccionario de palabras .pickle
    metrics: Output[Metrics],
    bucket_name: str,
    embedding_path: str,
    epochs: int = 5,
    batch_size: int = 32,
):
    import pandas as pd
    import numpy as np
    import pickle
    import tensorflow as tf
    from google.cloud import storage
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

    # --- 1. CONFIGURACIÓN NLP ---
    VOCAB_SIZE = 20000     # Máximo de palabras únicas a aprender
    MAX_LENGTH = 100       # Longitud fija de los posts
    EMBEDDING_DIM = 100    # Debe coincidir con tu archivo GloVe (100d)
    TRUNC_TYPE = 'post'
    PADDING_TYPE = 'post'
    OOV_TOK = "<OOV>"

    # --- 2. CARGAR DATOS ---
    print("Cargando dataset de entrenamiento...")
    df = pd.read_csv(train_dataset.path)
    
    # Asegurar tipos de datos (corpus=string, label=int)
    sentences = df['corpus'].astype(str).tolist()
    labels = df['label'].values

    # --- 3. DESCARGAR EMBEDDINGS DESDE GCS ---
    local_embedding_file = 'glove_embeddings.txt'
    print(f"Descargando embeddings desde gs://{bucket_name}/{embedding_path}...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(embedding_path)
    blob.download_to_filename(local_embedding_file)
    print("Descarga de embeddings completada.")

    # --- 4. TOKENIZACIÓN ---
    print("Tokenizando textos...")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index

    # Convertir texto a secuencias numéricas
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # --- 5. CREAR MATRIZ DE EMBEDDINGS ---
    # Esto cruza las palabras de tu dataset con las de GloVe
    print("Creando matriz de embeddings...")
    embeddings_index = {}
    with open(local_embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    num_words = min(VOCAB_SIZE, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    hits = 0
    misses = 0
    for word, i in word_index.items():
        if i >= VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            
    print(f"Embeddings encontrados: {hits}, Faltantes: {misses}")

    # --- 6. DEFINIR MODELO LSTM BIDIRECCIONAL ---
    # 
    print("Construyendo modelo LSTM...")
    model = Sequential([
        # Capa de Embeddings (Pre-entrenada con GloVe)
        Embedding(num_words, EMBEDDING_DIM, input_length=MAX_LENGTH, 
                  weights=[embedding_matrix], trainable=False), # False = No re-entrenar GloVe
        
        # Capa LSTM Bidireccional (Lee el texto en ambos sentidos)
        Bidirectional(LSTM(64, return_sequences=False)),
        
        # Capas Densas (Clasificación)
        Dropout(0.5), # Para evitar Overfitting
        Dense(32, activation='relu'),
        
        # Salida: 3 neuronas porque tenemos 3 clases (S, M, L -> 0, 1, 2)
        Dense(3, activation='softmax') 
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # --- 7. ENTRENAR ---
    print("Iniciando entrenamiento...")
    history = model.fit(
        padded_sequences, 
        labels, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1, # Usar 10% de train para validar al vuelo
        verbose=2
    )

    # --- 8. REGISTRAR MÉTRICAS ---
    final_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Accuracy Final: {final_acc}, Val Accuracy: {val_acc}")
    
    metrics.log_metric("accuracy", final_acc)
    metrics.log_metric("val_accuracy", val_acc)

    # --- 9. GUARDAR ARTEFACTOS ---
    # A) Guardar el Modelo (.keras)
    # Keras a veces necesita que la ruta termine en .keras explícitamente o sea un directorio
    model_path = model_artifact.path
    if not model_path.endswith('.keras'):
        model_path += '.keras'
    
    model.save(model_path)
    # Importante: Copiar al path original si KFP añadió una extensión diferente
    if model_path != model_artifact.path:
        import shutil
        shutil.copy(model_path, model_artifact.path)

    # B) Guardar el Tokenizer (.pickle)
    # Sin esto, el modelo es inútil en producción
    with open(tokenizer_artifact.path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Modelo y Tokenizer guardados correctamente.")






# --- 9. GUARDAR ARTEFACTOS ---
# Guardar el modelo como archivo .keras DENTRO del path real del contenedor
model_path = model_artifact.path

if os.path.isdir(model_path):
    model_file = os.path.join(model_path, "model.keras")
else:
    # si KFP entrega un archivo sin extensión, forzar extensión
    model_file = model_path if model_path.endswith(".keras") else model_path + ".keras"

model.save(model_file)

# Guardar tokenizer
with open(tokenizer_artifact.path, "wb") as f:
    pickle.dump(tokenizer, f)
