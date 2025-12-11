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
        "protobuf<5.0.0",
        "urllib3<2.0.0"
    ],
)
def train_lstm_model(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    tokenizer_artifact: Output[Model],
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
    import shutil

    # Config
    VOCAB_SIZE = 20000
    MAX_LENGTH = 100
    EMBEDDING_DIM = 100
    TRUNC_TYPE = 'post'
    PADDING_TYPE = 'post'
    OOV_TOK = "<OOV>"

    # Cargar dataset
    df = pd.read_csv(train_dataset.path)
    sentences = df['corpus'].astype(str).tolist()
    labels = df['label'].values

    # Descargar embeddings desde GCS
    local_embedding_file = 'glove_embeddings.txt'
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(embedding_path)
    blob.download_to_filename(local_embedding_file)

    # Tokenización
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # Matriz de embeddings
    embeddings_index = {}
    with open(local_embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    num_words = min(VOCAB_SIZE, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Modelo LSTM
    model = Sequential([
        Embedding(num_words, EMBEDDING_DIM, input_length=MAX_LENGTH, weights=[embedding_matrix], trainable=False),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(
        padded_sequences,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2
    )

    # Métricas
    final_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    metrics.log_metric("accuracy", final_acc)
    metrics.log_metric("val_accuracy", val_acc)

    # Guardar modelo
    model_path = model_artifact.path
    if not model_path.endswith(".keras"):
        fixed_path = model_path + ".keras"
        model.save(fixed_path)
        shutil.copy(fixed_path, model_artifact.path)
    else:
        model.save(model_path)

    # Guardar tokenizer
    with open(tokenizer_artifact.path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Modelo y Tokenizer guardados correctamente.")
