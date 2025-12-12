import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from google.cloud import storage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tempfile

# --- Configuración ---
# Detalles del bucket y archivos
BUCKET_NAME = "proyecto-reddit-maestria"
BLOB_FOLDER = "proyecto-reddit-maestria/modelos/reddit_produccion_v1"
MODEL_BLOB = f"{BLOB_FOLDER}/model.keras"
TOKENIZER_BLOB = f"{BLOB_FOLDER}/tokenizer.pickle"

# Configuración del modelo (de tu notebook)
MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'
CLASES = {0: "Clase 0 (Bajo Debate)", 1: "Clase 1 (Medio Debate)", 2: "Clase 2 (Alto Debate)"} # Ajusta las etiquetas según tu necesidad

# --- Inicialización de la Aplicación y Carga del Modelo ---
app = Flask(__name__)

# Variable global para almacenar el modelo y el tokenizer
model = None
tokenizer = None

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Descarga un archivo (blob) del bucket de GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    print(f"Descargando {source_blob_name} a {destination_file_name}...")
    blob.download_to_filename(destination_file_name)
    print("Descarga completa.")

def load_brain():
    """Descarga y carga el modelo y el tokenizer de GCS."""
    global model, tokenizer

    try:
        # Usamos tempfile para crear archivos temporales donde guardar los modelos descargados
        with tempfile.NamedTemporaryFile(suffix='.pickle', delete=False) as tmp_tokenizer:
            tokenizer_path = tmp_tokenizer.name
        
        with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_model:
            model_path = tmp_model.name
        
        # 1. Descargar el tokenizer
        download_blob(BUCKET_NAME, TOKENIZER_BLOB, tokenizer_path)
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)

        # 2. Descargar y cargar el modelo
        # NOTA: Los modelos Keras cargados desde GCS deben ser guardados/leídos en un sistema de archivos local primero.
        download_blob(BUCKET_NAME, MODEL_BLOB, model_path)
        model = load_model(model_path)
        
        print("✅ ¡Modelo y Tokenizer cargados y listos!")

    except Exception as e:
        print(f"❌ Error al cargar los archivos: {e}")
        # En Cloud Run, fallar la carga aquí provocará que el contenedor no inicie
    finally:
        # Limpiar archivos temporales
        if 'tokenizer_path' in locals() and os.path.exists(tokenizer_path):
            os.remove(tokenizer_path)
        if 'model_path' in locals() and os.path.exists(model_path):
            os.remove(model_path)


# Cargar el modelo al inicio (antes de que Cloud Run empiece a enviar tráfico)
# Esto se llama justo cuando el servidor Flask se inicia.
load_brain()

# --- Rutas de Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Ruta principal para la interfaz y la predicción."""
    prediction_result = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('post_text', '')
        
        if input_text and model and tokenizer:
            # Procesar (igual que en tu notebook)
            seq = tokenizer.texts_to_sequences([input_text])
            padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
            
            # Predecir
            pred = model.predict(padded, verbose=0)
            clase = np.argmax(pred)
            confianza = np.max(pred) * 100
            
            prediction_result = {
                'clase': CLASES.get(clase, f"Clase desconocida: {clase}"),
                'confianza': f"{confianza:.2f}%"
            }
        elif not input_text:
            prediction_result = {'error': "Por favor, introduce algún texto para predecir."}
        else:
            prediction_result = {'error': "Error: Modelo no cargado. Revisa los logs de Cloud Run."}

    return render_template('index.html', result=prediction_result, input_text=input_text)

if __name__ == '__main__':
    # Para ejecución local (no necesario en Cloud Run, pero útil para pruebas)
    # Cloud Run usa el comando 'gunicorn' o similar para ejecutar la aplicación, no este bloque.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)