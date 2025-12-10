import os
from kfp.dsl import Input, Model, component

# --- COMPONENTE DE DESPLIEGUE ---
@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-storage<3.0.0",  # <--- FIX 1: Versión compatible con Kubeflow
        "kfp",                         # <--- FIX 2: Instala dependencias base
        "protobuf<5.0.0",
        "urllib3<2.0.0"
    ],
)
def save_model_artifacts(
    model_artifact: Input[Model],       
    tokenizer_artifact: Input[Model],   
    bucket_name: str,
    destination_folder: str = "modelos/produccion", 
):
    from google.cloud import storage
    import os
    
    print(f"Iniciando guardado de artefactos en gs://{bucket_name}/{destination_folder}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # --- 1. Guardar el Modelo ---
    model_local_path = model_artifact.path
    print(f"Path recibido del modelo: {model_local_path}")

    # FIX 3: Si Vertex nos da una carpeta, buscamos el archivo .keras dentro
    final_model_path = model_local_path
    if os.path.isdir(model_local_path):
        print("El artefacto es una carpeta, buscando archivo .keras o .h5 dentro...")
        files = os.listdir(model_local_path)
        # Buscamos .keras o .h5
        model_file = next((f for f in files if f.endswith(('.keras', '.h5'))), None)
        if model_file:
            final_model_path = os.path.join(model_local_path, model_file)
            print(f"Archivo de modelo encontrado: {final_model_path}")
        else:
            print("ADVERTENCIA: No se encontró archivo .keras/.h5 en la carpeta. Se intentará subir la ruta original.")
            
    # Asignar nombre destino
    destination_model_blob = f"{destination_folder}/model.keras"
    blob_model = bucket.blob(destination_model_blob)
    
    print(f"Subiendo modelo desde {final_model_path} a {destination_model_blob}...")
    try:
        blob_model.upload_from_filename(final_model_path)
        print("Modelo subido exitosamente.")
    except Exception as e:
        print(f"Error subiendo modelo: {e}")
        # Si falló, listar contenido para debuggear
        if os.path.exists(final_model_path):
             print(f"El archivo existe localmente. Tamaño: {os.path.getsize(final_model_path)}")

    # --- 2. Guardar el Tokenizer ---
    tokenizer_local_path = tokenizer_artifact.path
    destination_tok_blob = f"{destination_folder}/tokenizer.pickle"
    blob_tok = bucket.blob(destination_tok_blob)
    
    print(f"Subiendo tokenizer desde {tokenizer_local_path} a {destination_tok_blob}...")
    blob_tok.upload_from_filename(tokenizer_local_path)
    print("Tokenizer subido exitosamente.")

    print("¡Despliegue a Bucket completado!")
