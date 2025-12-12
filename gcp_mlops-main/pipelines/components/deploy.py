import os
from kfp.dsl import Input, Model, component

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-storage<3.0.0",
        "kfp",
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
    
    print(f"Iniciando despliegue a: gs://{bucket_name}/{destination_folder}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # --- 1. ENCONTRAR EL MODELO (SIN RENOMBRARLO) ---
    base_path = model_artifact.path
    final_source_path = None
    
    print(f"Path base recibido de Vertex: {base_path}")

    # ESTRATEGIA: Buscar qué archivo existe realmente
    if os.path.exists(base_path) and os.path.isfile(base_path):
        print("-> Encontrado archivo exacto en el path base.")
        final_source_path = base_path
    elif os.path.exists(base_path + ".keras"):
        print("-> Encontrado archivo con sufijo .keras oculto.")
        final_source_path = base_path + ".keras"
    elif os.path.isdir(base_path):
        print("-> Es un directorio. Buscando contenido...")
        files = os.listdir(base_path)
        # Buscar cualquier archivo grande que parezca el modelo
        found = next((f for f in files if f.endswith(('.keras', '.h5'))), None)
        if found:
            final_source_path = os.path.join(base_path, found)
            print(f"-> Archivo encontrado dentro del directorio: {found}")
    
    # Si aún no lo encontramos, listamos el directorio padre para ver qué demonios hay
    if not final_source_path:
        parent_dir = os.path.dirname(base_path)
        print(f"ERROR: No se encuentra el archivo. Listando directorio padre {parent_dir}:")
        if os.path.exists(parent_dir):
            print(os.listdir(parent_dir))
        raise RuntimeError("No se pudo localizar el archivo del modelo para subirlo.")

    # --- 2. SUBIRLO CON EL NOMBRE CORRECTO ---
    # Aquí está el truco: Leemos 'final_source_path' (que puede no tener extensión)
    # y lo escribimos en el bucket como 'model.keras'.
    
    destination_model_blob = f"{destination_folder}/model.keras"
    blob_model = bucket.blob(destination_model_blob)
    
    print(f"Subiendo desde: {final_source_path}")
    print(f"Hacia Bucket: {destination_model_blob}")
    
    blob_model.upload_from_filename(final_source_path)
    print(">>> Modelo subido EXITOSAMENTE.")

    # --- 3. SUBIR TOKENIZER ---
    tokenizer_path = tokenizer_artifact.path
    destination_tok_blob = f"{destination_folder}/tokenizer.pickle"
    blob_tok = bucket.blob(destination_tok_blob)
    
    print(f"Subiendo tokenizer...")
    blob_tok.upload_from_filename(tokenizer_path)
    print(">>> Tokenizer subido EXITOSAMENTE.")

    print("¡Todo completado!")
