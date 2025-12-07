import os
from kfp.dsl import Input, Model, component
# from dotenv import load_dotenv # (Opcional si ya cargas variables antes)

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-storage<3.0.0",  # <--- EL FIX: Forzamos versión antigua compatible
        "kfp",                         # <--- Agregamos esto para asegurar que instale las dependencias de kubeflow
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
    import shutil

    print(f"Iniciando guardado de artefactos en gs://{bucket_name}/{destination_folder}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # --- 1. Guardar el Modelo ---
    model_local_path = model_artifact.path
    print(f"Path recibido del modelo: {model_local_path}")

    # Lógica para encontrar el archivo real si es una carpeta
    final_model_path = model_local_path
    if os.path.isdir(model_local_path):
        files = os.listdir(model_local_path)
        # Buscamos .keras o .h5
        model_file = next((f for f in files if f.endswith(('.keras', '.h5'))), None)
        if model_file:
            final_model_path = os.path.join(model_local_path, model_file)
    
    # IMPORTANTE: Vertex a veces entrega el archivo sin extensión si no se renombró antes.
    # Si final_model_path no tiene extensión, asumimos que es el archivo correcto.
    
    destination_model_blob = f"{destination_folder}/model.keras"
    blob_model = bucket.blob(destination_model_blob)
    
    print(f"Subiendo modelo desde {final_model_path} a {destination_model_blob}...")
    blob_model.upload_from_filename(final_model_path)
    print("Modelo subido exitosamente.")

    # --- 2. Guardar el Tokenizer ---
    tokenizer_local_path = tokenizer_artifact.path
    destination_tok_blob = f"{destination_folder}/tokenizer.pickle"
    blob_tok = bucket.blob(destination_tok_blob)
    
    print(f"Subiendo tokenizer desde {tokenizer_local_path} a {destination_tok_blob}...")
    blob_tok.upload_from_filename(tokenizer_local_path)
    print("Tokenizer subido exitosamente.")

    print("¡Despliegue a Bucket completado!")
