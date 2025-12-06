import os
from kfp.dsl import Input, Model, component
from dotenv import load_dotenv

load_dotenv()

@component(
    base_image=os.getenv("BASE_IMAGE", "python:3.10-slim"),
    packages_to_install=[
        "google-cloud-storage",
        "protobuf<5.0.0",
        "urllib3<2.0.0"
    ],
)
def save_model_artifacts(
    model_artifact: Input[Model],      # El modelo que salió del entrenamiento
    tokenizer_artifact: Input[Model],  # El tokenizer que salió del entrenamiento
    bucket_name: str,
    destination_folder: str = "modelos/produccion", # Carpeta limpia donde lo quieres
):
    from google.cloud import storage
    import os
    import shutil

    print(f"Iniciando guardado de artefactos en gs://{bucket_name}/{destination_folder}")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # 1. Guardar el Modelo (.keras)
    # Buscamos el archivo real dentro del artefacto de entrada
    model_local_path = model_artifact.path
    # Si KFP nos dio una carpeta, buscamos el archivo dentro
    if os.path.isdir(model_local_path):
        files = os.listdir(model_local_path)
        # Asumimos que hay un archivo .keras o .h5
        model_file = next((f for f in files if f.endswith(('.keras', '.h5'))), None)
        if model_file:
            model_local_path = os.path.join(model_local_path, model_file)
    
    # Definir nombre de destino
    destination_model_blob = f"{destination_folder}/model.keras"
    blob_model = bucket.blob(destination_model_blob)
    
    print(f"Subiendo modelo desde {model_local_path} a {destination_model_blob}...")
    blob_model.upload_from_filename(model_local_path)
    print("Modelo subido exitosamente.")

    # 2. Guardar el Tokenizer (.pickle)
    tokenizer_local_path = tokenizer_artifact.path
    # Definir nombre de destino
    destination_tok_blob = f"{destination_folder}/tokenizer.pickle"
    blob_tok = bucket.blob(destination_tok_blob)
    
    print(f"Subiendo tokenizer desde {tokenizer_local_path} a {destination_tok_blob}...")
    blob_tok.upload_from_filename(tokenizer_local_path)
    print("Tokenizer subido exitosamente.")

    print("¡Despliegue a Bucket completado!")
