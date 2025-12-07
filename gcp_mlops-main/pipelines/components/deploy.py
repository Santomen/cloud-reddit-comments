import os
from kfp.dsl import Input, Model, component
# from dotenv import load_dotenv # (Opcional si ya cargas variables antes)

@component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "google-cloud-storage<3.0.0",
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

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # === 1) Resolver path real del modelo ===
    model_local_path = model_artifact.path
    print("Path recibido:", model_local_path)

    final_model_path = model_local_path

    # Si es carpeta, busca archivo dentro
    if os.path.isdir(model_local_path):
        files = os.listdir(model_local_path)
        model_file = next((f for f in files if f.endswith(('.keras', '.h5'))), None)
        if model_file:
            final_model_path = os.path.join(model_local_path, model_file)

    # Si el archivo no tiene extensión, ponle .keras
    if not final_model_path.endswith(".keras"):
        new_path = final_model_path + ".keras"
        os.rename(final_model_path, new_path)
        final_model_path = new_path

    # === 2) Subir modelo ===
    model_blob_path = f"{destination_folder}/model.keras"
    bucket.blob(model_blob_path).upload_from_filename(final_model_path)

    # === 3) Subir tokenizer ===
    tokenizer_blob_path = f"{destination_folder}/tokenizer.pickle"
    bucket.blob(tokenizer_blob_path).upload_from_filename(tokenizer_artifact.path)

    print("🎉 Artefactos subidos correctamente.")
