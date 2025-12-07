import os
from kfp.dsl import Input, Model, component

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
    import os
    from google.cloud import storage

    print("=== GUARDANDO MODELO Y TOKENIZER ===")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # -------------------------
    # 🔹 1. UBICAR ARCHIVO .KERAS
    # -------------------------
    model_path = model_artifact.path
    print(f"Ruta entregada por KFP: {model_path}")

    # Si es carpeta, buscar ahí dentro
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        keras_file = next((f for f in files if f.endswith(".keras")), None)
        h5_file = next((f for f in files if f.endswith(".h5")), None)

        if keras_file:
            model_path = os.path.join(model_path, keras_file)
        elif h5_file:
            model_path = os.path.join(model_path, h5_file)
        else:
            # último recurso: cualquier archivo dentro
            model_path = os.path.join(model_artifact.path, files[0])

    print(f"Archivo real del modelo: {model_path}")

    # -------------------------
    # 🔹 2. SUBIR MODELO
    # -------------------------
    dst_model = f"{destination_folder}/model.keras"
    blob = bucket.blob(dst_model)

    print(f"Subiendo modelo a gs://{bucket_name}/{dst_model}")
    blob.upload_from_filename(model_path)
    print("✔ Modelo subido correctamente.")

    # -------------------------
    # 🔹 3. SUBIR TOKENIZER
    # -------------------------
    tokenizer_path = tokenizer_artifact.path
    dst_tokenizer = f"{destination_folder}/tokenizer.pickle"

    print(f"Subiendo tokenizer a gs://{bucket_name}/{dst_tokenizer}")
    blob2 = bucket.blob(dst_tokenizer)
    blob2.upload_from_filename(tokenizer_path)

    print("✔ Tokenizer subido correctamente.")
    print("=== PUBLICACIÓN COMPLETA ===")
