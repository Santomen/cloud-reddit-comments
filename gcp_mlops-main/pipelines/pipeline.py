import os
import sys
import google.cloud.aiplatform as aip
import kfp
from kfp import dsl
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Asegurar que Python encuentre la carpeta de componentes
sys.path.append("src")

# Configuración Global obtenida del .env
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "reddit-lstm-pipeline-v1")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")

@dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    bucket_name: str,
    embedding_path: str,
    epochs: int = 5,
    destination_folder: str = "modelos/reddit_produccion" # Carpeta final limpia
):
    # Importamos los componentes aquí dentro para evitar errores de compilación
    from components.data import load_data
    from components.models import train_lstm_model
    from components.evaluation import evaluate_lstm_model
    from components.deploy import save_model_artifacts

    # --- PASO 1: Ingesta y Balanceo (BigQuery) ---
    data_op = load_data(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table=bq_table
    ).set_display_name("1. Ingesta y Balanceo")

    # --- PASO 2: Entrenamiento (LSTM + Embeddings) ---
    train_op = train_lstm_model(
        train_dataset=data_op.outputs["train_dataset"],
        bucket_name=bucket_name,
        embedding_path=embedding_path,
        epochs=epochs
    ).set_display_name("2. Entrenar LSTM Bidireccional")
    
    # (Opcional) Aumentar memoria si el entrenamiento falla por OOM
    # train_op.set_memory_limit('4G')

    # --- PASO 3: Evaluación (Test Set) ---
    eval_op = evaluate_lstm_model(
        test_dataset=data_op.outputs["test_dataset"],
        model_artifact=train_op.outputs["model_artifact"],
        tokenizer_artifact=train_op.outputs["tokenizer_artifact"]
    ).set_display_name("3. Evaluar Exactitud")

    # --- PASO 4: Publicación de Artefactos (Sin Endpoint) ---
    # Guarda el .keras y .pickle en una carpeta limpia del bucket
    save_op = save_model_artifacts(
        model_artifact=train_op.outputs["model_artifact"],
        tokenizer_artifact=train_op.outputs["tokenizer_artifact"],
        bucket_name=bucket_name,
        destination_folder=destination_folder
    ).set_display_name("4. Publicar Archivos Finales")
    
    # Ordenar ejecución: Guardar solo después de evaluar
    save_op.after(eval_op)

if __name__ == "__main__":
    # Nombre del archivo compilado
    compiler_file = "pipeline.yaml"

    # 1. Compilar el pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline, 
        package_path=compiler_file
    )
    print(f"✅ Pipeline compilado en {compiler_file}")

    # 2. Inicializar Vertex AI SDK
    aip.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=PIPELINE_ROOT,
    )

    # 3. Definir parámetros dinámicos (leyendo del .env)
    pipeline_params = {
        "project_id": PROJECT_ID,
        "bq_dataset": os.getenv("BQ_DATASET"),
        "bq_table": os.getenv("BQ_TABLE"),
        "bucket_name": os.getenv("BUCKET_NAME"),
        "embedding_path": os.getenv("EMBEDDING_PATH"),
        "epochs": 5,
        "destination_folder": "modelos/reddit_produccion_v1" # Puedes cambiar la versión aquí
    }

    # 4. Crear y Ejecutar el Job
    print("🚀 Enviando Job a Vertex AI...")
    job = aip.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=compiler_file,
        pipeline_root=PIPELINE_ROOT,
        parameter_values=pipeline_params,
        enable_caching=False, # False para obligar a correr todo de nuevo (útil para debug)
    )

    job.run()
