from pathlib import Path

# Configuración de modelos

EMBEDDING_MODEL = "BAAI/bge-m3"
QUERY_MODEL = "openai/gpt-oss-120b"
QUERY_MODEL_PROVIDER = "groq"
QUERY_TEMPERATURE = 0.0
GENERATION_MODEL = "openai/gpt-oss-120b"
GENERATION_MODEL_PROVIDER = "groq"
GENERATION_TEMPERATURE = 0.0

# Configuración del vector store

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(PROJECT_ROOT / "Chroma_db")

# Configuración del retriever

SEARCH_TYPE = "mmr"
MMR_DIVERSITY = 0.7
MMR_FETCH_K = 20
SEARCH_K = 2