"""
Descripción:
    Carga documentos PDF de contratos, divide su contenido en fragmentos
    de texto (chunks), genera embeddings mediante un modelo local de Hugging Face
    y los almacena en una base de datos vectorial Chroma.

    Usa HuggingFaceEmbeddings que es el conector o adaptador (Wrapper) de LangChain 
    (langchain-huggingface). Es el apropiado para ejecutar modelos de embeddings 
    de forma local con bases de datos vectoriales (Chroma, Pinecone, etc) 
    integradas dentro de LangChain.

    Chroma.from_documents(), Retrievers, o cadenas conversacionales (RetrievalQA) exigen que el 
    objeto de embeddings implemente la interfaz estándar de LangChain. Un cliente nativo de Google 
    fallará aquí porque no tiene los métodos internos que LangChain espera. 

    Ademas, permite cambiar de proveedor de embeddings cambiando una sola línea de código sin romper 
    el resto de la arquitectura de base de datos.
"""    

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from config import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = EMBEDDING_MODEL

loader = PyPDFDirectoryLoader(PROJECT_ROOT/"Tema 3"/"Contratos")
documents = loader.load()

print(f"Se cargaron {len(documents)} documentos")

# Se descarga una sola vez y corre 100% en la computadora (sin API key ni límites)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap = 1000
)
split_documents = text_splitter.split_documents(documents)

print(f"Se crearon {len(split_documents)} chunks")

vectorstore = Chroma.from_documents(
    split_documents,
    embedding=embeddings,
    persist_directory=str(PROJECT_ROOT/"Tema 3"/"Chroma_db")
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"

resultados = vectorstore.similarity_search(consulta, k=5)

print("\nDocumentos mas similares a la consulta:\n")
for i, doc in enumerate(resultados, start=1):
    print(f"Chunk {i}\n")
    print(f"Contenido: {doc.page_content}\n")
