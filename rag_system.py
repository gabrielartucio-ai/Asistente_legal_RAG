from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import streamlit as st
from pathlib import PureWindowsPath
from config import *
from prompts import *  

@st.cache_resource
def build_rag_components() -> tuple[Runnable, MultiQueryRetriever]:

    # vector store
    
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )

    # modelos de lenguaje

    llm_queries = init_chat_model(
        model=QUERY_MODEL,
        model_provider=QUERY_MODEL_PROVIDER,
        temperature=QUERY_TEMPERATURE 
    )
    llm_generation = init_chat_model(
        model=GENERATION_MODEL,
        model_provider=GENERATION_MODEL_PROVIDER,
        temperature=GENERATION_TEMPERATURE         
    )

    # Retriever MMR (Maximal Margin Relevance)

    base_retriever = vector_store.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_K,
            "lambda_mult": MMR_DIVERSITY,
            "fetch_k": MMR_FETCH_K
        } 
    )

    # Prompt personalizado para el MultiQueryRetriever

    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado

    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    rag_prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    """
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split('\\')[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Página {doc.metadata['page']}"
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)    
    """
    # Función para preprocesar y formatear los fragmentos recuperados
         
    def format_docs(
        docs: list[Document],
        *,
        prefix: str = "Fragmento",
        source_label: str = "Fuente",
        page_label: str = "Página",
        separator: str = "\n\n") -> str:
        """Une documentos recuperados en un único bloque de contexto para el prompt.

        Args:
            docs: Documentos devueltos por el retriever.
            prefix: Texto del encabezado de cada fragmento.
            source_label: Etiqueta del campo ``metadata["source"]``.
            page_label: Etiqueta del campo ``metadata["page"]``.
            page_offset: Se suma al número de página (``1`` si el loader indexa desde 0).
            separator: Separador entre fragmentos.
        """

        def header(i: int, metadata: dict) -> str:
            parts = [f"[{prefix} {i}]"]
            if source := metadata.get("source"):
                parts.append(f"{source_label}: {PureWindowsPath(source).name}")
            if (page := metadata.get("page")) is not None:
                parts.append(f"{page_label} {page + 1}")
            return " - ".join(parts)

        return separator.join(
            f"{header(i, doc.metadata)}\n{doc.page_content.strip()}"
            for i, doc in enumerate(docs, 1)
        )

    rag_chain = (
        {"context": mmr_multi_retriever | format_docs,
        "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever


def query_rag(question) -> tuple[str, list[dict]]:
    try:
        rag_chain, retriever = build_rag_components()

        # Obtener respuesta. 
        # Es un str porque uso StrOutputParser al final de la cadena langchain
        response: str = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos. 
        # El método invoke() de MultiQueryRetriever retorna una lista de objetos Document 
        # de Langchain. 
        # Cada objeto Document contiene principalmente: 
        # - page_content: contenido textual del fragmento (str). 
        # - metadata: metadatos asociados al fragmento (dict).

        documents: list[Document] = retriever.invoke(question)

        # Formatear los documentos para mostrarlos
        documents_info = []
        for i, doc in enumerate(documents[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000],
                "fuente": PureWindowsPath(doc.metadata.get('source', '')).name or 'Fuente no especificada',
                "pagina": doc.metadata.get('page', 'Página no especificada')
            }
            documents_info.append(doc_info)
        return response, documents_info

    except Exception as e:
        error_msg = f"Error al procesar la consulta {str(e)}"
        return error_msg, []

def get_retriever_info():
    """Obtiene información sobre la configuración del retriever"""
    return{
        "tipo": f"{SEARCH_TYPE.upper()}",
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY,
        "candidatos": MMR_FETCH_K,
        "umbral": None
    }

def get_models_info() -> dict[str, str]:
    """Obtiene información sobre los modelos configurados"""
    return {
        "consulta": QUERY_MODEL,
        "proveedor_consulta": QUERY_MODEL_PROVIDER,
        "generacion": GENERATION_MODEL,
        "proveedor_generacion": GENERATION_MODEL_PROVIDER,
    }