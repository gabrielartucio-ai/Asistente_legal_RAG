from langchain_core.documents import Document
from pathlib import PureWindowsPath

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
