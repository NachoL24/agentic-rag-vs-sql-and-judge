import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "historias_clinicas"

LLM_MODEL = "qwen2.5:0.5b"
EMBEDDINGS_MODEL = "nomic-embed-text"


def load_documents():
    #Cargamos todas las historias clínicas desde la carpeta data/
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"No existe el directorio {DATA_DIR}. "
            f"Crealo y poné ahí tus historias clínicas."
        )

    docs = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            if path.suffix.lower() == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            elif path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())

    if not docs:
        raise ValueError(
            f"No se encontraron documentos en {DATA_DIR}. "
            f"Agregá historias clínicas simuladas en .txt o .pdf."
        )

    return docs


def split_documents(docs):
    # Hace chunking de los documentos para mejorar el RAG.
    # Ajustá chunk_size y chunk_overlap según tus textos.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store_from_scratch():
    # Crea el índice en Chroma a partir de los documentos en data/.
    # Se persiste en chroma_db/ para reusar entre ejecuciones.
    print("Cargando documentos...")
    docs = load_documents()
    print(f"   -> {len(docs)} documentos encontrados")

    print("Creamos los chunks de los documentos")
    splits = split_documents(docs)
    print(f"   -> {len(splits)} chunks generados")

    print("Creando embeddings")
    
    import os
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST,
    )


    print("Creando vectores")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print("Indice construido y persistido en", CHROMA_DIR)
    return vectordb


def get_vector_store():
    # Si existe un índice persistido en chroma_db/, lo carga.
    # Si no, lo construye desde cero.
    import os
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDINGS_MODEL,
        base_url=OLLAMA_HOST,
    )

    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return build_vector_store_from_scratch()

    print("Cargando índice Chroma existente")
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectordb


def format_docs(docs):
    """
    Formatea los documentos recuperados para inyectarlos al prompt.
    Le agrega numeración para debugging y evaluación.
    """
    partes = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "desconocido")
        partes.append(
            f"[FRAGMENTO {i} - {source}]\n{d.page_content.strip()}"
        )
    return "\n\n".join(partes)


def create_rag_chain():
    # Crea la cadena RAG    
    vectordb = get_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    import os

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_HOST,
    )

    system_template = """
Eres un asistente médico que responde exclusivamente en base
a la información de historias clínicas proporcionadas en el CONTEXTO.

Reglas:
- Si la información no está en el contexto, responde claramente que no puedes asegurarlo.
- No inventes diagnósticos ni medicaciones.
- Si la pregunta es ambigua, acláralo en la respuesta.

Contexto:
{context}
"""

    human_template = """
Consulta del usuario:
{question}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            ("human", human_template),
        ]
    )

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_docs,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def chat_loop():
    print("Escribí tu pregunta médica (o 'salir' para terminar).\n")

    rag_chain = create_rag_chain()

    while True:
        try:
            question = input("👤> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Saliendo...")
            break

        if question.lower() in {"salir", "exit", "quit"}:
            print("👋 Listo, nos vemos.")
            break

        if not question:
            continue

        print("🤖> (Pensando...)\n")
        answer = rag_chain.invoke(question)
        print(f"🤖 {answer}\n")


if __name__ == "__main__":
    chat_loop()