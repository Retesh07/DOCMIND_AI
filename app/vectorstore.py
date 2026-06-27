import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.ingestion import load_and_split_pdf
from langchain_chroma import Chroma
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PATH = "/tmp/chroma_db"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        show_progress=False    
    )


def get_or_create_vectorstore(pdf_path: str):
    embeddings = get_embeddings()
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    db_path = os.path.join(CHROMA_PATH, pdf_name)
    os.makedirs(db_path, exist_ok=True)

    # Check if vectorstore already exists
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    if vectorstore._collection.count() > 0:
        print(f"✅ Vectorstore already exists for {pdf_name} — skipping ingestion")
        return vectorstore

    # Only create if empty
    chunks = load_and_split_pdf(pdf_path)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    print(f"✅ Created vectorstore for {pdf_name}")
    print(f"✅ Total chunks: {len(chunks)}")
    return vectorstore


def load_vectorstore(pdf_path: str):

    embeddings = get_embeddings()

    pdf_name = os.path.splitext(
        os.path.basename(pdf_path)
    )[0]

    db_path = os.path.join(
        CHROMA_PATH,
        pdf_name
    )

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    print(
        "Loaded collection count:",
        vectorstore._collection.count()
    )

    return vectorstore


def get_retriever(vectorstore):

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )