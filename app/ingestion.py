from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str):

    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"\nLoaded {len(documents)} pages")

 

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks




if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Error: Please provide the path to a PDF file.")
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File not found at '{pdf_path}'")
        sys.exit(1)

    print(f"Starting ingestion for: {pdf_path}")

    try:
        chunks = load_and_split_pdf(pdf_path)
        if chunks:
            print(f"\nSuccess! Created {len(chunks)} chunks from the PDF.")

    except Exception as e:
        print(f"Error during processing: {e}")