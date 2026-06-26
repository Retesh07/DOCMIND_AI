from typing import TypedDict, List, Optional
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    pdf_path: str
    documents: List[Document]
    filtered_documents: List[Document]
    answer: str                   
    sources: Optional[List[dict]]
    retry_count: int
    hallucination_status: str