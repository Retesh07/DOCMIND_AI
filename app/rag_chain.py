from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

def get_fast_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )
def get_prompt_template():
    template = """You are a helpful assistant that answers questions 
based ONLY on the provided context.

Rules:
- Only use information from the context below
- If the answer is not in the context, say "I don't have enough information in the document to answer this."
- Always be concise and accurate
- Do not make up information

Context:
{context}

Question: {question}

Answer:"""
    return ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    llm = get_llm()
    prompt = get_prompt_template()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain