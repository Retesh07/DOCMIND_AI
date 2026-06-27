from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from app.state import AgentState
from app.vectorstore import load_vectorstore, get_retriever
from app.rag_chain import format_docs, get_llm, get_fast_llm
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# SCHEMAS (for structured LLM output)
# ============================================================

class GradeDocument(BaseModel):
    """Binary score for relevance check on retrieved document."""
    binary_score: str = Field(
        description="'yes' if the chunk contains ANY information that could help answer the question, even partially. Default to 'yes' when uncertain."
    )

class HallucinationCheck(BaseModel):
    """Check whether answer is grounded in retrieved documents."""
    
    binary_score: str = Field(
        description="'yes' if answer is grounded in documents, otherwise 'no'"
    )


# ============================================================
# NODE 2: RETRIEVER
# No LLM call — just vector search
# ============================================================

def retriever_node(state: AgentState):
    question = state["question"]
    pdf_path = state["pdf_path"]      # ← read from state

    print(f"🔍 Retrieving chunks for: {question}")

   
    vectorstore = load_vectorstore(pdf_path)   # ← pass pdf_path
    retriever = get_retriever(vectorstore)
    documents = retriever.invoke(question)

    print(f"📄 Retrieved {len(documents)} chunks")
    return {"documents": documents}
# ============================================================
# NODE 3: GRADER
# Uses powerful LLM — filters retrieved chunks by relevance
# ============================================================

def grader_node(state: AgentState):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("⚠️ No documents retrieved — skipping grading")
        return {"filtered_documents": []}

    llm = get_llm()
    structured_llm = llm.with_structured_output(GradeDocument)

    grade_prompt = ChatPromptTemplate.from_template("""
You are a lenient grader assessing document chunk relevance.

Retrieved chunk:
{document}

User question:
{question}

Does this chunk contain ANY information that could help answer the question, even partially?
When in doubt, answer 'yes'.
Answer ONLY 'yes' or 'no'.
""")

    grader_chain = grade_prompt | structured_llm

    filtered_docs = []

    for doc in documents:
        result = grader_chain.invoke({
            "question": question,
            "document": doc.page_content
        })

        if isinstance(result, dict):
            binary_score = result.get("binary_score", "no")
        elif isinstance(result, BaseModel):
            binary_score = getattr(result, "binary_score", "no")
        else:
            print(f"⚠️ Unexpected grade result format: {type(result)}")
            binary_score = "no"

        if str(binary_score).strip().lower() == "yes":
            filtered_docs.append(doc)
            print(f"✅ Relevant: {doc.page_content[:60]}...")
        else:
            print(f"❌ Not relevant: {doc.page_content[:60]}...")

    print(f"📊 Grading complete: {len(filtered_docs)}/{len(documents)} chunks kept")

    if not filtered_docs:
        return {"filtered_documents": []}

    return {"filtered_documents": filtered_docs}

# ============================================================
# NODE 4: GENERATOR
# Uses powerful LLM — generates answers based on filtered chunks
# ============================================================

def generator_node(state: AgentState):
    question = state["question"]
    filtered_documents = state.get("filtered_documents", [])

    # Safety net — graph.py should route around this node entirely
    # when there are no relevant docs, but we guard here too.
    if not filtered_documents:
        print("⚠️ generator_node called with no filtered documents")
        return {
            "answer": "I could not find the answer in the document.",
            "sources": []
        }

    llm = get_llm()

    context = format_docs(filtered_documents)

    generation_prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question ONLY using the provided context.
Always respond in complete sentences with sufficient detail.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
""")

    generation_chain = generation_prompt | llm | StrOutputParser()

    answer = generation_chain.invoke({
        "context": context,
        "question": question
    })

    sources = [
        {
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:150]
        }
        for doc in filtered_documents
    ]

    print(f"💬 Generated answer using {len(filtered_documents)} source chunk(s)")

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# NODE 5: QUERY REWRITER
# Rephrases question when retrieval quality is poor
# Uses fast LLM — simple rephrasing task
# ============================================================

def rewriter_node(state: AgentState):
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    print(f"🔄 Rewriting query (attempt {retry_count})")
    
    rewriter_prompt = ChatPromptTemplate.from_template("""
    You are a query rewriter for document retrieval.

    IMPORTANT RULES:
    - Keep ALL acronyms, technical terms, and proper nouns EXACTLY as they appear
    - Only rephrase the question structure  
    - Do NOT expand or interpret acronyms
    - Do NOT add information not in the original question

    Original question: {question}

    Rewritten question:""")
    
    llm = get_fast_llm()
    rewriter_chain = rewriter_prompt | llm | StrOutputParser()
    
    rewritten_question = rewriter_chain.invoke({
        "question": question
    }).strip()

    print(f"✏️ Original:  {question}")
    print(f"✏️ Rewritten: {rewritten_question}")

    return {
        "question": rewritten_question,
        "retry_count": retry_count
    }


# ============================================================
# NODE 6: Hallucination node
# ============================================================
def hallucination_checker_node(state: AgentState):
    question = state["question"]
    answer = state["answer"]
    filtered_documents = state.get("filtered_documents", [])

    if not filtered_documents:
        return {"hallucination_status": "yes"}  # no docs = skip check = pass

    docs_text = "\n\n".join([doc.page_content for doc in filtered_documents])

    hallucination_prompt = ChatPromptTemplate.from_template("""
You are a hallucination checker.

Retrieved Documents:
{documents}

Generated Answer:
{answer}

Determine whether the generated answer is fully supported by the retrieved documents.

Return:
- yes -> if answer is grounded in documents
- no -> if answer contains information not supported by documents
""")

    llm = get_llm()
    structured_llm = llm.with_structured_output(HallucinationCheck)
    hallucination_chain = hallucination_prompt | structured_llm

    result = hallucination_chain.invoke({
        "documents": docs_text,
        "answer": answer
    })

    if isinstance(result, dict):
        score = result.get("binary_score", "yes")
    elif isinstance(result, BaseModel):
        score = getattr(result, "binary_score", "yes")
    else:
        score = "yes"

    score = str(score).strip().lower()

    print(f"🧠 Hallucination Check: {score}")

    return {"hallucination_status": score}  # ← no retry_count here