from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.state import AgentState
from app.vectorstore import load_vectorstore, get_retriever
from app.rag_chain import format_docs, get_llm, get_fast_llm

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# NODE 2: RETRIEVER
# No LLM call — just vector search
# ============================================================

def retriever_node(state: AgentState):
    question = state["question"]
    pdf_path = state["pdf_path"]

    print(f"🔍 Retrieving chunks for: {question}")

    vectorstore = load_vectorstore(pdf_path)
    retriever = get_retriever(vectorstore)

    documents = retriever.invoke(question)

    print(f"📄 Retrieved {len(documents)} chunks")

    return {
        "documents": documents
    }


# ============================================================
# NODE 3: GRADER
# Uses FAST/LOW LLM
# Filters retrieved chunks by relevance
# ============================================================

def grader_node(state: AgentState):
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("⚠️ No documents retrieved — skipping grading")

        return {
            "filtered_documents": []
        }

    # Use FAST model for simple yes/no grading
    llm = get_fast_llm()

    grade_prompt = ChatPromptTemplate.from_template("""
You are a document relevance grader.

Your task is to determine whether the retrieved document chunk
contains ANY information that could help answer the user's question.

Retrieved chunk:
{document}

User question:
{question}

Rules:
- Answer ONLY with "yes" or "no".
- Answer "yes" if the chunk contains even partial information
  that could help answer the question.
- When uncertain, answer "yes".
- Do not explain your answer.
""")

    # No structured output / tool calling.
    # This avoids Groq tool_use_failed errors.
    grader_chain = grade_prompt | llm | StrOutputParser()

    filtered_docs = []

    for doc in documents:

        try:
            result = grader_chain.invoke({
                "question": question,
                "document": doc.page_content
            })

            binary_score = str(result).strip().lower()

            # Normalize model output
            if binary_score.startswith("yes"):
                binary_score = "yes"
            else:
                binary_score = "no"

        except Exception as e:
            print(f"⚠️ Grader error: {e}")

            # If grader fails, keep the document rather than
            # accidentally throwing away potentially useful context.
            binary_score = "yes"

        if binary_score == "yes":
            filtered_docs.append(doc)

            print(
                f"✅ Relevant: "
                f"{doc.page_content[:60]}..."
            )

        else:
            print(
                f"❌ Not relevant: "
                f"{doc.page_content[:60]}..."
            )

    print(
        f"📊 Grading complete: "
        f"{len(filtered_docs)}/{len(documents)} chunks kept"
    )

    if not filtered_docs:
        return {
            "filtered_documents": []
        }

    return {
        "filtered_documents": filtered_docs
    }


# ============================================================
# NODE 4: GENERATOR
# Uses POWERFUL LLM
# Generates final answer from filtered chunks
# ============================================================

def generator_node(state: AgentState):
    question = state["question"]
    filtered_documents = state.get("filtered_documents", [])

    # Safety net
    if not filtered_documents:

        print(
            "⚠️ generator_node called with "
            "no filtered documents"
        )

        return {
            "answer": "I could not find the answer in the document.",
            "sources": []
        }

    # Use POWERFUL model for final answer
    llm = get_llm()

    context = format_docs(filtered_documents)

    generation_prompt = ChatPromptTemplate.from_template("""
You are a helpful document assistant.

Answer the question ONLY using the provided context.

Rules:
- Use only information from the context.
- Do not make up information.
- If the answer is not present in the context, say:
  "I could not find the answer in the document."
- Always respond in complete sentences.
- Provide sufficient detail when the context supports it.

Context:
{context}

Question:
{question}

Answer:
""")

    generation_chain = (
        generation_prompt
        | llm
        | StrOutputParser()
    )

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

    print(
        f"💬 Generated answer using "
        f"{len(filtered_documents)} source chunk(s)"
    )

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# NODE 5: QUERY REWRITER
# Uses FAST LLM
# Rephrases question when retrieval quality is poor
# ============================================================

def rewriter_node(state: AgentState):
    question = state["question"]
    retry_count = state.get("retry_count", 0) + 1

    print(
        f"🔄 Rewriting query "
        f"(attempt {retry_count})"
    )

    rewriter_prompt = ChatPromptTemplate.from_template("""
You are a query rewriter for document retrieval.

IMPORTANT RULES:

- Keep ALL acronyms exactly as they appear.
- Keep ALL technical terms exactly as they appear.
- Keep ALL proper nouns exactly as they appear.
- Only rephrase the question structure.
- Do NOT expand acronyms.
- Do NOT interpret acronyms.
- Do NOT add information that was not in the original question.
- Return ONLY the rewritten question.

Original question:
{question}

Rewritten question:
""")

    # Use FAST model
    llm = get_fast_llm()

    rewriter_chain = (
        rewriter_prompt
        | llm
        | StrOutputParser()
    )

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
# NODE 6: HALLUCINATION CHECKER
# Uses FAST LLM
# Checks whether final answer is supported by documents
# ============================================================

def hallucination_checker_node(state: AgentState):
    question = state["question"]
    answer = state["answer"]

    filtered_documents = state.get(
        "filtered_documents",
        []
    )

    # No documents means there is nothing to verify.
    if not filtered_documents:

        return {
            "hallucination_status": "yes"
        }

    docs_text = "\n\n".join(
        doc.page_content
        for doc in filtered_documents
    )

    hallucination_prompt = ChatPromptTemplate.from_template("""
You are a strict hallucination checker.

Your task is to determine whether the generated answer
is fully supported by the retrieved documents.

Retrieved Documents:
{documents}

Generated Answer:
{answer}

Rules:

- Answer ONLY with "yes" or "no".
- Answer "yes" if the answer is fully supported by the documents.
- Answer "no" if the answer contains information that is
  not supported by the documents.
- Do not explain your answer.

Answer:
""")

    # Use FAST model.
    # Do NOT use with_structured_output().
    llm = get_fast_llm()

    hallucination_chain = (
        hallucination_prompt
        | llm
        | StrOutputParser()
    )

    try:

        result = hallucination_chain.invoke({
            "documents": docs_text,
            "answer": answer
        })

        score = str(result).strip().lower()

        # Normalize model output
        if score.startswith("yes"):
            score = "yes"
        else:
            score = "no"

    except Exception as e:

        print(
            f"⚠️ Hallucination checker error: {e}"
        )

        # Fail safely.
        score = "yes"

    print(
        f"🧠 Hallucination Check: {score}"
    )

    return {
        "hallucination_status": score
    }