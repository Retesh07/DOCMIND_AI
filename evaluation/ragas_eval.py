import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.vectorstore import get_or_create_vectorstore
from app.graph import build_graph
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configure RAGAS to use Groq instead of OpenAI
# ============================================================
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is not None:
    os.environ["GROQ_API_KEY"] = groq_api_key

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

ragas_llm = LangchainLLMWrapper(groq_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

# ============================================================
# Test dataset — change to match YOUR PDF
# ============================================================

PDF_PATH = "sample.pdf"

test_samples = [
    {
        "question": "Sample question",
        "ground_truth": "Sample answer"
    }
]

# ============================================================
# Run agent on each question
# ============================================================

def collect_results(pdf_path, test_samples):
    print("🔧 Building graph...")
    graph = build_graph()

    print("📄 Loading vectorstore...")
    get_or_create_vectorstore(pdf_path)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for sample in test_samples:
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        print(f"\n❓ Testing: {question}")

        result = graph.invoke({
            "question": question,
            "pdf_path": pdf_path,
            "documents": [],
            "filtered_documents": [],
            "answer": "",
            "sources": [],
            "retry_count": 0,
            "hallucination_status": ""
        })

        answer = result.get("answer", "")
        filtered_docs = result.get("filtered_documents", [])
        context_texts = [doc.page_content for doc in filtered_docs]

        if not context_texts:
            context_texts = ["No relevant context found"]

        print(f"✅ Answer: {answer[:100]}...")

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    return questions, answers, contexts, ground_truths

# ============================================================
# Run RAGAS evaluation
# ============================================================

def run_evaluation():
    questions, answers, contexts, ground_truths = collect_results(
        PDF_PATH, test_samples
    )

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n🔍 Running RAGAS evaluation...")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION RESULTS")
    print("="*50)

    df = results.to_pandas()
    print("\nAvailable metrics:", df.columns.tolist())
    print(df.mean(numeric_only=True))
    print("="*50)

    return results

if __name__ == "__main__":
    run_evaluation()