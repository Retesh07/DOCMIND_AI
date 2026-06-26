from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes import (
    retriever_node,
    grader_node,
    generator_node,
    rewriter_node,
    hallucination_checker_node
)

MAX_RETRIES = 3


# ============================================================
# ROUTING FUNCTIONS
# ============================================================


def route_after_grading(state: AgentState) -> str:

    filtered_documents = state.get(
        "filtered_documents",
        []
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    if not filtered_documents:

        if retry_count >= MAX_RETRIES:
            print("🚦 Max retries reached → END")
            return "max_retries"

        print(
            f"🚦 No relevant docs → rewrite "
            f"(attempt {retry_count + 1})"
        )

        return "rewrite"

    print("🚦 Relevant docs found → generator")
    return "generate"


def route_after_hallucination(state: AgentState) -> str:

    status = state.get(
        "hallucination_status",
        "no"
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    if status == "yes":
        print("✅ Answer grounded → END")
        return "end"

    if retry_count >= MAX_RETRIES:
        print("⚠️ Hallucination retries exceeded → END")
        return "end"

    print("🔄 Hallucinated answer → regenerate")
    return "generator"


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes — no router
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("hallucination_checker", hallucination_checker_node)

    # Entry point — directly to retriever
    workflow.set_entry_point("retriever")

    # Retriever → grader
    workflow.add_edge("retriever", "grader")

    # Grader → generator, rewriter, or END
    workflow.add_conditional_edges(
        "grader",
        route_after_grading,
        {
            "generate": "generator",
            "rewrite": "rewriter",
            "max_retries": END,
        }
    )

    # Rewriter → retriever
    workflow.add_edge("rewriter", "retriever")

    # Generator → hallucination checker
    workflow.add_edge("generator", "hallucination_checker")

    # Hallucination checker → END or regenerate
    workflow.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination,
        {
            "end": END,
            "generator": "rewriter",
        }
    )

    return workflow.compile()