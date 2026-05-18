from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.state import AgentState
from agents.researcher import researcher_agent, route_after_research
from agents.writer import writer_agent
from agents.critic import critic_agent, route_after_critique
from output_writer import save_approved_node, save_review_node


def build_graph():
    """
    Build and compile the LangGraph multi-agent workflow.

    Graph structure:
        researcher → (skipped → END) | (has_signals → writer)
        writer → critic
        critic → (approved → save_approved → END)
               | (retry → writer)          same signal, feedback injected
               | (next_signal → writer)    next ranked signal, fresh attempt
               | (exhausted → save_review → END)
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("save_approved", save_approved_node)
    workflow.add_node("save_review", save_review_node)

    workflow.set_entry_point("researcher")

    workflow.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "has_signals": "writer",
            "skipped": END,
        },
    )

    workflow.add_edge("writer", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critique,
        {
            "approved": "save_approved",
            "retry": "writer",
            "next_signal": "writer",
            "exhausted": "save_review",
        },
    )

    workflow.add_edge("save_approved", END)
    workflow.add_edge("save_review", END)

    return workflow.compile(checkpointer=MemorySaver())
