"""Wires the router + four agents into a LangGraph StateGraph.

Shape:

                     +-------------+
        question --> |   router    |  (one-shot classification, not ReAct)
                     +------+------+
                            |
        +---------+---------+---------+---------+
        v         v                   v         v
  conversation  grammar          correction    quiz
        \\         |                   |         /
         +---------+--------+---------+--------+
                            v
                          END

Each of the four agent nodes is its own little ReAct loop (see agents.py) —
this graph only handles routing and shared state, not the tool-calling
logic itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from german_buddy.agents import conversation_agent, correction_agent, grammar_agent, quiz_agent
from german_buddy.llm import get_llm
from german_buddy.tools import MistakeTracker

MISTAKES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "mistakes.json"
_mistake_tracker = MistakeTracker(MISTAKES_PATH)

Route = Literal["conversation", "grammar", "correction", "quiz"]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    route: Route
    answer: str


ROUTER_PROMPT = """\
Classify the learner's message into exactly one category. Reply with only
the category word, nothing else.

Categories:
- conversation: casual chat / free practice, not asking about a rule
- grammar: asking to explain a grammar rule, article, case, or asking to
  conjugate a verb
- correction: asking you to check/correct something they wrote in German
- quiz: asking to be quizzed or tested

Message: {message}
"""


def router_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0)
    raw = llm.invoke(ROUTER_PROMPT.format(message=state["question"])).content.strip().lower()
    route: Route = raw if raw in ("conversation", "grammar", "correction", "quiz") else "conversation"
    return {"route": route}


def _history_without_current(state: AgentState) -> list[BaseMessage]:
    # `messages` already has the new HumanMessage appended (see run_turn
    # below) before the graph runs, so agent nodes get history minus the
    # very last item, which they re-add via `user_message`.
    return state["messages"][:-1]


def conversation_node(state: AgentState) -> dict:
    answer = conversation_agent(_history_without_current(state), state["question"])
    return {"answer": answer, "messages": [AIMessage(answer)]}


def grammar_node(state: AgentState) -> dict:
    answer = grammar_agent(_history_without_current(state), state["question"])
    return {"answer": answer, "messages": [AIMessage(answer)]}


def correction_node(state: AgentState) -> dict:
    answer = correction_agent(_history_without_current(state), state["question"], _mistake_tracker)
    return {"answer": answer, "messages": [AIMessage(answer)]}


def quiz_node(state: AgentState) -> dict:
    answer = quiz_agent(_history_without_current(state), state["question"], _mistake_tracker)
    return {"answer": answer, "messages": [AIMessage(answer)]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("grammar", grammar_node)
    graph.add_node("correction", correction_node)
    graph.add_node("quiz", quiz_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "conversation": "conversation",
            "grammar": "grammar",
            "correction": "correction",
            "quiz": "quiz",
        },
    )
    for node in ("conversation", "grammar", "correction", "quiz"):
        graph.add_edge(node, END)

    return graph.compile()


def run_turn(app, question: str, history: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
    """Run one turn through the graph, given the running conversation
    history, and return (answer, updated_history)."""
    state_in: AgentState = {
        "messages": [*history, HumanMessage(question)],
        "question": question,
        "route": "conversation",  # placeholder, router_node overwrites it
        "answer": "",
    }
    result = app.invoke(state_in)
    return result["answer"], result["messages"]
