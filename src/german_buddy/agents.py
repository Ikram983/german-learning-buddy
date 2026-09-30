"""The four specialist agents. Each one is a small hand-rolled ReAct loop:
give the LLM a role-specific system prompt and a narrow set of tools, let
it decide whether it needs a tool before answering, execute any tool calls,
feed the results back, repeat until it answers with plain text.

Writing the loop by hand (instead of a one-line `create_react_agent`) is
deliberate here — it's the part of LangChain worth understanding, and it's
exactly the kind of "walk me through the agent loop" question that comes
up in interviews.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from german_buddy.llm import get_llm
from german_buddy.retriever import search_grammar, search_vocab
from german_buddy.tools import MistakeTracker, conjugate_verb

MAX_TOOL_ITERATIONS = 4


def run_tool_calling_agent(
    system_prompt: str,
    tools: list[BaseTool],
    history: list[BaseMessage],
    user_message: str,
    temperature: float = 0.3,
) -> str:
    """The shared ReAct loop every agent below uses."""
    llm = get_llm(temperature=temperature)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    tools_by_name = {t.name: t for t in tools}

    messages: list[BaseMessage] = [SystemMessage(system_prompt), *history, HumanMessage(user_message)]

    for _ in range(MAX_TOOL_ITERATIONS):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return response.content

        for call in response.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Sorry, I couldn't settle on an answer in time — try rephrasing the question."


# --------------------------------------------------------------------------- #
# 1. Conversation Partner — free chat practice, no RAG needed for small talk
# --------------------------------------------------------------------------- #

CONVERSATION_PROMPT = """\
You are a friendly, encouraging German conversation partner. Chat naturally
in German at a level suited to a learner (assume roughly A2-B1 unless told
otherwise). Keep responses short (2-4 sentences). If the learner writes in
English, gently nudge them to try German, but don't refuse to help.
You have no tools — just have a natural conversation.
"""


def conversation_agent(history: list[BaseMessage], user_message: str) -> str:
    return run_tool_calling_agent(CONVERSATION_PROMPT, tools=[], history=history, user_message=user_message, temperature=0.7)


# --------------------------------------------------------------------------- #
# 2. Grammar Coach — RAG-grounded explanations
# --------------------------------------------------------------------------- #

GRAMMAR_PROMPT = """\
You are a precise German grammar coach. When asked about a grammar rule
(cases, articles, verb tense, word order, etc.), ALWAYS call the
search_grammar tool first to check the reference book before answering —
do not rely on what you already "know" about German grammar, since you
might misstate a rule. Base your explanation on what the tool returns, and
mention the page number so the learner can look it up themselves.
If asked to conjugate a specific verb, use the conjugate_verb tool instead
of guessing — conjugation should never be a guess.
"""


def grammar_agent(history: list[BaseMessage], user_message: str) -> str:
    return run_tool_calling_agent(
        GRAMMAR_PROMPT, tools=[search_grammar, conjugate_verb], history=history, user_message=user_message
    )


# --------------------------------------------------------------------------- #
# 3. Correction Agent — checks learner's German, grounded in the same book
# --------------------------------------------------------------------------- #

CORRECTION_PROMPT = """\
You are a German writing corrector. The learner will give you a sentence
or short paragraph they wrote in German. Your job:
1. Call search_grammar to verify any rule you're unsure about before
   flagging something as wrong — don't invent a rule.
2. List each mistake, the correction, and a one-line reason.
3. If there are no mistakes, say so plainly and briefly.
Be encouraging but accurate — do not soften an actual error into "it's fine".
"""


def correction_agent(history: list[BaseMessage], user_message: str, mistake_tracker: MistakeTracker | None = None) -> str:
    result = run_tool_calling_agent(CORRECTION_PROMPT, tools=[search_grammar], history=history, user_message=user_message)
    # A stretch task for you: parse `result` for mistake topics and call
    # mistake_tracker.record_mistake(topic) here so the Quiz agent can use
    # them later. Left unimplemented on purpose — this is a good exercise
    # in extracting structured info from an LLM's free-text response.
    return result


# --------------------------------------------------------------------------- #
# 4. Quiz Agent — generates practice questions, prioritizing past mistakes
# --------------------------------------------------------------------------- #

QUIZ_PROMPT = """\
You are a German quiz generator. Create ONE multiple-choice or fill-in-
the-blank question testing the learner on the topic(s) you're given.
Keep it short. After showing the question, wait for the learner's answer
in a follow-up turn — do not reveal the correct answer yet.
"""


def quiz_agent(history: list[BaseMessage], user_message: str, mistake_tracker: MistakeTracker) -> str:
    top = mistake_tracker.top_mistakes(n=3)
    topics_hint = ", ".join(t for t, _ in top) if top else "general A2-level grammar and vocab"
    prompt_with_context = f"{user_message}\n\n(Focus on these topics if relevant: {topics_hint})"
    return run_tool_calling_agent(QUIZ_PROMPT, tools=[search_grammar, search_vocab], history=history, user_message=prompt_with_context)
