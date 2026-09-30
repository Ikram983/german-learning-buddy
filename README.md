# 🇩🇪 German Learning Buddy — Multi-Agent RAG Tutor

A multi-agent, retrieval-augmented German tutor: four specialist agents
(Conversation Partner, Grammar Coach, Correction, Quiz) coordinated by a
LangGraph router, grounded in your own grammar/vocab PDF books via RAG.

## Architecture

```
question --> router (LLM classification)
                |
    +-----------+-----------+-----------+
    v           v           v           v
conversation  grammar   correction    quiz
    |           |           |           |
    +-----------+-----------+-----------+
                v
               END
```

- **Conversation Partner** — free chat practice, no tools
- **Grammar Coach** — RAG over `grammar_book.pdf` + a deterministic verb
  conjugator tool (never guesses a conjugation)
- **Correction** — checks your written German, grounded in the same
  grammar book so it doesn't invent rules
- **Quiz** — generates practice questions, prioritizing topics from your
  `MistakeTracker` history

Each specialist agent is its own small hand-rolled ReAct loop (see
`agents.py`) — the LLM decides whether it needs a tool before answering,
rather than a fixed pipeline order.

```
src/german_buddy/
├── config.py       # env-driven settings
├── ingest.py        # PDF -> cleaned chunks -> Chroma vector store
├── retriever.py      # Chroma collection wrapped as LangChain @tool functions
├── tools.py         # verb conjugator + mistake tracker (deterministic tools)
├── llm.py           # ChatOpenAI factory (DeepSeek or OpenAI)
├── agents.py         # the 4 specialist agents + shared tool-calling loop
└── graph.py          # LangGraph StateGraph wiring router + agents
app.py                # CLI chat loop
tests/                # pytest — chunking/PDF extraction/conjugator/tracker
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your DEEPSEEK_API_KEY
```

Put your two PDFs at:
```
data/raw/grammar_book.pdf
data/raw/vocab_book.pdf
```
(or edit `GRAMMAR_PDF`/`VOCAB_PDF` in `config.py` if you want different
filenames/locations.)

## Build the vector store

```bash
python -m german_buddy.ingest
```

Rerun this any time you change the PDFs or chunking settings.

## Run

```bash
python app.py
```

## Test

```bash
pytest
```

Note: `test_ingest.py` and `test_tools.py` only exercise the pure-Python
logic (PDF extraction, chunking, verb conjugation, mistake tracking) —
they don't need API keys or a built vector store. `agents.py`/`graph.py`
need real API access to test end-to-end since they call the LLM.

## Known rough edges / things to expect on first run

This was written and syntax-checked, and the ingestion/chunking/tool logic
was actually executed and tested — but the LangChain/LangGraph wiring in
`agents.py`/`graph.py` was **not** run end-to-end in the environment this
was written in (no network access to install `langgraph`/`langchain-openai`
there). When you run it for the first time on your machine, expect to hit
at least one real error — read it carefully rather than assuming the code
is wrong; likely candidates:

- A LangGraph API signature mismatch if your installed version differs
  from what this was written against (check `pip show langgraph`)
- `bind_tools` behavior differences between LangChain versions
- Your PDF extraction producing messier text than the test sample — print
  `extract_pdf_pages(...)` output before assuming chunking is broken

Bring me the actual traceback when you hit one — that's the real debugging
practice this project is for.

## Extending

- Wire `correction_agent`'s TODO: parse mistakes out of its response and
  call `mistake_tracker.record_mistake(topic)` so the Quiz agent actually
  learns from your errors over time
- Add a `MemorySaver` checkpointer to the compiled graph for persistent
  multi-session history
- Port `app.py`'s loop into a Streamlit UI (same pattern as your fitness
  coach project)
