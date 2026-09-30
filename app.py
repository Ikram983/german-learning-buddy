"""Simple CLI chat loop for the German Learning Buddy.

Run with:  python app.py

This is deliberately a plain terminal loop, not Streamlit — get the
agent/graph logic working and debuggable first, then port to a UI once
it's solid. (Your fitness-coach project already has a working Streamlit
pattern you can copy once this works.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from german_buddy.graph import build_graph, run_turn  # noqa: E402


def main() -> None:
    print("German Learning Buddy — type 'quit' to exit.\n")
    app = build_graph()
    history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, history = run_turn(app, question, history)
        print(f"Buddy: {answer}\n")


if __name__ == "__main__":
    main()
