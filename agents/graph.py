"""Compatibility entry points for the framework-free research runtime."""
from agents.runtime import ResearchRuntime, initial_state


def build_graph():
    return ResearchRuntime()


def run_research(ticker: str, run_id=None, on_event=None, cancelled=None):
    return ResearchRuntime().invoke(
        initial_state(ticker, run_id), on_event=on_event, cancelled=cancelled,
    )
