"""Week 7 (Python) - a research assistant built as a LangGraph workflow.

Usage:  python -m brief "How do LangGraph agents differ from LangChain agents?"

Five agents (routing, planning, retrieval, summarization, final answer) are coordinated by the
graph in graph.py. The result is saved as a markdown brief with numbered citations under reports/.
"""

import asyncio
import sys

from brief.agents import create_agents
from brief.graph import create_research_graph
from brief.models import get_cloud_chat_model
from brief.report import save_report_to_disk
from brief.sources import DEFAULT_FETCHERS
from brief.state import initial_state
from brief.tools.weather import open_meteo_weather
from brief.utils import print_section_header

# A whole run makes a dozen or so model calls plus web requests, and free-tier latency swings
# widely. This ceiling exists so a stalled model or network can never leave the terminal hanging
# forever; progress lines are printed throughout, so a long run is never silent.
RUN_TIMEOUT_S = 8 * 60


async def main(topic: str) -> int:
    print_section_header(f"Researching: {topic}")

    graph = create_research_graph(
        agents=create_agents(get_cloud_chat_model()),
        fetchers=DEFAULT_FETCHERS,
        weather=open_meteo_weather,
        save_report=save_report_to_disk,
    )

    try:
        result = await asyncio.wait_for(graph.ainvoke(initial_state(topic)), timeout=RUN_TIMEOUT_S)
    except Exception as error:  # noqa: BLE001
        print(f"\nResearch failed: {str(error) or type(error).__name__}", file=sys.stderr)
        print(
            "\nMost likely causes: the free OpenRouter model is slow, unavailable or was removed from the "
            "free tier (see CLOUD_MODEL in brief/config.py), or the network is down.",
            file=sys.stderr,
        )
        return 1

    print_section_header("Finished", "-")
    print(result["report"])
    return 0


def run() -> None:
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        print('Usage: python -m brief "your research topic"', file=sys.stderr)
        print('Example: python -m brief "How do LangGraph agents differ from LangChain agents?"', file=sys.stderr)
        sys.exit(1)
    sys.exit(asyncio.run(main(topic)))
