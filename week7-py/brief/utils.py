import time

CYAN = "\033[36m"
RESET = "\033[0m"


def print_section_header(text: str, char: str = "=") -> None:
    border = char * 70
    print(f"\n{border}\n{text}\n{border}")


def since(started_at: float) -> str:
    """Seconds elapsed since `started_at` (a time.monotonic() value), e.g. '4.2s'."""
    return f"{time.monotonic() - started_at:.1f}s"


def log_step(agent: str, message: str) -> None:
    """One line per graph step, so the flowchart is visible while it runs."""
    print(f"{CYAN}[{agent}]{RESET} {message}", flush=True)
