import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../week10
REPO_ROOT = PROJECT_ROOT.parent

# Prefer a .env inside week10/, then the repo-root .env, then week8's .env (where the key already
# lives if you ran that version).
_CANDIDATES = [PROJECT_ROOT / ".env", REPO_ROOT / ".env", REPO_ROOT / "week8" / ".env"]
ENV_PATH = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])
load_dotenv(ENV_PATH, override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free OpenRouter models come and go (see week7/src/config.ts for the history and the comparison
# of the free tool-calling models). This model needs reliable tool/function calling, since the
# planner and retrieval agents return structured output. If a run fails with a model error,
# this line is the only place to change.
CLOUD_MODEL = "dots-studio/dots-3-note-preview:free"

# Where finished research briefs are saved.
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_api_key() -> str:
    """Checked when the model is created, not at import, so tests run with no key."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"OPENROUTER_API_KEY is not set. Add it to {ENV_PATH} (see .env.example).")
    return key

# Where checkpoints (saved run state) live. One SQLite file, no server.
CHECKPOINT_DB = PROJECT_ROOT / "checkpoints.sqlite"

# Where per-run trace files (brief/tracing.py) are written.
TRACES_DIR = PROJECT_ROOT / "traces"

# $/1000 tokens, used only to compute tracing.py's estimated_cost_usd. 0 because CLOUD_MODEL is
# always a :free OpenRouter model (see the project rule in README.md) - kept as a named constant,
# not a hardcoded 0 in tracing.py, so the one line to change if that ever stops being true is here.
PRICE_PER_1K_TOKENS = 0.0
