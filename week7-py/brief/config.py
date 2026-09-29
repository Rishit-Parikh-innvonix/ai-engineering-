import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../week7-py
REPO_ROOT = PROJECT_ROOT.parent

# Prefer a .env inside week7-py/, then the repo-root .env, then the TypeScript week7's .env
# (where the key already lives if you ran that version).
_CANDIDATES = [PROJECT_ROOT / ".env", REPO_ROOT / ".env", REPO_ROOT / "week7" / ".env"]
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
