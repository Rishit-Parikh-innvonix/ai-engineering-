import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../week9
REPO_ROOT = PROJECT_ROOT.parent

# Prefer a .env inside week9/, then the repo-root .env, then week8's (where the key already
# lives if you ran that one) - same fallback chain week7-py/week8 use.
_CANDIDATES = [PROJECT_ROOT / ".env", REPO_ROOT / ".env", REPO_ROOT / "week8" / ".env"]
ENV_PATH = next((path for path in _CANDIDATES if path.exists()), _CANDIDATES[0])
load_dotenv(ENV_PATH, override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free OpenRouter models come and go (see week7/src/config.ts and week8/brief/config.py for the
# history). Needs reliable tool/function calling, since the client binds the catalog's MCP tools
# to it. If a run fails with a model error, this line is the only place to change.
CLOUD_MODEL = "dots-studio/dots-3-note-preview:free"

PRODUCTS_FILE = PROJECT_ROOT / "data" / "products.json"

# The MCP server script the client spawns as a subprocess (stdio transport, same pattern as
# week6's TypeScript MultiServerMCPClient).
SERVER_SCRIPT = PROJECT_ROOT / "catalog_mcp" / "server.py"


def get_api_key() -> str:
    """Checked when the model is created, not at import, so tests run with no key."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"OPENROUTER_API_KEY is not set. Add it to {ENV_PATH} (see .env.example).")
    return key
