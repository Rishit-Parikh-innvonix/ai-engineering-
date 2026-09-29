from datetime import datetime, timezone

from brief.config import REPORTS_DIR
from brief.logic import slugify


async def save_report_to_disk(topic: str, markdown: str) -> str:
    """A timestamp in the name means running the same topic twice never overwrites the first brief."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{slugify(topic)}-{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)
