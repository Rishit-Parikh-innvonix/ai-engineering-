import logging
import time

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

import config
import models
import prompts
import schemas

logger = logging.getLogger("week2.extraction")

parser = PydanticOutputParser(pydantic_object=schemas.SupportTicket)


async def extract_support_ticket(email_text, request_id="-", max_retries=None):
    """Reject-and-retry loop for Exercise 2. Each attempt: call the model in JSON mode,
    then run the raw text through PydanticOutputParser, which both parses the JSON and
    validates it against SupportTicket's field types/enums. A malformed reply (bad JSON,
    a priority value outside the allowed enum, a badly-formed email address, ...) raises
    instead of silently passing through, and we retry rather than return garbage."""
    max_retries = max_retries or config.EXTRACTION_MAX_RETRIES
    chat = models.get_json_mode_chat_model(config.EXTRACTION_MODEL)
    messages = prompts.extraction_prompt_template.format_messages(
        format_instructions=parser.get_format_instructions(),
        email_text=email_text,
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        start = time.perf_counter()
        try:
            response = await models.ainvoke_with_retry(chat, messages)
            ticket = parser.parse(response.content)
            elapsed = round(time.perf_counter() - start, 2)
            usage = models.extract_token_usage(response)
            logger.info(
                f"[{request_id}] extraction attempt {attempt}/{max_retries} succeeded in {elapsed}s "
                f"(tokens={usage})"
            )
            return {"ticket": ticket.model_dump(), "attempts": attempt, "error": None}
        except (OutputParserException, ValidationError) as error:
            last_error = f"malformed model output: {error}"
            logger.warning(
                f"[{request_id}] extraction attempt {attempt}/{max_retries} rejected - {str(error)[:200]}"
            )
        except Exception as error:
            last_error = str(error)
            logger.error(f"[{request_id}] extraction attempt {attempt}/{max_retries} failed - {last_error[:200]}")

    return {
        "ticket": None,
        "attempts": max_retries,
        "error": f"could not extract a valid ticket after {max_retries} attempts. Last error: {last_error}",
    }
