"""Runs Exercise 2's extraction logic directly against the sample emails in config.py,
without needing the FastAPI server running - useful for a quick demo or for testing the
reject/retry logic in isolation. Run: python extraction_demo.py
"""

import asyncio

import config
import extraction
import utils


async def main():
    for email in config.SAMPLE_SUPPORT_EMAILS:
        utils.print_panel("Input email", email)
        result = await extraction.extract_support_ticket(email)
        if result["error"]:
            utils.print_panel("Extraction FAILED", result["error"])
        else:
            fields = "\n".join(f"{key}: {value}" for key, value in result["ticket"].items())
            utils.print_panel(f"Extracted ticket (succeeded on attempt {result['attempts']})", fields)


if __name__ == "__main__":
    asyncio.run(main())
