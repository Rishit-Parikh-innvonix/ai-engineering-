import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Prefer a .env inside week2/ (matches week1's convention: copy .env.example
# to .env here). Fall back to the repo-root .env so this also works out of
# the box in this repo, where the key currently lives one level up.
_local_env = os.path.join(BASE_DIR, ".env")
_root_env = os.path.join(REPO_ROOT, ".env")
load_dotenv(dotenv_path=_local_env if os.path.exists(_local_env) else _root_env, override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free OpenRouter models, chosen the same way week1 chose them (see week1/README.md,
# section "Which models are free right now"). Swap the strings here if any get pulled.
STREAM_MODEL = "openai/gpt-oss-20b:free"
EXTRACTION_MODEL = "openai/gpt-oss-20b:free"
BENCHMARK_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
TOOL_DEMO_MODEL = "openai/gpt-oss-20b:free"

EXTRACTION_MAX_RETRIES = 3

# USD per 1M tokens, as (input, output). Every model above is a free OpenRouter
# model, so real cost is $0 - this dict exists to demonstrate the technique.
# Fill in real numbers (from https://openrouter.ai/models) if you switch to a paid model.
MODEL_PRICING_USD_PER_MILLION = {}

# Exercise 3: kept short and with modest concurrency on purpose - OpenRouter's
# free tier caps at 20 requests/minute (see week1/README.md), and this script
# runs the same prompts twice (once sync, once async) in a single process.
BENCHMARK_PROMPTS = [
    "In two sentences, explain what an API is.",
    "In two sentences, explain what a database index does.",
    "In two sentences, explain the difference between TCP and UDP.",
    "In two sentences, explain what caching is used for.",
    "In two sentences, explain what a load balancer does.",
    "In two sentences, explain what a race condition is.",
]
ASYNC_CONCURRENCY = 3

# Exercise 2 demo input - realistic customer support emails with varying priority/type.
SAMPLE_SUPPORT_EMAILS = [
    (
        "Subject: Can't log in since this morning\n\n"
        "Hi team, this is Priya Shah (priya.shah@example.com). Since about 9am today I keep "
        "getting 'invalid credentials' even though I'm sure my password is right, and I have a "
        "client demo in an hour. Please help ASAP, this is blocking my whole team."
    ),
    (
        "Subject: Question about my invoice\n\n"
        "Hello, I'm Daniel Cho (daniel.cho99@example.com). I noticed I was charged twice on my "
        "last invoice. Could someone take a look when they get a chance? Not urgent, just want "
        "it corrected eventually."
    ),
    (
        "Subject: Feature idea\n\n"
        "Hey, this is Marta (marta.k@example.com). It would be great if the dashboard supported "
        "dark mode. No rush at all, just a suggestion for a future release."
    ),
]
