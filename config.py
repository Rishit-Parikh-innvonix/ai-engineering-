import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

COMPARISON_MODELS = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "poolside/laguna-s-2.1:free",
]

EXPERIMENT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

JUDGE_MODEL = "openai/gpt-oss-20b:free"

COMPARISON_PROMPTS = [
    "Explain AI Engineering to a complete beginner in 3-4 sentences.",
    "Explain how Transformers, the neural network architecture behind modern AI, work in simple language for a beginner.",
    "Write Python code that reverses a list. Show only the code and a one-line explanation.",
    "A farmer has 17 sheep, and all but 9 die. How many sheep does the farmer have left? Explain your reasoning briefly.",
    "Summarize the following paragraph in 2 sentences: 'Artificial intelligence engineering is the discipline of designing, building, and deploying AI-powered applications. It combines software engineering practices with machine learning concepts, focusing on integrating large language models, prompt design, and system architecture to build reliable, real-world products rather than researching new models from scratch.'",
]

TEMPERATURE_VALUES = [0, 0.2, 0.5, 0.8, 1.0]
TOP_P_VALUES = [0.2, 0.5, 0.8, 1.0]
MAX_TOKENS_VALUES = [50, 100, 200, 500]

CREATIVITY_PROMPT = "Write a short story, a few sentences long, about a robot learning to paint."
LENGTH_PROMPT = "Explain how neural networks work."

HALLUCINATION_PROMPTS = [
    "Describe the Kingdom of Mars founded in 1850.",
    "Explain the programming language HyperPython++ released in 2032.",
    "Who won the FIFA World Cup in 2050?",
    "What happened in tomorrow's world news?",
    "What is today's weather?",
    "What happened in the latest global event?",
]

REQUEST_DELAY_SECONDS = 3
