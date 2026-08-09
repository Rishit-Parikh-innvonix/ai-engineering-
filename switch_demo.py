import config
import models

DEMO_MODEL = config.COMPARISON_MODELS[3]


def ask_demo_question(model_name):
    chat = models.get_chat_model(model_name, temperature=0.5)
    response = chat.invoke("hi what fuck are you doing?")
    return response.content


def run_demo():
    print(f"\nAsking model: {DEMO_MODEL}")
    print(ask_demo_question(DEMO_MODEL))

    other_model = config.COMPARISON_MODELS[1]
    print(f"\nNow changing only the model name to: {other_model}")
    print(ask_demo_question(other_model))

    print("\nask_demo_question() and get_chat_model() never changed.")
    print("Only the model name string changed - that is LangChain's model abstraction in action.")


if __name__ == "__main__":
    run_demo()
