import os
from openai import OpenAI
from anthropic import Anthropic

def make_llm_call(chat_messages: list[dict], model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.2, max_tokens: int = 2000, additional_stop_sequences: list[str] = [], response_starter_text: str = "") -> str:
    openai_models = ["gpt-4-0125-preview", "gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"]
    anthropic_models = ["claude-2", "claude-instant-1", "claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]

    if model_name in openai_models:
        return openai_api_call(chat_messages, model_name, temperature, max_tokens, additional_stop_sequences)
    elif model_name in anthropic_models:
        return anthropic_api_call(chat_messages, model_name, temperature, max_tokens, additional_stop_sequences, response_starter_text)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

def openai_api_call(chat_messages: list[dict], model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0.2, max_tokens: int = 1000, additional_stop_sequences: list[str] = []) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    max_tokens = int(max_tokens)
    temperature = float(temperature)

    # call the OpenAI API
    response = client.chat.completions.create(model=model_name,
        messages=chat_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=additional_stop_sequences,
    )
    llm_output = response.choices[0].message.content.strip()
    
    return llm_output

def anthropic_api_call(chat_messages: list[dict], model_name: str = "claude-3-sonnet-20240229", temperature: float = 0.2, max_tokens: int = 1000, additional_stop_sequences: list[str] = [], response_starter_text: str = "") -> str:
    max_tokens = int(max_tokens)
    temperature = float(temperature)

    # extract the system message from the chat_messages list
    system_message = ""
    num_system_messages = 0
    normal_chat_messages = [] # chat messages that are not system messages
    for message in chat_messages:
        if message["role"] == "system":
            if num_system_messages > 0:
                raise ValueError("ERROR: more than one system message detected")
            system_message = message["content"]
            num_system_messages += 1
        else:
            normal_chat_messages.append(message)

    if response_starter_text != "":
        normal_chat_messages.append({"role": "assistant", "content": response_starter_text})
    
    # call the Anthropic API
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        system=system_message,
        messages=normal_chat_messages,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=additional_stop_sequences,
    )

    return message.content[0].text