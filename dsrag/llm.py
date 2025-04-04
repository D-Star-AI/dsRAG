from abc import ABC, abstractmethod
import os
from dsrag.utils.imports import openai, anthropic, ollama
import google.generativeai as genai


class LLM(ABC):
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def to_dict(self):
        return {
            'subclass_name': self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, config):
        subclass_name = config.pop('subclass_name', None)  # Remove subclass_name from config
        subclass = cls.subclasses.get(subclass_name)
        if subclass:
            return subclass(**config)  # Pass the modified config without subclass_name
        else:
            raise ValueError(f"Unknown subclass: {subclass_name}")

    @abstractmethod
    def make_llm_call(self, chat_messages: list[dict]) -> str:
        """
        Takes in chat_messages (OpenAI format) and returns the response from the LLM as a string.
        """
        pass

class OpenAIChatAPI(LLM):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        base_url = os.environ.get("DSRAG_OPENAI_BASE_URL", None)
        if base_url is not None:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base_url)
        else:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        llm_output = response.choices[0].message.content.strip()
        return llm_output
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        })
        return base_dict

class AnthropicChatAPI(LLM):
    def __init__(self, model: str = "claude-3-haiku-20240307", temperature: float = 0.2, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        base_url = os.environ.get("DSRAG_ANTHROPIC_BASE_URL", None)
        if base_url is not None:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url)
        else:
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        system_message = ""
        num_system_messages = 0
        normal_chat_messages = []
        for message in chat_messages:
            if message["role"] == "system":
                if num_system_messages > 0:
                    raise ValueError("ERROR: more than one system message detected")
                system_message = message["content"]
                num_system_messages += 1
            else:
                normal_chat_messages.append(message)

        message = client.messages.create(
            system=system_message,
            messages=normal_chat_messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return message.content[0].text
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        })
        return base_dict

class OllamaAPI(LLM):
    def __init__(
        self, model: str = "llama3", temperature: float = 0.2, max_tokens: int = 1000, client: "ollama.Client" = None
    ):
        self.client = client or ollama.Client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client.pull(self.model)

    def make_llm_call(self, chat_messages: list[dict]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=chat_messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return response["message"]["content"].strip()

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        )
        return base_dict

class GeminiAPI(LLM):
    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.2, max_tokens: int = 1000):
        self.model_name = model # Renamed to avoid conflict with genai.GenerativeModel instance
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Configure API key
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        # It's generally better practice to configure the API key once globally,
        # but doing it here for simplicity within the class structure.
        try:
             genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        except Exception as e:
             # Avoid crashing if configure is called multiple times with the same key,
             # although the genai library might handle this internally.
             # Check the specific exception type if needed.
             print(f"Warning: Could not configure genai, possibly already configured: {e}")
             pass
        # Pre-initialize the model object
        self.model = genai.GenerativeModel(self.model_name)

    def _convert_messages(self, chat_messages: list[dict]) -> list[dict]:
        """
        Converts OpenAI format messages to Google format.
        Prepends system instruction to the first message's content if present.
        """
        google_messages = []
        system_instruction = None
        temp_messages = [] # Temporary list to hold non-system messages before prepending

        for message in chat_messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if system_instruction is not None:
                    # Combine multiple system messages if necessary
                    system_instruction += "\n" + content
                else:
                    system_instruction = content
            elif role == "user":
                temp_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                # Google uses 'model' role for assistant messages
                temp_messages.append({"role": "model", "parts": [content]})
            else:
                # Raise error for unsupported roles
                raise ValueError(f"Unsupported role encountered: {role}")

        # Prepend system instruction to the first message if it exists
        if system_instruction and temp_messages:
            # Ensure the first part is text content before prepending
            if isinstance(temp_messages[0]["parts"][0], str):
                 temp_messages[0]["parts"][0] = f"System instruction:\n{system_instruction}\n\nUser query:\n{temp_messages[0]['parts'][0]}"
            else:
                 # If the first part isn't a simple string (e.g., image), insert text part at the beginning
                 temp_messages[0]["parts"].insert(0, f"System instruction:\n{system_instruction}\n\nUser query following:")
                 
            google_messages = temp_messages
        elif not system_instruction:
             google_messages = temp_messages
        # Handle edge case: only system message provided (though unlikely for generate_content)
        elif system_instruction and not temp_messages:
             # Cannot directly send only system instruction in contents.
             # Maybe create a dummy user message? Or raise error?
             # For now, let's create a user message containing the system instruction.
             google_messages.append({"role": "user", "parts": [f"System instruction:\n{system_instruction}"]})


        return google_messages


    def make_llm_call(self, chat_messages: list[dict]) -> str:
        # Convert messages, embedding system instruction within the list
        google_messages = self._convert_messages(chat_messages)

        generation_config = genai.GenerationConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            # You can add other config parameters like top_p, top_k if needed
        )

        try:
            # Call generate_content without the system_instruction kwarg
            response = self.model.generate_content(
                contents=google_messages,
                generation_config=generation_config,
            )
            # Accessing the response text using response.text is generally robust
            # Check response.prompt_feedback for safety blocks
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason = response.prompt_feedback.block_reason
                 print(f"Warning: Gemini API call blocked due to: {block_reason}")
                 # Optionally return a specific message or raise a custom error
                 return f"[ERROR: Content blocked due to {block_reason}]"

            # Check if the response has text content
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                 # Handle cases where response might be empty or structured differently
                 # Check candidates if response.text is not available
                 if response.candidates and response.candidates[0].content.parts:
                      return "".join(part.text for part in response.candidates[0].content.parts).strip()
                 else:
                      finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                      print(f"Warning: Gemini API call returned no content. Finish Reason: {finish_reason}")
                      return "" # Return empty string for empty but valid responses

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            # Re-raise the exception to signal failure upstream
            raise

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict.update({
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        })
        return base_dict