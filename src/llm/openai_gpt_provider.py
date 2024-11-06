import logging

import tiktoken
from openai import OpenAI

from llm.conversation_history import ConversationHistory
from llm.large_language_model import LargeLanguageModel


class OpenAiGptProvider(LargeLanguageModel):

    logger = logging.getLogger(__name__)

    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_O = "gpt-4o"

    TOKEN_REMOVAL_CONSTANT = 150

    def __init__(self, openai_api_key: str, model_name: str, max_output_tokens: int = 500):
        """
        Initializes the OpenAIProvider with the given API key, model name, and maximum output tokens.

        Args:
            openai_api_key (str): The API key for authenticating with the OpenAI service.
            model_name (str): The name of the model to use (e.g., "gpt-3.5-turbo").
            max_output_tokens (int, optional): The maximum number of tokens for the output. Defaults to 500.
        """
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=openai_api_key)
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:  # TODO: temporal! no tokenizer for chat gpt 4o
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def get_provider_name(self) -> str:
        return f"OpenAI_{self.model_name}"

    def generate_text(self, conversation_history: ConversationHistory) -> str:
        # Get conversation history
        chat_gpt_prompt = conversation_history.get_chat_gpt_conversation_history()

        # Get response
        response = self.client.chat.completions.create(
            messages=chat_gpt_prompt,
            model=self.model_name
        )
        response_text = response.choices[0].message.content

        return response_text
