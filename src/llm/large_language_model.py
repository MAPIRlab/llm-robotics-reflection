
import json
import logging
from abc import ABC, abstractmethod
from typing import Tuple

from llm.conversation_history import ConversationHistory


class LargeLanguageModel(ABC):

    JSON_MAX_ATTEMPTS = 10

    logger = logging.getLogger(__name__)

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Abstract method to get the name of the LLM service provider.

        Returns:
            str: The name of the LLM service provider.
        """
        pass

    @abstractmethod
    def generate_text(self, conversation_history: ConversationHistory) -> Tuple[str, float]:
        """
        Generates text based on the provided conversation history.

        Args:
            conversation_history (ConversationHistory): The conversation history
                that the LLM will use as context to generate the text.

        Returns:
            Tuple[str, float]: A tuple containing the generated text and the
                cost associated with the request.
            TODO: update
        """
        pass

    def _clean_response(self, text: str) -> str:
        """
        Extract the JSON-like portion from the model's response by finding the text
        between the first occurrence of "{" or "[" and the last occurrence of "}" or "]".

        Args:
            text (str): The text containing the model's response.

        Returns:
            str: The extracted text that is likely in JSON format.
        """
        # Find indices for JSON object
        obj_start_index = text.find("{")
        obj_end_index = text.rfind("}") + 1

        # Find indices for JSON array
        array_start_index = text.find("[")
        array_end_index = text.rfind("]") + 1

        # Determine which start index is valid and which comes first
        if obj_start_index == -1 or (array_start_index != -1 and array_start_index < obj_start_index):
            start_index = array_start_index
            end_index = array_end_index
        else:
            start_index = obj_start_index
            end_index = obj_end_index

        if start_index != -1 and end_index != -1 and start_index < end_index:
            return text[start_index:end_index]
        else:
            return ""

    def generate_json(self, conversation_history: ConversationHistory) -> Tuple[str, float]:
        """
        Generates a JSON-like response by repeatedly attempting to generate text
        from the conversation history and parsing it as JSON.

        Args:
            conversation_history (ConversationHistory): The conversation history
                that the LLM will use as context to generate the text.

        Returns:
            Tuple[str, float]: A tuple containing the valid JSON string (or "{}" if
                no valid response was found) and the total cost associated with the requests.
                # TODO: update
        """
        attempt = 1
        while attempt <= self.JSON_MAX_ATTEMPTS:
            try:
                response = self.generate_text(conversation_history)
                # print(response)

                # Clean response
                response = self._clean_response(response)
                # Try to parse response
                json.loads(response)

                return response  # Return the valid JSON response

            except json.decoder.JSONDecodeError as e:
                self.logger.info(f"Error generating JSON on attempt {
                    attempt}: {str(e)}")
                self.logger.info("WARNING: wrong response: " + response)

            attempt += 1  # Increment attempt counter

        self.logger.info(
            "Couldn't get a valid JSON response, max attempts exceeded")
        return "{}"
