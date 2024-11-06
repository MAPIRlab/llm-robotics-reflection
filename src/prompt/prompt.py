

from abc import ABC, abstractmethod


class Prompt(ABC):

    def __init__(self, **prompt_data_dict):
        """
        Initializes the Prompt instance with a dictionary of prompt data.

        Args:
            **prompt_data_dict: Arbitrary keyword arguments representing key-value pairs
                that are used for dynamic prompt generation or replacement.
        """
        self.prompt_data_dict = prompt_data_dict

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Abstract method to retrieve the system prompt text.

        Returns:
            str: The system prompt that will be used as the base template for further processing.
        """
        pass

    @abstractmethod
    def global_replace(self, prompt_text: str) -> str:
        """
        Abstract method to perform global replacements in the prompt text.

        Args:
            prompt_text (str): The initial prompt text in which replacements need to be made.

        Returns:
            str: The prompt text after performing global replacements.
        """
        pass

    def replace_prompt_data_dict(self, prompt_data_dict: dict, prompt_text: str):
        """
        Replaces placeholders in the prompt text using the provided dictionary.

        The method substitutes placeholders in the format "{{key}}" in the prompt text with
        their corresponding values from the `prompt_data_dict`.

        Args:
            prompt_data_dict (dict): A dictionary containing key-value pairs where the key is
                the placeholder (without curly braces) and the value is the text to replace it with.
            prompt_text (str): The text containing placeholders that need to be replaced.

        Returns:
            str: The prompt text with all the placeholders replaced with the corresponding values.
        """
        # Substitute keys in prompt_data_dict
        for key in prompt_data_dict:
            prompt_text = prompt_text.replace(
                "{{"+key+"}}", prompt_data_dict[key])
        return prompt_text

    def get_prompt_text(self) -> str:
        """
        Generates the final prompt text by applying global replacements to the system prompt.

        This method calls `get_system_prompt` to retrieve the base system prompt and then applies
        the `global_replace` method to perform any necessary substitutions.

        Returns:
            str: The final prompt text after all replacements.
        """
        return self.global_replace(self.get_system_prompt())
