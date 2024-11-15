
from vertexai.generative_models import Content, Part

from utils.dict_utils import search_dict_by_key_value

ROLE_SYSTEM = "system"
ROLE_ASSISTANT = "assistant"
ROLE_USER = "user"
ROLE_MODEL = "model"

KEY_ROLE = "role"
KEY_CONTENT = "content"


class ConversationHistory:

    @staticmethod
    def system_message(system_text: str):
        """
        Creates a system message with the specified content.

        Args:
            system_text (str): The system message text to be included.

        Returns:
            dict: A dictionary representing the system message with 'role' set to 'system' 
            and the 'content' set to the provided system_text.
        """
        return {KEY_ROLE: ROLE_SYSTEM, KEY_CONTENT: system_text}

    @staticmethod
    def assistant_message(assistant_text: str):
        """
        Creates an assistant message with the specified content.

        Args:
            assistant_text (str): The assistant message text to be included.

        Returns:
            dict: A dictionary representing the assistant message with 'role' set to 'assistant' 
            and the 'content' set to the provided assistant_text.
        """
        return {KEY_ROLE: ROLE_ASSISTANT, KEY_CONTENT: assistant_text}

    @staticmethod
    def user_message(user_text: str):
        """
        Creates a user message with the specified content.

        Args:
            user_text (str): The user message text to be included.

        Returns:
            dict: A dictionary representing the user message with 'role' set to 'user' 
            and the 'content' set to the provided user_text.
        """
        return {KEY_ROLE: ROLE_USER, KEY_CONTENT: user_text}

    def __init__(self):
        """
        Initializes a ConversationHistory instance to track the history of messages exchanged
        in a conversational context.

        Args:
            conversation_history_list (list, optional): A list containing the initial conversation
                history. Defaults to an empty list.
        """
        self.conversation_history_list = []

    def __str__(self):
        """
        Returns a string representation of the conversation history.

        The representation includes the role and content of each message in the history,
        formatted in a human-readable way.

        Returns:
            str: A formatted string representation of the conversation history.
        """
        formatted_messages = []
        for message in self.conversation_history_list:
            formatted_messages.append(f"{message[KEY_ROLE].capitalize()}: {
                                      message[KEY_CONTENT].replace("\n", "").replace("\t", "")}...")
        return "\n".join(formatted_messages)

    def append_system_message(self, system_text: str):
        """
        Appends a system message to the conversation history.

        Args:
            system_text (str): The system message text to be added to the conversation history.
        """
        self.conversation_history_list.append(
            ConversationHistory.system_message(system_text))

    def append_assistant_message(self, assistant_text: str):
        """
        Appends an assistant message to the conversation history.

        Args:
            assistant_text (str): The assistant's message text to be added to the conversation history.
        """
        self.conversation_history_list.append(
            ConversationHistory.assistant_message(assistant_text))

    def append_user_message(self, user_text: str):
        """
        Appends a user message to the conversation history.

        Args:
            user_text (str): The user's message text to be added to the conversation history.
        """
        self.conversation_history_list.append(
            ConversationHistory.user_message(user_text))

    def clear(self):
        """
        TODO: documentation
        """
        self.conversation_history_list = list()

    def get_chat_gpt_conversation_history(self):
        """
        Retrieves the conversation history in a format that is compatible with models like ChatGPT.

        Returns:
            list: The complete list of conversation history in its current format.
        """
        return self.conversation_history_list

    def get_gemini_conversation_history(self):
        """
        Retrieves the conversation history in a format compatible with the Gemini model API.

        Extracts system instructions separately and formats the remaining conversation for Gemini.

        Returns:
            Tuple[str, list]: A tuple containing:
                - str: The system instruction extracted from the conversation.
                - list: A list of formatted Content objects for the Gemini API.
        """
        # Get system instruction
        system_message = search_dict_by_key_value(
            self.conversation_history_list, KEY_ROLE, ROLE_SYSTEM)
        if system_message is not None:
            system_instruction = system_message[KEY_CONTENT]
        else:
            system_instruction = None

        # Get conversation history
        gemini_contents = list()
        # Special case: only ONE MESSAGE -> SYSTEM MESSAGE
        if len(self.conversation_history_list) == 1 and self.conversation_history_list[0][KEY_ROLE] == ROLE_SYSTEM:
            gemini_contents = [Content(role=ROLE_USER,
                                       parts=[Part.from_text(self.conversation_history_list[0][KEY_CONTENT])])]
        else:
            for message in self.conversation_history_list:
                role = message[KEY_ROLE]
                content = message[KEY_CONTENT]

                if role != ROLE_SYSTEM:
                    gemini_role = ROLE_MODEL if role == ROLE_ASSISTANT else role
                    gemini_content = Content(role=gemini_role,
                                             parts=[Part.from_text(content)])

                    gemini_contents.append(gemini_content)

        # print(len(gemini_contents))

        return system_instruction, gemini_contents
