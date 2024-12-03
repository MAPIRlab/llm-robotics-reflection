

import google.cloud.aiplatform as aiplatform
import google.oauth2.service_account
from vertexai.preview.generative_models import GenerativeModel

from llm.conversation_history import ConversationHistory
from llm.large_language_model import LargeLanguageModel


class GoogleGeminiProvider(LargeLanguageModel):

    GEMINI_1_0_PRO = "gemini-1.0-pro"
    GEMINI_1_0_PRO_VISION = "gemini-1.0-pro-vision"
    GEMINI_1_5_PRO = "gemini-1.5-pro"

    def __init__(self, credentials_file: str, project_id: str, project_location: str, model_name: str):
        """
        Initialize the GoogleGeminiProvider with the specified credentials, project ID, project location, and model name.

        Args:
            credentials_file (str): Path to the service account credentials file.
            project_id (str): Google Cloud project ID.
            project_location (str): Google Cloud project location.
            model_name (str): Name of the model to be used.
        """
        credentials = (
            google.oauth2.service_account.Credentials.from_service_account_file(
                filename=credentials_file
            )
        )
        aiplatform.init(project=project_id,
                        location=project_location,
                        credentials=credentials)

        self.model_name = model_name

    def get_provider_name(self) -> str:
        return f"Google_{self.model_name}"

    def generate_text(self, conversation_history: ConversationHistory) -> str:
        # Get conversation history
        system_instruction, contents = conversation_history.get_gemini_conversation_history()

        # Instantiate model
        model = GenerativeModel(model_name=self.model_name,
                                system_instruction=system_instruction)

        # print("#"*100)
        # print(f"system_instruction = {system_instruction}")
        # print(f"contents = {contents}")

        # Get response
        response = model.generate_content(contents)

        response_text = response.candidates[0].content.parts[0].text
        print("RESPONSE")
        print(response_text)
        print("#"*100)

        # time.sleep(7)

        return response_text
