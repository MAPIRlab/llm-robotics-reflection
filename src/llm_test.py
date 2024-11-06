
from constants import (
    GOOGLE_GEMINI_CREDENTIALS_FILENAME,
    GOOGLE_GEMINI_PROJECT_ID,
    GOOGLE_GEMINI_PROJECT_LOCATION,
)
from llm.conversation_history import ConversationHistory
from llm.google_gemini_provider import GoogleGeminiProvider

gemini = GoogleGeminiProvider(credentials_file=GOOGLE_GEMINI_CREDENTIALS_FILENAME,
                              project_id=GOOGLE_GEMINI_PROJECT_ID,
                              project_location=GOOGLE_GEMINI_PROJECT_LOCATION,
                              model_name=GoogleGeminiProvider.GEMINI_1_0_PRO)


if __name__ == "__main__":

    conversation_history = ConversationHistory()
    conversation_history.append_system_message(
        "Tienes que responder con todas en mayúsculas")
    conversation_history.append_user_message("Buenas tardes!")
    conversation_history.append_assistant_message(
        "HOLA, QUÉ TAL")
    conversation_history.append_user_message(
        "Por qué hablas en mayúsculas?...")
    conversation_history.append_assistant_message(
        "HE SIDO PROGRAMADO PARA ELLO")
    conversation_history.append_user_message(
        "Vale, puedes contarme algo sobre ti?")

    response = gemini.generate_text(conversation_history)

    print(response)
