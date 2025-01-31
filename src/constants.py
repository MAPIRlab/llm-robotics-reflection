import os

from dotenv import load_dotenv

from llm.google_gemini_provider import GoogleGeminiProvider
from llm.openai_gpt_provider import OpenAiGptProvider

load_dotenv()

# Credentials
GOOGLE_GEMINI_CREDENTIALS_FILENAME = "credentials/capable-alcove-439311-g6-0ec434da875e.json"
GOOGLE_GEMINI_PROJECT_ID = "capable-alcove-439311-g6"
GOOGLE_GEMINI_PROJECT_LOCATION = "us-central1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
SEMANTIC_MAPS_FOLDER_PATH = "data/semantic_maps/"
DATA_FOLDER_PATH = "data/"
GROUND_TRUTH_FOLDER_PATH = "data/responses"

QUERIES_FILE_PATH = "data/queries.yaml"

LLM_RESULTS_FOLDER_PATH = "results/llm_results"

# Code constants
MODE_CERTAINTY = "certainty"
MODE_UNCERTAINTY = "uncertainty"

METHOD_BASE = "base"
METHOD_SELF_REFLECTION = "self_reflection"
METHOD_MULTIAGENT_REFLECTION = "multiagent_reflection"
METHOD_ENSEMBLE = "ensemble"
METHODS = [METHOD_BASE, METHOD_SELF_REFLECTION,
           METHOD_MULTIAGENT_REFLECTION, METHOD_ENSEMBLE]

LLM_GEMINI_1_0_PRO = "g10p"
LLM_GEMINI_1_5_PRO = "g15p"


def get_llm_provider_name_from_constant(constant_value: str):
    if constant_value == LLM_GEMINI_1_0_PRO:
        return "Google_gemini-1.0-pro"
    elif constant_value == LLM_GEMINI_1_5_PRO:
        return "Google_gemini-1.5-pro"
    else:
        raise ValueError(f"Constant value {constant_value} not known")


METRIC_TOP_1 = "top_1"
METRIC_TOP_2 = "top_2"
METRIC_TOP_3 = "top_3"
METRIC_TOP_ANY = "top_any"


def pretty_metric_constant(metric: str):
    if metric == METRIC_TOP_1:
        return "Top-1"
    elif metric == METRIC_TOP_2:
        return "Top-2"
    elif metric == METRIC_TOP_3:
        return "Top-3"
    elif metric == METRIC_TOP_ANY:
        return "Top-Any"


EVALUATION_TABLE_WORKFLOWS = "table_workflows"
EVALUATION_TABLE_WORKFLOWS_GENERAL = "table_workflows_general"
EVALUATION_CHART_COMPLEXITY = "chart_complexity"
EVALUATION_REFLECTION_ERRORS = "reflection_errors"

# LLM models
GEMINI_1_0_PRO = GoogleGeminiProvider(credentials_file=GOOGLE_GEMINI_CREDENTIALS_FILENAME,
                                      project_id=GOOGLE_GEMINI_PROJECT_ID,
                                      project_location=GOOGLE_GEMINI_PROJECT_LOCATION,
                                      model_name=GoogleGeminiProvider.GEMINI_1_0_PRO)

GEMINI_1_5_PRO = GoogleGeminiProvider(credentials_file=GOOGLE_GEMINI_CREDENTIALS_FILENAME,
                                      project_id=GOOGLE_GEMINI_PROJECT_ID,
                                      project_location=GOOGLE_GEMINI_PROJECT_LOCATION,
                                      model_name=GoogleGeminiProvider.GEMINI_1_5_PRO)

CHAT_GPT_3_5_TURBO = OpenAiGptProvider(openai_api_key=OPENAI_API_KEY,
                                       model_name=OpenAiGptProvider.GPT_3_5_TURBO,
                                       max_output_tokens=4096)

CHAT_GPT_4_O = OpenAiGptProvider(openai_api_key=OPENAI_API_KEY,
                                 model_name=OpenAiGptProvider.GPT_4_O,
                                 max_output_tokens=4096)

LLM_PROVIDERS = [GEMINI_1_0_PRO, GEMINI_1_5_PRO]
