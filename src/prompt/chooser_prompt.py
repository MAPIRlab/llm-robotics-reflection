
from prompt.prompt import Prompt


class ChooserPrompt(Prompt):

    # TODO: include user query
    SYSTEM_PROMPT = """
<CONTEXT>
We have received some responses from other LLMs tasked with interpreting and answering questions about a semantic map described in a JSON format.

The INSTRUCTION literally was:
| The input to the model is a 3D semantic map in a JSON format (### SEMANTIC MAP ###). Each entry of the "instances" JSON object describes one object in the scene, with the following fields:
| 
| 1. "bbox": 3D bounding box of the object
|     1.1. "bbox.center": Center of the bounding box of the object
|     1.2. "bbox.size": Size of the bounding box of the object
| 2. "n_observations": Number of observations of the object in the scene
| 3. "results": Results of the classification of the object, indicating a category and the certainty that the object belongs to the category
| 
| Read and parse the JSON in order to answer users questions about the scene, indicating the most related object to perform the task.
| For each user question, respond with a JSON dictionary with the following fields:
| 
| 1. "inferred_query": (String) Your interpretation of the user query in a succinct form
| 2. "query_achievable": (Boolean) Whether or not the user specified query is achievable using the objects and descriptions provided in the 3D scene.
| 3. "explanation": (String) A brief explanation of what the most relevant object(s) is(are), and how they achieve the user-specified task.
| 4. "chosen_object": (String) Object identifier of the object most related to the task requested by the user.

The SEMANTIC MAP was:
{{semantic_map}}

The RESPONSES from which you have to select the best one are:
{{llm_responses}}
</CONTEXT>

<INSTRUCTION>
Your task is to choose the best answer from the above answers based on correctness, relevance and clarity. 
</INSTRUCTION>

<OUTPUT_FORMAT>
Your output will be a JSON with two fields:
    - "chosen_response": (Integer) Index of the best response.
    - "explaination": (String) Explanation of why the chosen answer is better than the others.
</OUTPUT_FORMAT>
    """

    def __init__(self, llm_responses: list, **prompt_data_dict):
        self.llm_responses = llm_responses
        self.prompt_data_dict = prompt_data_dict

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_llm_responses(self, llm_responses: list, prompt_text: str) -> str:
        responses_text = ""

        for llm_response_idx, llm_response in enumerate(llm_responses):
            responses_text += f"RESPONSE {llm_response_idx}: \n"
            responses_text += f"{llm_response}\n"

        return prompt_text.replace("{{llm_responses}}", responses_text)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_llm_responses(
            self.llm_responses, prompt_text)
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        return prompt_text
