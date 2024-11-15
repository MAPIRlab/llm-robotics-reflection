
from prompt.prompt import Prompt


class ChooserPrompt(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
We have received multiple responses from LLMs that interpreted a 3D semantic map provided in JSON format to answer user questions about the environment.
Only one of them is the one that the user should receive as the final answer, so it is necessary to evaluate which one is the best.
</CONTEXT>

<INSTRUCTION>
valuate these responses and select the best one based on three criteria: correctness (accuracy of information), relevance (usefulness and focus of information provided), and clarity (ease of understanding and explanation quality).
</INSTRUCTION>

<EVALUATION_CRITERIA>
Some hints you should consider for selecting the best answer among the received:
Correctness:
- Query achievable: Is the query achievable status appropriately determined?
- Inferred query: Is the inferred query related to the user's query?
- Relevant objects: Are the relevant objects accurately identified?
- Explanation: Is the explanation logically sound and coherent?
- JSON structure 1: Is the response a JSON object with four keys: "inferred_query", "query_achievable", "relevant_objects", and "explanation"?
- JSON structure 2: Is the list in "relevant_objects" a String list containing only object ids like "obj1", "obj102", etc.?
Relevance:
- Are all relevant objects identified?
- Are all relevant objects sorted by relevance accurately?
- Are any crucial objects or details omitted?
- Are there non-relevant objects in the answer, which should be omitted?
Clarity:
- Is the response clear and easy to understand?
- Are there any ambiguities or vague descriptions?
</EVALUATION_CRITERIA>

<ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>
The main instruction from the LLMs that generated the answers from which you are going to select the most accurate one was:
The input is a 3D semantic map in JSON format, where each object in the "instances" array includes the following properties:
- "bbox": Bounding box (center and size)
- "n_observations": Times observed
- "results": Classification results (category and certainty)
The goal is to parse this input to answer user questions and respond with a JSON structured as follows:
- "inferred_query": Summary of the user's query
- "query_achievable": Whether the task can be done
- "relevant_objects": List of relevant objects
- "explanation": How these objects help
</ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>

<OUTPUT_FORMAT>
Your output will be a JSON with two fields:
    - "chosen_response": (Integer) Index of the best response.
    - "explaination": (String) Explanation of why the chosen answer is better than the others.
It is very important that the value of “chosen_response” is only a number (and nothing else), which identifies the index of the chosen answer.
Otherwise the response won't be accepted and the user won't receive response.
</OUTPUT_FORMAT>

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>

<USER_QUERY>
The user's query to which the LLMs had to respond was.
{{query}}
</USER_QUERY>

<LLM_RESPONSES>
The RESPONSES from which you have to select the best one are:
{{llm_responses}}
</LLM_RESPONSES>

Now taking into account the EVALUATION_CRITERIA and the provided JSON formatted SEMANTIC_MAP, choose one response from LLM_RESPONSE and create an output following the OUTPUT_FORMAT.
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
