
from prompt.prompt import Prompt


class PromptReflect(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
We have received a preliminary response from an LLM tasked with interpreting and answering questions about a 3D semantic map (<SEMANTIC_MAP>) provided in JSON format. Your task is to evaluate this response, providing constructive and detailed feedback to help refine and improve it.
</CONTEXT>

The instruction of the first LLM that generated the response you are going to reflect on, was:
<ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>
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

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>

<INSTRUCTION>
Analyze the LLM-generated response, and offer SPECIFIC and ACTIONABLE feedback. This feedback will be used to make the response more accurate, relevant, and clear.
Reflect critically on the response based on the evaluation criteria below. 
The output should only contain constructive feedback—do not directly correct or rewrite the response, as that will be the task of another LLM.
</INSTRUCTION>

<EVALUATION_CRITERIA>
Some hints on how to perform the feedback:
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

<OUTPUT_FORMAT>
Organize your response into four clearly identified sections. Sections should be identified with its number from 1 to 4 and its title.
1. Comments on Correctness:
Provide a score from 0 to 10 indicating the response's accuracy, followed by an evaluation of the response's accuracy in relation to the criteria for correctness.
2. Comments on Relevance:
Provide a score from 0 to 10 indicating how relevant and well-targeted the response is, followed by your findings on its relevance to the user's query.
3. Comments on Clarity:
Comment on the clarity and readability of the response, identifying any ambiguities.
4. Actions to Improve Response:
Provide a score from 0 to 10 indicating the response's clarity, followed by your comments on its readability and any ambiguities.
</OUTPUT_FORMAT>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


class PromptReflectUser(Prompt):

    SYSTEM_PROMPT = """
<QUERY>
The query was:
{{query}}
</QUERY>

<PRELIMINARY_RESPONSE>
The preliminary response was:
{{plan_response}}
</PRELIMINARY_RESPONSE>

Now based on the QUERY and the PRELIMINARY_RESPONSE provided, generate feedback taking into account the OUTPUT_FORMAT.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
