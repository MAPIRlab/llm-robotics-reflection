
from prompt.prompt import Prompt


class SelfReflectionSystemPrompt(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
We have received a preliminary response from another LLM tasked with interpreting and answering questions about a semantic map (<SEMANTIC_MAP>) described in a JSON format. Your task is to reflect on this response, providing constructive feedback to help refine and improve it.

The instruction of the first LLM that generated the response you are going to reflect on, was:
Input is a 3D semantic map in JSON format. Each object in "instances" has:
- "bbox": Bounding box (center and size)
- "n_observations": Times observed
- "results": Classification results (category and certainty)
Parse this to answer user questions. Respond with a JSON containing:
- "inferred_query": Summary of the user's query
- "query_achievable": Whether the task can be done
- "relevant_objects": List of relevant objects
- "explanation": How these objects help
</CONTEXT>

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>

<INSTRUCTION>
Your task is to REFLECT on the LLM-generated response the user will introduce and suggest SPECIFIC CHANGES that another LLM could implement in the response to make it more accurate, relevant, and clear.
</INSTRUCTION>

<EVALUATION_TAKE_INTO_ACCUNT>
Some hints on how to perform the feedback:
Correctess:
- Query achievable: Is the query achievable status appropriately determined?
- Inferred query: Is the inferred query related to the user's query?
- Relevant objects: Are the relevant objects accurately identified?
- Explaination: Is the explanation logically sound and coherent?
Relevance:
- Are all relevant objects identified, and are they sorted by relevance accurately?
- Are any crucial objects or details omitted?
Clarity:
- Is the response clear and easy to understand?
- Are there any ambiguities or vague descriptions?
</EVALUATION_TAKE_INTO_ACCUNT>

<OUTPUT_FORMAT>
Based on the query and the preliminary answer obtained, make a list of corrections for the preliminary answer so that another LLM can apply those corrections. Do not return the corrected answer, that is the task of another LLM.
</OUTPUT_FORMAT>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


class SelfReflectionUserPrompt(Prompt):

    SYSTEM_PROMPT = """
<QUERY>
The query was:
{{query}}
</QUERY>

<PRELIMINARY_RESPONSE>
The preliminary response was:
{{preliminary_response}}
</PRELIMINARY_RESPONSE>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
