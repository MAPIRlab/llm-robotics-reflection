
from prompt.prompt import Prompt


class PromptCorrect(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
You will receive a response from an LLM tasked with interpreting and answering questions about a semantic map described in JSON format (<PRELIMINARY_RESPONSE>).
You will also receive a response from an LLM tasked with reflection on the first response, providing constructive feedback consisting on specific actions to help refine and improve the first response (<FEEDBACK_RESPONSE>).
</CONTEXT>

<INSTRUCTION>
You are provided the preliminary LLM response (<PRELIMINARY_RESPONSE>) and the reflection response (<FEEDBACK>).
Apply the feedback provided in the critique to produce a refined and corrected response.
</INSTRUCTION>

<INSTRUCTION_FOR_LLM_THAT_GENERATED_PRELIMINARY_RESPONSE>
The instruction of the first LLM that generated the response you are going to reflect on, was:
The input is a 3D semantic map in JSON format, where each object in the "instances" array includes the following properties:
- "bbox": Bounding box (center and size)
- "n_observations": Times observed
- "results": Classification results (category and certainty)
The goal is to parse this input to answer user questions and respond with a JSON structured as follows:
- "inferred_query": Summary of the user's query
- "query_achievable": Whether the task can be done
- "relevant_objects": List of relevant objects
- "explanation": How these objects help
</INSTRUCTION_FOR_LLM_THAT_GENERATED_PRELIMINARY_RESPONSE>

<INSTRUCTION_FOR_LLM_THAT_GENERATED_FEEDBACK>
The instruction of the LLM that generated the reflection was:
Review a response from another LLM that interpreted a 3D semantic map in JSON format, focusing on:
1. Correctness: Does the response accurately infer the query, determine task achievability, identify relevant objects, and provide a logical explanation? (Score 0-10)
2. Relevance: Are all relevant objects included, ordered by importance, with no key details missing? (Score 0-10)
3. Clarity: Is the response clear, understandable, and unambiguous? (Score 0-10)
4. Provide specific actions to improve the response (not a corrected answer). If no corrections are needed, leave this list empty. 
Each section should be clearly identified.
<INSTRUCTION_FOR_LLM_THAT_GENERATED_FEEDBACK>

<OUTPUT_FORMAT>
Note that the response must again be a JSON, with the same keys as the first LLM call, but with the values corrected according to the feedback.
1 "inferred_query": (String)
2 "query_achievable": (Boolean)
3 "relevant_objects": (List of String)
4 "explanation": (String)
The response should only be a JSON object, no explanations, titles or additional information required.
</OUTPUT_FORMAT>

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>

Now you will receive PRELIMINARY_RESPONSES and their FEEDBACK, and your task is to correct the preliminary response and generate a new response based on the feedback.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


class PromptCorrectAgent(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
You will receive a response from an LLM tasked with interpreting and answering questions about a semantic map described in JSON format (<PRELIMINARY_RESPONSE>).
You will also receive a response from an LLM tasked with reflection on the first response, providing constructive feedback consisting on specific actions to help refine and improve the first response (<FEEDBACK_RESPONSE>).
</CONTEXT>

<ROLE> 
You are an agent responsible for refining and correcting preliminary responses generated by a language model (LLM). 
Using both the preliminary response (<PRELIMINARY_RESPONSE>) and the feedback from a critique (<FEEDBACK>), your task is to apply the feedback effectively to produce an improved, polished response.
</ROLE>

<INSTRUCTION_FOR_LLM_THAT_GENERATED_PRELIMINARY_RESPONSE>
The instruction of the first LLM that generated the response you are going to reflect on, was:
The input is a 3D semantic map in JSON format, where each object in the "instances" array includes the following properties:
- "bbox": Bounding box (center and size)
- "n_observations": Times observed
- "results": Classification results (category and certainty)
The goal is to parse this input to answer user questions and respond with a JSON structured as follows:
- "inferred_query": Summary of the user's query
- "query_achievable": Whether the task can be done
- "relevant_objects": List of relevant objects
- "explanation": How these objects help
</INSTRUCTION_FOR_LLM_THAT_GENERATED_PRELIMINARY_RESPONSE>

<INSTRUCTION_FOR_LLM_THAT_GENERATED_FEEDBACK>
The instruction of the LLM that generated the reflection was:
Review a response from another LLM that interpreted a 3D semantic map in JSON format, focusing on:
1. Correctness: Does the response accurately infer the query, determine task achievability, identify relevant objects, and provide a logical explanation? (Score 0-10)
2. Relevance: Are all relevant objects included, ordered by importance, with no key details missing? (Score 0-10)
3. Clarity: Is the response clear, understandable, and unambiguous? (Score 0-10)
4. Provide specific actions to improve the response (not a corrected answer). If no corrections are needed, leave this list empty. 
Each section should be clearly identified.
<INSTRUCTION_FOR_LLM_THAT_GENERATED_FEEDBACK>

<OUTPUT_FORMAT>
Note that the response must again be a JSON, with the same keys as the first LLM call, but with the values corrected according to the feedback.
1 "inferred_query": (String)
2 "query_achievable": (Boolean)
3 "relevant_objects": (List of String)
4 "explanation": (String)
The response should only be a JSON object, no explanations, titles or additional information required.
</OUTPUT_FORMAT>

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>

Now you will receive PRELIMINARY_RESPONSES and their FEEDBACK, and your task is to correct the preliminary response and generate a new response based on the feedback.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


class PromptCorrectUser(Prompt):

    SYSTEM_PROMPT = """
<PRELIMINARY_RESPONSE>
The prelimiary response was:
{{plan_response}}
</PRELIMINARY_RESPONSE>

<FEEDBACK>
The reflection response was:
{{self_reflection_response}}
</FEEDBACK>

Now correct the feedback in FEEDBACK_RESPONSE to improve the PRELIMINARY_RESPONE quality, take into account the SEMANTIC_MAP.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
