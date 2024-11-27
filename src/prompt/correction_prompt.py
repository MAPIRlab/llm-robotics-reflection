
from prompt.prompt import Prompt

# EXAMPLE 1: malformed JSON
EXAMPLE_1 = """
<EXAMPLE_1>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[2.0,3.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":100,"results":{"book":80}},
"obj2":{"bbox":{"center":[2.0,3.0,0.0],"size":[2.0,1.0,0.5]},"n_observations":50,"results":{"dining table":50}},
"obj3":{"bbox":{"center":[5.0,5.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":80,"results":{"chair":70}},
"obj4":{"bbox":{"center":[4.0,4.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":90,"results":{"book":85}}}}
</SEMANTIC_MAP>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Find laptops",
    "query_achievable": true,
    "relevant_objects": [
        {
            "object": "obj4",
            "relevance": "highly relevant"
        }
    ],
    "explanation": "Object 'obj4' is a laptop located on 'obj2' (office desk), making it the relevant object."
}
</PRELIMINARY_RESPONSE>

<FEEDBACK>
1. Comments on Correctness. Score: 1/10
- Inferred query: Partially correct. The inferred query should be more specific, as the query requested searching laptops that were on tables, not only laptops.
- Relevant objects: Error. It is true that "obj4" is a laptop, but it is false that "obj4" is on the table, because of its bounding box center coordinates. Instead, "obj1" is a laptop with high confidence and it is located on a table ("obj2", which is classified as office desk with a high score).
- Explanation: error. As the relevant objects are incorrectly identified, the explanation is wrong.
- JSON structure 1: error. The preliminary response does not follow the required JSON structure; "relevant_objects" should be a list of object IDs as strings, not a list of dictionaries.
- JSON structure 2: N/A since the JSON structure is incorrect.
2. Comments on Relevance. Score: 4/10
- Identification: error. The object identified for the task is wrong.
- Sorting: N/A.
- Details: ok.
- Leftovers: "obj4" should not belong to the response.
3. Comments on Clarity. Score: 7/10
- Clear: Even if the answer is incorrect, it contains a clear explanation.
- Ambiguities: ok.
4. Actions to Improve Response:
- Include "inferred_query": "Identify laptops that are on tables."
- Change "relevant_objects" to ["obj1"].
- Update the "explanation" accordingly.
- Adjust the JSON structure to match the required format, where "relevant_objects" is a list of object IDs as strings.
</FEEDBACK>

EXPECTED RESPONSE:
{
    "inferred_query": "Identify laptops that are on tables.",
    "query_achievable": true,
    "relevant_objects": ["obj1"],
    "explanation": "The semantic map contains two laptops, 'obj1' and 'obj4'. 'obj1' is located on 'obj2' (office desk) based on their center coordinates, making it the correct relevant object."
}
</EXAMPLE_1>
"""

# EXAMPLE 2: missing objects
EXAMPLE_2 = """
<EXAMPLE_2>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,2,0],"size":[0.5,0.5,1]},"n_observations":100,"results":{"trash can":90}},
"obj2":{"bbox":{"center":[2,2.5,0],"size":[0.8,0.8,1.2]},"n_observations":80,"results":{"toilet":85}},
"obj3":{"bbox":{"center":[3,3,0],"size":[1,1,1]},"n_observations":70,"results":{"refrigerator":75}},
"obj4":{"bbox":{"center":[4,4,0],"size":[0.5,0.5,0.5]},"n_observations":60,"results":{"flower":80}}}}
</SEMANTIC_MAP>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify objects that may emit unpleasant odors",
    "query_achievable": true,
    "relevant_objects": ["obj1"],
    "explanation": "The semantic map contains 'obj1' (trash can), which is typically associated with bad smells."
}
</PRELIMINARY_RESPONSE>

<FEEDBACK>
1. Comments on Correctness. Score: 4/10
- Query achievable: OK.
- Inferred query: Partially correct. It should include a broader range of cooking-related objects, not just appliances.
- Relevant objects: Error. It is true that "obj1" (oven) and "obj3" (sink) are relevant, but "obj2" (refrigerator), "obj4" (microwave), "obj5" (cutting board), and "obj6" (blender) are also used for cooking and should be included.
- Explanation: Error. The explanation fails to mention the omitted relevant objects and their importance in cooking.
- JSON structure 1: OK. The preliminary response follows the required JSON structure.
- JSON structure 2: OK. The "relevant_objects" is a list of object identifiers.
2. Comments on Relevance. Score: 6/10
- Identification: Partially correct. Some relevant objects are identified, but others are omitted, as for example "obj2" (refrigerator), "obj4" (microwave), "obj5" (cutting board), and "obj6" (blender)
- Sorting: N/A.
- Details: **Partially lacking.** The explanation does not cover all relevant objects.
- Leftovers: N/A.
3. Comments on Clarity. Score: 8/10
- Clear: The response is clear but incomplete.
- Ambiguities: OK.
4. Actions to Improve Response:
- Update "inferred_query" to "Identify objects used for cooking."
- Include "obj2" (refrigerator), "obj4" (microwave), "obj5" (cutting board), and "obj6" (blender) in the "relevant_objects" list.
- Adjust the relevance in the "relevant_objects" list appropriately, e.g., "obj1" (oven) and "obj2" (refrigerator) first, "obj4" (microwave) and "obj6" (blender) then, and "obj5" (cutting board) last.
- Enhance the "explanation" to include all relevant objects and their roles in cooking.
</FEEDBACK>

EXPECTED RESPONSE:
{
    "inferred_query": "Identify objects used for cooking.",
    "query_achievable": true,
    "relevant_objects": ["obj1", "obj2", "obj4", "obj6", "obj5"],
    "explanation": "The semantic map contains several cooking-related objects: 'obj1' (oven) and 'obj2' (refrigerator) are essential appliances for cooking, 'obj4' (microwave), 'obj6' (blender) and 'obj5' (cutting board) are also commonly used in the kitchen ."
}
</EXAMPLE_2>
"""

# EXAMPLE 3: wrong order
EXAMPLE_3 = """
<EXAMPLE_3>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,1,0],"size":[1,0.5,0.5]},"n_observations":100,"results":{"desk":95}},
"obj2":{"bbox":{"center":[2,2,0],"size":[0.5,0.5,0.5]},"n_observations":80,"results":{"laptop":90}},
"obj3":{"bbox":{"center":[3,1,0.5],"size":[0.2,0.2,0.1]},"n_observations":50,"results":{"coffee mug":85}},
"obj4":{"bbox":{"center":[4,3,0],"size":[0.4,0.4,0.2]},"n_observations":60,"results":{"chair":75}}}}
</SEMANTIC_MAP>

<PRELIMINARY_RESPONSE>
{
"inferred_query": "Identify objects needed for working.",
"query_achievable": true,
"relevant_objects": ["obj3", "obj1", "obj2", "obj4"],
"explanation": "The semantic map contains several objects relevant for work: 'obj3' (coffee mug), 'obj1' (desk), 'obj2' (laptop), and 'obj4' (chair) are identified as useful objects."
}
</PRELIMINARY_RESPONSE>

<FEEDBACK>
1. Comments on Correctness. Score: 7/10 
- Relevant objects: All relevant objects are identified, but they are not ordered by relevance for the query. 
- Explanation: The explanation identifies the objects but does not justify their order based on their importance to the query.
2. Comments on Relevance. Score: 6/10
- Sorting: Error. The objects should be ordered by their relevance for working, prioritizing essential items like the laptop and desk.
- Details: Missing clarification on sorting criteria.
3. Comments on Clarity. Score: 8/10
- Ambiguities: None, apart from the lack of sorting.
4. Actions to Improve Response:
- Reorder the "relevant_objects" list to prioritize essential work-related objects, such as the laptop and desk, over supplementary ones. The correct order should be: obj2 (laptop), obj1 (desk), obj4 (chair), obj3 (coffee mug), as this reflects their relevance to the query.
- Update the "explanation" to reflect the new order and justify the relevance of each object.
</FEEDBACK>

EXPECTED RESPONSE:
{
"inferred_query": "Identify objects needed for working.",
"query_achievable": true,
"relevant_objects": ["obj2", "obj1", "obj4", "obj3"],
"explanation": "The semantic map contains several objects relevant for working. obj2 (laptop) is the most essential item for work, followed by obj1 (desk) as a workspace, obj4 (chair) for seating, and obj3 (coffee mug) as a supplementary item."
}
</EXAMPLE_3>
"""

CORRECTION_EXAMPLES = f"""
{EXAMPLE_1}
"""


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
The response should ALWAYS be a JSON object, no explanation, titles or additional information required. It is very important that the response is a JSON parsable object, because your response will be directly passed to a software that should save these response as JSON.
</OUTPUT_FORMAT>

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now you will receive a PRELIMINARY_RESPONSE and its FEEDBACK, and your task is to follow the step indicated in the feedback and correct the preliminary response by generating another one.

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", CORRECTION_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


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
The response should ALWAYS be a JSON object, no explanation, titles or additional information required. It is very important that the response is a JSON parsable object, because your response will be directly passed to a software that should save these response as JSON.
</OUTPUT_FORMAT>

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now you will receive PRELIMINARY_RESPONSES and their FEEDBACK, and your task is to correct the preliminary response and generate a new response based on the feedback.

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", CORRECTION_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


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
