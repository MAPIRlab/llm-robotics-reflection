
from prompt.prompt import Prompt

# EXAMPLE 1: spatial reasoning + malformed JSON + correct whole response
EXAMPLE_1 = """
<EXAMPLE_1>
<SEMANTIC_MAP>
{"instances":{"obj1":{"bbox":{"center":[2.0,3.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":100,"results":{"book":80}},"obj2":{"bbox":{"center":[2.0,3.0,0.0],"size":[2.0,1.0,0.5]},"n_observations":50,"results":{"dining table":50}},"obj3":{"bbox":{"center":[5.0,5.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":80,"results":{"chair":70}},"obj4":{"bbox":{"center":[4.0,4.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":90,"results":{"book":85}}}}
</SEMANTIC_MAP>

<QUERY>
I need the book that is on a table
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify books that are on tables",
    "query_achievable": true,
    "relevant_objects": [
        {
            "object": "obj4",
            "relevance": "highly relevant"
        }
    ],
    "explanation": "The most relevant object is 'obj4', classified as a book, which is located on 'obj2' (dining table) based on their coordinates."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 3/10
- Query achievable: OK.
- Inferred query: OK.
- Relevant objects: Error. It is true that "obj4" is a book, but it is false that "obj4" is on the table, because of its bounding box center coordinates. Instead, "obj1" is a book with high confidence and it is located on a table ("obj2", which is classified as dining table with a high score).
- Explanation: Error. As the relevant objects are incorrectly identified, the explanation is wrong.
- JSON structure 1: OK. The response contains a JSON dictionary with four keys: "inferred_query", "query_achievable", "relevant_objects", and "explanation".
- JSON structure 2: Error. The preliminary response does not follow the required JSON structure; "relevant_objects" should be a list of object IDs as strings, not a list of dictionaries.
2. Comments on Relevance. Score: 4/10
- Identification: Error. The object identified for the task is wrong.
- Sorting: N/A.
- Details: OK.
- Leftovers: "obj4" should not belong to the response.
3. Comments on Clarity. Score: 7/10
- Clear: Even if the answer is incorrect, it contains a clear explanation.
- Ambiguities: OK.
4. Actions to improve response:
- Change "relevant_objects" to ["obj1"].
- Update the "explanation" accordingly.
- Adjust the JSON structure to match the required format, where "relevant_objects" is a list of object IDs as strings.
</EXAMPLE_1>
"""


# EXAMPLE 2: functionality + wrong order
EXAMPLE_2 = """
<EXAMPLE_2>
<SEMANTIC_MAP>
{"instances":{"obj1":{"bbox":{"center":[1,2,0],"size":[1,1,1]},"n_observations":100,"results":{"chair":90}},"obj2":{"bbox":{"center":[2,2,0],"size":[2,1,1.5]},"n_observations":80,"results":{"sofa":85}},"obj3":{"bbox":{"center":[3,2,0],"size":[1.5,1,0.5]},"n_observations":70,"results":{"bench":75}},"obj4":{"bbox":{"center":[4,2,0],"size":[2,1.5,0.5]},"n_observations":60,"results":{"dining table":80}},"obj5":{"bbox":{"center":[5,2,0],"size":[1,0.5,0.5]},"n_observations":50,"results":{"stool":65}}}}
</SEMANTIC_MAP>

<QUERY>
I need the book that is on a table
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Find objects suitable for sitting and working on a laptop",
    "query_achievable": true,
    "relevant_objects": ["obj5", "obj3", "obj2", "obj1"],
    "explanation": "Objects 'obj5', 'obj3', 'obj2', and 'obj1' can be used to sit while working on your laptop."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 4/10
- Query achievable: OK.
- Inferred query: OK.
- Relevant objects: Partially correct. The objects 'obj1' (chair), 'obj2' (sofa), 'obj3' (bench), and 'obj5' (stool) are all seating options. However, for working on a laptop, 'obj1' and 'obj2' are more suitable due to comfort and ergonomics. The preliminary response includes all seating objects but doesn't prioritize them correctly.
- Explanation: The explanation mentions the objects but fails to justify their relevance or prioritize them according to suitability.
- JSON structure 1: OK.
- JSON structure 2: OK.
2. Comments on Relevance. Score: 3/10
- Identification: Relevant objects are identified but not ordered by their suitability for the task.
- Sorting: Objects are not sorted by relevance; less suitable options like 'obj5' (stool) and 'obj3' (bench) are listed before more suitable ones like 'obj1' (chair) and 'obj2' (sofa).
- Details: The response lacks details on why certain objects are more appropriate than others.
- Leftovers: N/A.
3. Comments on Clarity. Score: 8/10
- Clear: The response is clear and easy to understand.
- Ambiguities: No significant ambiguities present.
4. Actions to Improve Response:
- Reorder the "relevant_objects" list to prioritize the most suitable objects for working on a laptop, listing 'obj1' and 'obj2' first.
- Enhance the "explanation" to justify the ordering, highlighting why certain objects are more appropriate for the task.
</EXAMPLE_2>
"""

# EXAMPLE 3: functionality extension
EXAMPLE_3 = """
<EXAMPLE_3>
<SEMANTIC_MAP>
{"instances":{"obj1":{"bbox":{"center":[1,2,0],"size":[0.5,0.5,1]},"n_observations":100,"results":{"trash can":90}},"obj2":{"bbox":{"center":[2,2.5,0],"size":[0.8,0.8,1.2]},"n_observations":80,"results":{"toilet":85}},"obj3":{"bbox":{"center":[3,3,0],"size":[1,1,1]},"n_observations":70,"results":{"refrigerator":75}},"obj4":{"bbox":{"center":[4,4,0],"size":[0.5,0.5,0.5]},"n_observations":60,"results":{"flower":80}}}}
</SEMANTIC_MAP>

<QUERY>
Find something that smells bad
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify objects that smell bad",
    "query_achievable": false,
    "relevant_objects": [],
    "explanation": "No objects that smell bad are found in the semantic map."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 3/10
- Query achievable: Error. The query is achievable using indirectly related objects present in the semantic map.
- Inferred query: Error. The inferred query is too narrow; it should include objects that may emit unpleasant odors.
- Relevant objects: Error. Objects like 'obj1' (trash can) and 'obj2' (toilet) are commonly associated with bad smells and should be included.
- Explanation: Error. The explanation overlooks indirectly relevant objects that could fulfill the user's request.
- JSON structure 1: OK.
- JSON structure 2: OK.
2. Comments on Relevance. Score: 5/10
- Identification: Error. The response fails to identify objects indirectly related to the query.
- Sorting: N/A.
- Details: Lacking. Additional details on why certain objects are relevant are missing.
- Leftovers: N/A.
3. Comments on Clarity. Score: 8/10
- Clear: The response is clear but incomplete.
- Ambiguities: None. The response is straightforward but misses pertinent information.
4. Actions to Improve Response:
- Change "query_achievable" to true since the task can be achieved with existing objects.
- Update "inferred_query" to "Find objects that may smell bad" to encompass indirectly related items.
- Include "relevant_objects": ["obj1", "obj2"], representing the trash can and the toilet.
- Revise the "explanation" to state that while no objects are explicitly labeled as smelling bad, these objects are associated with unpleasant odors and may satisfy the query.
</EXAMPLE_3>
"""

# EXAMPLE 4: response is ok
EXAMPLE_4 = """
<EXAMPLE_4>
<SEMANTIC_MAP>
{"instances":{"obj1":{"bbox":{"center":[1,2,0.5],"size":[1,0.5,0.5]},"n_observations":100,"results":{"sink":90}},"obj4":{"bbox":{"center":[3,3,0.5],"size":[1,0.5,0.5]},"n_observations":60,"results":{"sink":80}},"obj5":{"bbox":{"center":[5,5,0],"size":[1,1,1]},"n_observations":50,"results":{"bed":65}}}}
</SEMANTIC_MAP>

<QUERY>
I need to find a place to wash my hands
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify locations where you can wash your hands",
    "query_achievable": true,
    "relevant_objects": ["obj1", "obj4"],
    "explanation": "You can wash your hands at 'obj1' and 'obj4', which are both sinks."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 9/10
- Query achievable: OK.
- Inferred query: Excellent. The inferred query accurately captures the user's intent to find places suitable for washing hands.
- Relevant objects: OK.
- Explanation: Good. The explanation correctly identifies the sinks and their suitability for the task.
- JSON structure 1: OK.
- JSON structure 2: OK.
Comments on Relevance. Score: 10/10
- Identification: OK. All relevant objects are correctly identified.
- Sorting: N/A. Both sinks are equally relevant for the task.
- Details: Good level of detail provided in the explanation.
- Leftovers: No irrelevant objects are included in the response.
Comments on Clarity. Score: 10/10
- Clear: The response is clear and easy to understand.
- Ambiguities: No ambiguities present. The explanation is straightforward and concise.
Actions to Improve Response:
- Shortly indicate where each object is located to contribute to the improvement of the explanation.
</EXAMPLE_4>"""

SELF_REFLECTION_EXAMPLES = f"""
{EXAMPLE_1}
{EXAMPLE_2}
{EXAMPLE_3}
{EXAMPLE_4}
"""


class PromptReflectAgent(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
We have received a preliminary response from an LLM tasked with interpreting and answering questions about a 3D semantic map (<SEMANTIC_MAP>) provided in JSON format. Your task is to evaluate this response, providing constructive and detailed feedback to help refine and improve it.
</CONTEXT>

<INSTRUCTION>
Analyze the LLM-generated response, and offer SPECIFIC and ACTIONABLE feedback. This feedback will be used to make the response more accurate, relevant, and clear.
Reflect critically on the response based on the evaluation criteria below. 
The output should only contain constructive feedback—do not directly correct or rewrite the response, as that will be the task of another LLM.
</INSTRUCTION>

<ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>
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
</ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>

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
- Identification: Are all relevant objects identified?
- Sorting: Are all relevant objects sorted by relevance accurately?
- Details: Are any crucial objects or details omitted?
- Leftovers: Are there non-relevant objects in the answer, which should be omitted?
Clarity:
- Clear: Is the response clear and easy to understand?
- Ambiguities: Are there any ambiguities or vague descriptions?
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
Provide a list of actions to modify the preliminary response in order to increase its quality.
</OUTPUT_FORMAT>

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now you will receive QUERIES and their PRELIMINARY_RESPONSES, and your task is to generate feedback on them, taking into account the SEMANTIC_MAP.

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", SELF_REFLECTION_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


class PromptReflectAgent(Prompt):

    SYSTEM_PROMPT = """
<CONTEXT>
We have received a preliminary response from an LLM tasked with interpreting and answering questions about a 3D semantic map (<SEMANTIC_MAP>) provided in JSON format. Your task is to evaluate this response, providing constructive and detailed feedback to help refine and improve it.
</CONTEXT>

<ROLE>
You are an agent tasked with analyzing responses generated by a Large Language Model and providing specific, actionable feedback to improve accuracy, relevance, and clarity.
Your role is to critically reflect on the response based on established evaluation criteria, identifying areas where it could be enhanced. 
Focus on offering constructive insights that can guide another LLM in refining the response, without directly correcting or rewriting it yourself.
</ROLE>

<ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>
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
</ORIGINAL_TASK_INSTRUCTION_FOR_THE_INITIAL_LLM>

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

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now you will receive QUERIES and their PRELIMINARY_RESPONSES, and your task is to generate feedback on them, taking into account the SEMANTIC_MAP.

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", SELF_REFLECTION_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


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

Now generate feedback on the PRELIMINARY_RESPONSE (that is the response to the QUERY); take into account the OUTPUT_FORMAT.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
