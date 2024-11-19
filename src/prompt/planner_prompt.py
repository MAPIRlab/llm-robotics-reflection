
from prompt.prompt import Prompt

# EXAMPLE 1: descriptive queries
EXAMPLE_1 = """
<EXAMPLE_1>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,1.5,0.5],"size":[0.5,0.5,0.2]},"n_observations":90,"results":{"notebook":92}},
"obj2":{"bbox":{"center":[1,1.5,0],"size":[2,1,0.5]},"n_observations":75,"results":{"desk":89}},
"obj3":{"bbox":{"center":[3,3,0.5],"size":[0.6,0.4,0.3]},"n_observations":65,"results":{"magazine":88}},
"obj4":{"bbox":{"center":[3.5,3,0],"size":[2,1,0.5]},"n_observations":80,"results":{"coffee table":85}}}}
</SEMANTIC_MAP>

<QUERY>
Locate a notebook lying on a workspace.
</QUERY>

EXPECTED RESPONSE:
{  
    "inferred_query": "Identify a notebook lying on a workspace.",  
    "query_achievable": true,  
    "relevant_objects": ["obj1", "obj2"],  
    "explanation": "The semantic map identifies 'obj1' as a notebook with high confidence (92) and 'obj2' as a desk directly beneath it, based on their bounding box center coordinates. This makes 'obj1' the notebook lying on a workspace."  
}  
</EXAMPLE1>
"""

# EXAMPLE 2: affordance queries
EXAMPLE_2 = """
<EXAMPLE_2>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,1,0],"size":[2,1,0.5]},"n_observations":85,"results":{"sofa":90}},
"obj2":{"bbox":{"center":[0.5,1.5,0.5],"size":[0.5,0.5,0.3]},"n_observations":60,"results":{"side table":88}},
"obj3":{"bbox":{"center":[3,2,0.5],"size":[0.6,0.4,0.2]},"n_observations":50,"results":{"chair":85}},
"obj4":{"bbox":{"center":[2,3,0],"size":[1.5,0.5,0.5]},"n_observations":70,"results":{"coffee table":89}}}}
</SEMANTIC_MAP>

<QUERY>
Where can I rest comfortably and place a drink next to me?
</QUERY>

EXPECTED RESPONSE:
{  
    "inferred_query": "Identify a place to rest comfortably and place a drink nearby.",  
    "query_achievable": true,  
    "relevant_objects": ["obj1", "obj2"],  
    "explanation": "The semantic map identifies 'obj1' as a sofa with high confidence (90%), providing a comfortable place to rest, and 'obj2' as a side table nearby, suitable for holding a drink. Their bounding box center coordinates confirm their spatial proximity."  
}
</EXAMPLE2>
"""

EXAMPLE_3 = """
<EXAMPLE_3>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[2,1.5,0],"size":[0.4,0.4,0.3]},"n_observations":90,"results":{"cup":85}},
"obj2":{"bbox":{"center":[1,1,0],"size":[0.5,0.5,0.5]},"n_observations":88,"results":{"bowl":92}},
"obj3":{"bbox":{"center":[3,2,0.5],"size":[0.3,0.3,0.2]},"n_observations":75,"results":{"plate":90}},
"obj4":{"bbox":{"center":[4,1,0],"size":[0.8,0.5,0.5]},"n_observations":70,"results":{"tray":80}}}}
</SEMANTIC_MAP>

<QUERY>
Find something to hold food that isn't a plate.
</QUERY>

EXPECTED RESPONSE:
{  
    "inferred_query": "Identify an object to hold food that is not a plate.",  
    "query_achievable": true,  
    "relevant_objects": ["obj2", "obj1", "obj4"],  
    "explanation": "The semantic map identifies 'obj2' as a bowl with the highest confidence (92%), making it the best option to hold food. 'obj1' (cup) and 'obj4' (tray) are also suitable, but ranked lower based on their typical usage for food storage or serving."  
}
</EXAMPLE_3>
"""

PLAN_EXAMPLES = f"""
{EXAMPLE_1}
{EXAMPLE_2}
{EXAMPLE_3}"""


class PromptPlan (Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>
The input to the model is a 3D SEMANTIC_MAP in a JSON format (<SEMANTIC_MAP>). Each entry of the "instances" JSON object describes one object in the scene, with the following fields:

1 "bbox": 3D bounding box of the object
    1.1 "bbox.center": Center of the bounding box of the object
    1.2 "bbox.size": Size of the bounding box of the object
2 "n_observations": Number of observations of the object in the scene
3 "results": Results of the classification of the object, indicating a category and the certainty that the object belongs to that category

Analyze this SEMANTIC_MAP and respond to the user's QUERY by identifying and listing the objects most relevant to perform the specified task. 
This list should be ordered by relevance, from the most relevant to the least relevant. 
In determining relevance, you should consider both the classification data of each object and their spatial arrangements within the scene, such as which objects are next to, above, or below others. 
Although explicit relationships are not provided in the semantic map, you can infer relevance from the scene layout to best answer the query.
</INSTRUCTION>

<OUTPUT_FORMAT>
Respond with a JSON dictionary with the following fields:
1 "inferred_query": (String) Your interpretation of the user query in summary form
2 "query_achievable": (Boolean) Whether or not the user-specified query is achievable using the objects and descriptions provided in the semantic map
3 "relevant_objects": (List of String) List of objects relevant to the user's query (ordered by relevance, most relevant first, least relevant last); or empty list in case there is no relevant object. Objects must be represented here by their ids.
4 "explanation": (String) A brief explanation of what the most relevant objects are, and how they achieve the user-specified task
The response should only be a JSON object, no explanations, titles or additional information required.
</OUTPUT_FORMAT>

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now respond to the user query following the INSTRUCTION and the OUTPUT_FORMAT, based on the input SEMANTIC_MAP.

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>

<USER_QUERY>
{{query}}
</USER_QUERY>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", PLAN_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


class PromptPlanAgent (Prompt):

    SYSTEM_PROMPT = """
<ROLE>
You are an agent responsible for interpreting a 3D semantic map, provided in JSON format (<SEMANTIC_MAP>), to assist in answering user queries about objects in the scene. Each entry in the "instances" JSON object describes an object with the following attributes:
1 "bbox": 3D bounding box of the object
1.1 "bbox.center": Center of the bounding box of the object
1.2 "bbox.size": Size of the bounding box of the object
2 "n_observations": Number of observations of the object in the scene
3 "results": Classification results for the object, indicating a category and the certainty level

Your task is to analyze this SEMANTIC_MAP and respond to the user's QUERY by identifying and listing the objects most relevant to perform the specified task. 
This list should be ordered by relevance, from the most relevant to the least relevant. 
In determining relevance, you should consider both the classification data of each object and their spatial arrangements within the scene, such as which objects are next to, above, or below others. 
Although explicit relationships are not provided in the semantic map, you can infer relevance from the scene layout to best answer the query.
</ROLE>

<OUTPUT_FORMAT>
Respond with a JSON dictionary with the following fields:
1 "inferred_query": (String) Your interpretation of the user query in summary form
2 "query_achievable": (Boolean) Whether or not the user-specified query is achievable using the objects and descriptions provided in the semantic map
3 "relevant_objects": (List of String) List of objects relevant to the user's query (ordered by relevance, most relevant first, least relevant last); or empty list in case there is no relevant object. Objects must be represented here by their ids.
4 "explanation": (String) A brief explanation of what the most relevant objects are, and how they achieve the user-specified task
The response should only be a JSON object, no explanations, titles or additional information required.
</OUTPUT_FORMAT>

<EXAMPLES>
Here are some examples of the process:
{{examples}}
</EXAMPLES>

Now you will receive user queries and respond to them following the ROLE and the OUTPUT_FORMAT, based on the input SEMANTIC_MAP.

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def replace_examples(self, prompt_text):
        return prompt_text.replace("{{examples}}", PLAN_EXAMPLES)

    def global_replace(self, prompt_text: str) -> str:
        prompt_text = self.replace_prompt_data_dict(
            self.prompt_data_dict, prompt_text)
        prompt_text = self.replace_examples(prompt_text)
        return prompt_text


class PromptPlanUser (Prompt):

    SYSTEM_PROMPT = """
<USER_QUERY>
{{query}}
</USER_QUERY>
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
