
from prompt.prompt import Prompt


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

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>

<USER_QUERY>
{{query}}
</USER_QUERY>

Now respond to the user query following the INSTRUCTION and the OUTPUT_FORMAT, based on the input SEMANTIC_MAP.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


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

<SEMANTIC_MAP>
The semantic map in JSON format is the following:
{{semantic_map}}
</SEMANTIC_MAP>

Now you will receive user queries and respond to them following the ROLE and the OUTPUT_FORMAT, based on the input SEMANTIC_MAP.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)


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
