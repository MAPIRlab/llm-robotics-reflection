
from prompt.prompt import Prompt


class PromptPlan (Prompt):

    SYSTEM_PROMPT = """
<INSTRUCTION>
The input to the model is a 3D semantic map in a JSON format (<SEMANTIC_MAP>). Each entry of the "instances" JSON object describes one object in the scene, with the following fields:

1 "bbox": 3D bounding box of the object
    1.1 "bbox.center": Center of the bounding box of the object
    1.2 "bbox.size": Size of the bounding box of the object
2 "n_observations": Number of observations of the object in the scene
3 "results": Results of the classification of the object, indicating a category and the certainty that the object belongs to that category

Read and parse the JSON in order to answer users queries about the scene, indicating the list of most related objects to perform the task.
This list should be sorted according to relevance, with the most relevant object first and the least relevant object last.
To answer the user's queries, you must not only take into account the classification result of each object, but also the arrangement of these objects in the semantic map.
Although the relationships do not come in the semantic map the scene layout can be useful to see which objects are next to, above or below others.
</INSTRUCTION>

<OUTPUT_FORMAT>
For each user query, respond with a JSON dictionary with the following fields:
1 "inferred_query": (String) Your interpretation of the user query in summary form
2 "query_achievable": (Boolean) Whether or not the user-specified query is achievable using the objects and descriptions provided in the semantic map
3 "relevant_objects": (List of String) List of objects relevant to the user's query (ordered by relevance, most relevant first, least relevant last); or empty list in case there is no relevant object. Objects must be represented here by their ids.
4 "explanation": (String) A brief explanation of what the most relevant objects are, and how they achieve the user-specified task
</OUTPUT_FORMAT>

<SEMANTIC_MAP>
{{semantic_map}}
</SEMANTIC_MAP>

You will now receive queries from the user, which you have to answer following the INSTRUCTION and the OUTPUT_FORMAT, based on the input SEMANTIC_MAP.
"""

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def global_replace(self, prompt_text: str) -> str:
        return self.replace_prompt_data_dict(self.prompt_data_dict, prompt_text)
