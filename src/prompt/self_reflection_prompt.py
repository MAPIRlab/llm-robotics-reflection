
from prompt.prompt import Prompt

# EXAMPLE 1: spatial reasoning + malformed JSON
EXAMPLE_1 = """
<EXAMPLE_1>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[2.0,3.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":100,"results":{"book":80}},
"obj2":{"bbox":{"center":[2.0,3.0,0.0],"size":[2.0,1.0,0.5]},"n_observations":50,"results":{"dining table":50}},
"obj3":{"bbox":{"center":[5.0,5.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":80,"results":{"chair":70}},
"obj4":{"bbox":{"center":[4.0,4.0,0.5],"size":[0.5,0.5,0.2]},"n_observations":90,"results":{"book":85}}}}
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
            "object": "obj4 (book)",
            "relevance": "highly relevant"
        }
    ],
    "explanation": "The most relevant object is 'obj4', classified as a book, which is located on 'obj2' (dining table) based on their coordinates."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 0/10
- Relevant objects: Error. It is true that obj4 is a book, but it is false that obj4 is on the table, because of its bounding box center coordinates. Instead, obj1 is a book with high confidence and it is located on [2.0,3.0,0.5], which seems a to be on top of obj2 that is a dining table with a high score placed at [2.0,3.0,0.0].
- Explanation: Error. As the relevant objects are incorrectly identified, the explanation is wrong.
- JSON structure 2: Error. The preliminary response does not follow the required JSON structure; "relevant_objects" should be a list of object IDs as strings, not a list of dictionaries. This error makes the answer invalid, so the score is 0.
2. Comments on Relevance. Score: 4/10
- Identification: Error. The object identified for the task is wrong.
- Leftovers: obj4 should not belong to the response.
3. Comments on Clarity. Score: 7/10
4. Actions to improve response:
- Change "relevant_objects" to ["obj1"]. DO NOT INCLUDE THE OBJECT CLASS IN THE LIST!
- Update the "explanation" accordingly.
- Adjust the JSON structure to match the required format, where "relevant_objects" is a list of object IDs as strings.
</EXAMPLE_1>
"""


# EXAMPLE 2: correcting order
EXAMPLE_2 = """
<EXAMPLE_2>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,2,0],"size":[1,1,1]},"n_observations":100,"results":{"chair":90}},
"obj2":{"bbox":{"center":[2,2,0],"size":[2,1,1.5]},"n_observations":80,"results":{"sofa":85}},
"obj3":{"bbox":{"center":[3,2,0],"size":[1.5,1,0.5]},"n_observations":70,"results":{"bench":75}},
"obj4":{"bbox":{"center":[4,2,0],"size":[2,1.5,0.5]},"n_observations":60,"results":{"chair":150}},
"obj5":{"bbox":{"center":[5,2,0],"size":[1,0.5,0.5]},"n_observations":50,"results":{"stool":65}}}}
</SEMANTIC_MAP>

<QUERY>
I need to sit down while on my laptop
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Find objects suitable for sitting and working on a laptop",
    "query_achievable": true,
    "relevant_objects": ["obj5", "obj3", "obj4", "obj2", "obj1"],
    "explanation": "Objects obj5, obj3, obj2, and obj1 can be used to sit while working on your laptop."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 4/10
- Relevant objects: Partially correct. The objects  obj4 (chair), obj1 (chair), obj2 (sofa), obj3 (bench), and obj5 (stool) are all seating options. However, for working on a laptop, obj4, obj1, and obj2 are more suitable due to comfort and ergonomics. The preliminary response includes all seating objects but doesn't prioritize them correctly. Additionally, between obj4 and obj1, obj4 should be prioritized since it has greater certainty in the semantic map, 150 (obj4) compared to 90 (obj1).
- Explanation: The explanation mentions the objects but fails to justify their relevance or prioritize them according to suitability.
2. Comments on Relevance. Score: 3/10
- Identification: Relevant objects are identified but not ordered by their suitability for the task.
- Sorting: Objects are not sorted by relevance; less suitable options like obj5 (stool) and obj3 (bench) are listed before more suitable ones like obj1 (chair) and obj2 (sofa).
- Ties: obj4 should be before obj1 because both of them are equally convenient, but obj4 presents a higher certainty.
- Details: The response lacks details on why certain objects are more appropriate than others.
3. Comments on Clarity. Score: 8/10
4. Actions to Improve Response:
- Reorder the "relevant_objects" list to prioritize the most suitable objects for working on a laptop, listing first obj4, then obj1 and then obj2 first.
- Enhance the "explanation" to justify the ordering, highlighting why certain objects are more appropriate for the task.
</EXAMPLE_2>
"""

# EXAMPLE 3: functionality extension 1
EXAMPLE_3 = """
<EXAMPLE_3>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,2,0],"size":[0.5,0.5,1]},"n_observations":100,"results":{"trash can":90}},
"obj2":{"bbox":{"center":[2,2.5,0],"size":[0.8,0.8,1.2]},"n_observations":80,"results":{"toilet":85}},
"obj3":{"bbox":{"center":[3,3,0],"size":[1,1,1]},"n_observations":70,"results":{"refrigerator":75}},
"obj4":{"bbox":{"center":[4,4,0],"size":[0.5,0.5,0.5]},"n_observations":60,"results":{"flower":80}}}}
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
- Relevant objects: Error. Objects like obj1 (trash can) and obj2 (toilet) are commonly associated with bad smells and should be included.
- Explanation: Error. The explanation overlooks indirectly relevant objects that could fulfill the user's request.
2. Comments on Relevance. Score: 5/10
- Identification: Error. The response fails to identify objects indirectly related to the query.
- Details: Lacking. Additional details on why certain objects are relevant are missing.
3. Comments on Clarity. Score: 8/10
- Clear: The response is clear but incomplete.
4. Actions to Improve Response:
- Change "query_achievable" to true since the task can be achieved with existing objects.
- Update "inferred_query" to "Find objects that may smell bad" to encompass indirectly related items.
- Include "relevant_objects": ["obj2", "obj1"], representing the trash can and the toilet. obj2 should be placed first as toilets have more probability of smelling bad than trash cans (obj1).
- Revise the "explanation" to state that while no objects are explicitly labeled as smelling bad, these objects are associated with unpleasant odors and may satisfy the query.
</EXAMPLE_3>
"""

# EXAMPLE 4: no response
EXAMPLE_4 = """
<EXAMPLE_4>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,2,0.5],"size":[1,0.5,0.5]},"n_observations":100,"results":{"sink":90}},
"obj4":{"bbox":{"center":[3,3,0.5],"size":[1,0.5,0.5]},"n_observations":60,"results":{"sink":80}},
"obj5":{"bbox":{"center":[5,5,0],"size":[1,1,1]},"n_observations":50,"results":{"bed":65}}}}
</SEMANTIC_MAP>

<QUERY>
I need a refrigerator that is next to a sink
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify a refrigerator that is close to a sink",
    "query_achievable": false,
    "relevant_objects": [],
    "explanation": "There is no refrigerator on the map"
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 9/10
2. Comments on Relevance. Score: 10/10
3. Comments on Clarity. Score: 10/10
4. Actions to Improve Response:
- Extend the explanation by mentioning that, although there are sinks on the map, there are no refrigerators, so the query is not achievable.
</EXAMPLE_4>"""

# EXAMPLE 5: functionality extension 2
EXAMPLE_5 = """
<EXAMPLE_5>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[2,1,0.5],"size":[0.5,0.5,0.5]},"n_observations":85,"results":{"tv":90}},
"obj2":{"bbox":{"center":[3,1,0.5],"size":[0.2,0.2,0.3]},"n_observations":134,"results":{"laptop":80}},
"obj3":{"bbox":{"center":[5,5,0],"size":[1,1,1]},"n_observations":50,"results":{"bed":65}}}}
</SEMANTIC_MAP>

<QUERY>
I need to see the news
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify an object that can display the news",
    "query_achievable": false,
    "relevant_objects": [],
    "explanation": "The objects present in the scene are not typically used for showing the news."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 4/10
- Query achievable: Error, the query is achievable objects that 'may' display the news.
- Relevant objects: Error, objects like obj1 (tv) and obj2 (laptop) are usually capable of displaying up-to-date news.
- Explanation: Error, the explanation overlooks indirectly relevant objects that could fulfill the user's request.
2. Comments on Relevance. Score: 5/10
- Identification: Error. The response fails to identify objects indirectly related to the query.
- Sorting: In this case, the order of the objects should be as follows: obj2 (laptop) and then obj1 (tv). Laptops are more convenient objects to watch the news as you can search for a news page whenever you want, while on TV the news is broadcasted at a certain time.
- Details: Additional details on why certain objects are relevant are missing.
3. Comments on Clarity. Score: 8/10
4. Actions to Improve Response:
- Change "query_achievable" to true since the task can be achieved with existing objects.
- Include "relevant_objects": ["obj2", "obj1"], representing the trash can and the toilet. obj2 should be placed first as toilets have more probability of smelling bad than trash cans (obj1).
- Revise the "explanation" to state that there are objects related to the function of displaying the news, obj2 (a laptop) and obj1 (a tv), and include some details.
</EXAMPLE_5>"""

# EXAMPLE 6: don't imagine objects
EXAMPLE_6 = """
<EXAMPLE_6>
<SEMANTIC_MAP>
{"instances":{
"obj1":{"bbox":{"center":[1,2,0.5],"size":[0.6,0.6,0.8]},"n_observations":100,"results":{"washing machine":95}},
"obj2":{"bbox":{"center":[2,3,0.5],"size":[0.6,0.6,0.8]},"n_observations":120,"results":{"washing machine":90}},
"obj3":{"bbox":{"center":[4,5,0.3],"size":[0.4,0.4,0.5]},"n_observations":75,"results":{"chair":85}}}}
</SEMANTIC_MAP>

<QUERY>
Where can I dry my T-shirt?
</QUERY>

<PRELIMINARY_RESPONSE>
{
    "inferred_query": "Identify a place to dry a T-shirt",
    "query_achievable": true,
    "relevant_objects": [
        "obj1",
        "obj2",
        "obj3",
    ],
    "explanation": "While there's no dedicated 'clothesline' identified, the presence of washing machines ('obj1', 'obj2') suggests a potential clothesline location nearby. A chair (obj3) is another reasonable places to dry a T-shirt temporarily, but a clothesline would be the ideal location."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 6/10
- Relevant objects: Error, there are objects marked as relevant that are not really convenient for the task at hand.
- Explanation: Error, the explanation confuses which object is really relevant to perform the task (obj3), imagining that there will be an ideal object between two other objects in the semantic map (obj1 and obj2).
2. Comments on Relevance. Score: 4/10
- Sorting: In this case, you can not imagine that there will be a clothesline near two washing machines (obj1 and obj2), so they shouldn't be in the "relevant_objects" section. Instead, obj3 is a chair, where a T-shirt could be temporally placed to be dried.
- Details: Additional details on this reasoning is needed.
3. Comments on Clarity. Score: 7/10
4. Actions to Improve Response:
- Change "relevant_objects" to ["obj3"], as it is a chair where a  T-shirt could be temporally placed to be dried.
- Revise the "explanation" to mention that between the two washing machines a clothesline could be found, without including these objects in the "relevant_objects" list, as we are not sure if a clothesline is present in the semantic map.
</EXAMPLE_6>"""

# EXAMPLE 7 = object differentiation by surroundings
EXAMPLE_7 = """
<EXAMPLE_7> 
<SEMANTIC_MAP> 
{"instances":{
"obj1":{"bbox":{"center":[1,2,0.5],"size":[1,0.5,2]},"n_observations":80,"results":{"cabinet":95}},
"obj2":{"bbox":{"center":[1,3,0.5],"size":[0.7,0.7,0.9]},"n_observations":60,"results":{"toilet":98}},
"obj3":{"bbox":{"center":[5,5,0.5],"size":[1,0.5,2]},"n_observations":90,"results":{"cabinet":92}},
"obj4":{"bbox":{"center":[5,6,0.5],"size":[1,1,2.5]},"n_observations":85,"results":{"refrigerator":97}}}}
</SEMANTIC_MAP>

<QUERY> 
Where is the kitchen cabinet? 
</QUERY>

<PRELIMINARY_RESPONSE> 
{
  "inferred_query": "Locate the kitchen cabinet",
  "query_achievable": true,
  "relevant_objects": [
    "obj1"
  ],
  "explanation": "obj1 is identified as a cabinet and is the closest match to the query."
}
</PRELIMINARY_RESPONSE>

EXPECTED RESPONSE:
1. Comments on Correctness. Score: 5/10
- Relevant Objects: Error, the selected object (obj1) is near obj2 (a toilet), suggesting it's in a bathroom, not a kitchen.
- Explanation: Error, no contextual clues were considered to correctly identify the kitchen cabinet.
2. Comments on Relevance. Score: 4/10
- Sorting: obj3 should have been prioritized due to its proximity to the refrigerator, which is typically found in a kitchen.
- Details: Additional reasoning about the surrounding objects would improve accuracy.
3. Comments on Clarity. Score: 7/10
- Clarity: The explanation is clear but based on incorrect assumptions.
4. Actions to Improve Response:
- Update "relevant_objects": Change to ["obj3"] since it is adjacent to the refrigerator, indicating it is likely the kitchen cabinet.
- Revise "explanation": Mention that obj3 is near the refrigerator, a common kitchen appliance, making it the appropriate choice.
</EXAMPLE_7> """

SELF_REFLECTION_EXAMPLES = f"""
{EXAMPLE_1}
{EXAMPLE_2}
{EXAMPLE_4}
{EXAMPLE_5}
{EXAMPLE_6}
{EXAMPLE_7}
"""


class PromptReflect(Prompt):

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

<EVALUATION_SUGGESTIONS>
Here are some suggestions on how to perform the feedback:
- If the query mention specific objects in the scene, the preliminary answer should contain such objects that are present in the scene, it should not be assumed that there are objects that are not present in the map. In the same way, it should not assume that a similar object is the one that the query mentions (e.g. if the query mentions a TV, it wouldn't be correct to consider a laptop).
- If the query requests objects to perform a task, the object response may incorporate objects that “may” be related to that task.
- Assume that the semantic map received is the absolute truth, at no time can there be elements that have been misclassified. 
</EVALUATION_SUGGESTIONS>

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
Here are some examples of the process. Each example contains a false semantic map for the example, a user query, a preliminary response on that semantic map and query, and your expected response.
{{examples}}
</EXAMPLES>

Now you will receive QUERIES and their PRELIMINARY_RESPONSES, and your task is to generate feedback on them, taking into account the real SEMANTIC_MAP.

<SEMANTIC_MAP>
Forget about all the semantic maps you have seen so far. Those were just examples. From now on this will be the semantic map:
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
- Identification: Are all relevant objects identified?
- Sorting: Are all relevant objects sorted by relevance accurately?
- Ties: In case of ties for equally convenient objects, are they ordered by certainty?
- Details: Are any crucial objects or details omitted?
- Leftovers: Are there non-relevant objects in the answer, which should be omitted?
Clarity:
- Clear: Is the response clear and easy to understand?
- Ambiguities: Are there any ambiguities or vague descriptions?
</EVALUATION_CRITERIA>

<EVALUATION_SUGGESTIONS>
Here are some suggestions on how to perform the feedback:
1. If the query mention specific objects in the scene:
- The feedback should ensure that only the objects requested by the user are in the response. The feedback should make sure that no objects are invented or imagined, since the user has requested one in question.
- In the same way, the feedback should ensure that no objects similar or similar to those requested by the user are included. The user asked for a specific one and if it is not in the scene then the feedback should be empty.
2. The feedback can suggest objects that “may” be related to the task, even if they are not specifically made for it.

These previous rules are crucial to avoid false positives.
</EVALUATION_SUGGESTIONS>

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
Here are some examples of the process. Each example contains a false semantic map for the example, a user query, a preliminary response on that semantic map and query, and your expected response.
{{examples}}
</EXAMPLES>

Now you will receive QUERIES and their PRELIMINARY_RESPONSES, and your task is to generate feedback on them, taking into account the real SEMANTIC_MAP.

<SEMANTIC_MAP>
Forget about all the semantic maps you have seen so far. Those were just examples. From now on this will be the semantic map:
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
