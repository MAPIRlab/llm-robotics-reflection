

import os

import constants
from utils import file_utils


def load_semantic_maps_basenames():
    semantic_map_basenames = list()
    for semantic_map_file in os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH):
        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)
        semantic_map_basenames.append(semantic_map_basename)
    return semantic_map_basenames


def load_queries_ids():
    queries_ids = list()
    queries_dict = file_utils.load_yaml(constants.QUERIES_FILE_PATH)
    for query_id in queries_dict["queries"]:
        queries_ids.append(query_id)
    return queries_ids


def load_ai_results():
    data = []

    semantic_map_basenames = load_semantic_maps_basenames()
    queries_ids = load_queries_ids()

    for mode in (constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY):
        for method in (constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLING):
            for llm in constants.LLM_PROVIDERS:
                for semantic_map_basename in semantic_map_basenames:
                    for query_id in queries_ids:
                        final_plan_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                            mode,
                                                            method,
                                                            llm.get_provider_name(),
                                                            semantic_map_basename,
                                                            query_id,
                                                            "final_plan.json")

                        # Save in dict
                        try:
                            response_file_content = file_utils.load_json(
                                final_plan_file_path)
                            data[mode][method][llm.get_provider_name(
                            )][semantic_map_basename][query_id] = response_file_content["relevant_objects"]
                        except FileNotFoundError or KeyError:
                            data[mode][method][llm.get_provider_name(
                            )][semantic_map_basename][query_id] = []

    return data


def load_human_results():
    data = []

    semantic_map_basenames = load_semantic_maps_basenames()

    for semantic_map_basename in semantic_map_basenames:

        ground_truth_file_path = os.path.join(constants.GROUND_TRUTH_FOLDER_PATH,
                                              f"{semantic_map_basename}.json")
        ground_truth_file_content = file_utils.load_json(
            ground_truth_file_path)

        for query_id, query_result in ground_truth_file_content["queries"].items():
            data[semantic_map_basename][query_id] = query_result

    return data


if __name__ == "__main__":

    ai_results = load_ai_results()
    human_results = load_human_results()
