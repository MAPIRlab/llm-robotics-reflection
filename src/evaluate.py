

import argparse
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


def load_ai_results(reflection_iterations: int, semantic_map_basenames: list, queries_ids: list):
    data = dict()
    n_not_loaded_responses = 0
    n_total_responses = 2 * 4 * 2 * 10 * 30  # TODO: include variables here

    for mode in (constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY):
        data[mode] = dict()

        for method in (constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLING):
            data[mode][method] = dict()

            for llm in constants.LLM_PROVIDERS:
                data[mode][method][llm.get_provider_name()] = dict()

                for semantic_map_basename in semantic_map_basenames:
                    data[mode][method][llm.get_provider_name(
                    )][semantic_map_basename] = dict()

                    for query_id in queries_ids:
                        try:
                            # Get final response file path
                            query_results_folder_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                                     mode,
                                                                     method,
                                                                     llm.get_provider_name(),
                                                                     semantic_map_basename,
                                                                     query_id)

                            if method == constants.METHOD_BASE:
                                final_plan_file_path = os.path.join(query_results_folder_path,
                                                                    "final_plan.json")

                            elif method in (constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION):
                                final_plan_file_path = os.path.join(query_results_folder_path,
                                                                    f"plan_{reflection_iterations}.json")

                            elif method == constants.METHOD_ENSEMBLING:
                                # Get choice
                                choice_file_paths = file_utils.find_matching_files(
                                    query_results_folder_path, r"choice_\d+\.json")

                                if len(choice_file_paths) == 0:
                                    raise ValueError(
                                        "Not found 'choice_x.json' file")

                                choice_file_path = choice_file_paths[0]

                                # Get chosen response
                                choice = file_utils.load_json(choice_file_path)
                                chosen_response_idx = int(
                                    choice["chosen_response"])

                                final_plan_file_paths = file_utils.find_matching_files(
                                    query_results_folder_path, fr"plan_[a-zA-Z0-9\.\-_]+_{chosen_response_idx}\.json")

                                if len(final_plan_file_paths) == 0:
                                    raise ValueError(
                                        "Not found 'plan_x_y.json' file")

                                final_plan_file_path = final_plan_file_paths[0]

                            if not os.path.exists(final_plan_file_path):
                                raise ValueError(
                                    f"Path {final_plan_file_path} not found")

                            # Load content from final_plan_file_path
                            response_file_content = file_utils.load_json(
                                final_plan_file_path)

                            if not isinstance(response_file_content, dict):
                                raise ValueError(
                                    "Response is not a dict")
                            elif "relevant_objects" not in response_file_content:
                                raise ValueError(
                                    "Response does not contain 'relevant_objects'")
                            elif not isinstance(response_file_content["relevant_objects"], list):
                                raise ValueError(
                                    "Response's 'relevant_objects' is not a list")

                            data[mode][method][llm.get_provider_name(
                            )][semantic_map_basename][query_id] = response_file_content["relevant_objects"]

                        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
                            print(f"Skipped answer: {
                                  mode}\\{method}\\{llm.get_provider_name()}\\{semantic_map_basename}\\{query_id}: {e}")
                            data[mode][method][llm.get_provider_name(
                            )][semantic_map_basename][query_id] = []
                            n_not_loaded_responses += 1

    if n_not_loaded_responses != 0:
        print(f"WARNING: {
              n_not_loaded_responses}/{n_total_responses} responses could not be loaded")

    return data


def load_human_results(semantic_map_basenames: list):

    data = dict()

    for semantic_map_basename in semantic_map_basenames:
        data[semantic_map_basename] = dict()

        ground_truth_file_path = os.path.join(constants.GROUND_TRUTH_FOLDER_PATH,
                                              f"{semantic_map_basename}.json")
        ground_truth_file_content = file_utils.load_json(
            ground_truth_file_path)

        for query_id, query_result in ground_truth_file_content["responses"].items():
            data[semantic_map_basename][query_id] = query_result

    return data


def compare_human_ai_results(ai_result, human_result):
    pass


def main(args):

    semantic_map_basenames = load_semantic_maps_basenames()
    queries_ids = load_queries_ids()

    ai_results = load_ai_results(
        args.reflection_iterations, semantic_map_basenames, queries_ids)
    human_results = load_human_results(semantic_map_basenames)

    # pprint.pprint(ai_results)
    # pprint.pprint(human_results)

    # for mode in (constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY):
    #     for method in (constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLING):
    #         for llm in constants.LLM_PROVIDERS:
    #             for semantic_map_basename in semantic_map_basenames:
    #                 for query_id in queries_ids:

    #                     ai_result = ai_results[mode][method][llm][semantic_map_basename][query_id]
    #                     human_result = human_results[semantic_map_basename][query_id]

    #                     compare_human_ai_results(ai_result, human_result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TODO")  # TODO

    # (only for METHODs self_reflection and multiagent_reflection)
    parser.add_argument("-i", "--reflection-iterations",
                        type=int,
                        help="Number of reflection iterations",
                        default=2)

    args = parser.parse_args()

    main(args)
