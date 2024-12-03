

import argparse
import os

import pandas as pd
import tqdm

import constants
from compare.comparison_result import ComparisonResult
from results import chart_semantic_map_complexity, table_workflows_comparison
from utils import file_utils


def load_semantic_maps_basenames():
    semantic_map_basenames = list()
    for semantic_map_file in os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH):
        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)
        semantic_map_basenames.append(semantic_map_basename)
    return semantic_map_basenames


def load_semantic_maps_sizes():
    semantic_map_number_objects = list()
    for semantic_map_file in os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH):
        semantic_map_object = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                                semantic_map_file))
        semantic_map_number_objects.append(
            len(semantic_map_object["instances"]))
    return semantic_map_number_objects


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

    for mode in tqdm.tqdm((constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY),
                          desc="Loading AI results..."):
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
                            # print(f"Skipped answer: {
                            #       mode}\\{method}\\{llm.get_provider_name()}\\{semantic_map_basename}\\{query_id}: {e}")
                            data[mode][method][llm.get_provider_name(
                            )][semantic_map_basename][query_id] = None
                            n_not_loaded_responses += 1

    if n_not_loaded_responses != 0:
        print(f"WARNING: {
              n_not_loaded_responses}/{n_total_responses} responses could not be loaded")

    return data


def load_human_results(semantic_map_basenames: list):

    data = dict()

    for semantic_map_basename in tqdm.tqdm(semantic_map_basenames,
                                           desc="Loading human results..."):
        data[semantic_map_basename] = dict()

        ground_truth_file_path = os.path.join(constants.GROUND_TRUTH_FOLDER_PATH,
                                              f"{semantic_map_basename}.json")
        ground_truth_file_content = file_utils.load_json(
            ground_truth_file_path)

        for query_id, query_result in ground_truth_file_content["responses"].items():
            data[semantic_map_basename][query_id] = query_result

    return data


def compare_human_ai_results(ai_result, human_result):

    if human_result is None or not isinstance(human_result, list):
        raise ValueError("Human result is not in the correct format")
    elif ai_result is None or not isinstance(ai_result, list):
        return ComparisonResult.no_hit()
    elif len(ai_result) == 0 and len(human_result) == 0:  # ai empty, human empty
        return ComparisonResult.top_1_hit()
    elif len(ai_result) == 0 and len(human_result) != 0:  # ai empty, human not empty
        return ComparisonResult.no_hit()
    elif len(ai_result) >= 1 and ai_result[0] in human_result:
        return ComparisonResult.top_1_hit()
    elif len(ai_result) >= 2 and ai_result[1] in human_result:
        return ComparisonResult.top_2_hit()
    elif len(ai_result) >= 3 and ai_result[2] in human_result:
        return ComparisonResult.top_3_hit()
    elif any(obj in ai_result for obj in human_result):
        return ComparisonResult.top_any_hit()
    else:
        return ComparisonResult.no_hit()


def compute_all_comparison_results(semantic_map_basenames: list, queries_ids: list, ai_results: dict, human_results: dict, reflection_iterations: int):

    # pprint.pprint(ai_results)
    # pprint.pprint(human_results)

    all_comparison_results = dict()

    for mode in (constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY):
        all_comparison_results[mode] = dict()

        for method in (constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLING):
            all_comparison_results[mode][method] = dict()

            for llm in constants.LLM_PROVIDERS:
                all_comparison_results[mode][method][llm.get_provider_name(
                )] = dict()

                for semantic_map_basename in semantic_map_basenames:
                    all_comparison_results[mode][method][llm.get_provider_name(
                    )][semantic_map_basename] = dict()

                    for query_id in queries_ids:

                        ai_result = ai_results[mode][method][llm.get_provider_name(
                        )][semantic_map_basename][query_id]
                        human_result = human_results[semantic_map_basename][query_id]

                        comparison_result = compare_human_ai_results(
                            ai_result, human_result)
                        all_comparison_results[mode][method][llm.get_provider_name(
                        )][semantic_map_basename][query_id] = comparison_result

                        if mode == "certainty" and method == "ensembling" and semantic_map_basename == "scannet_scene0000_00.json":
                            print(method)
                            print("#"*100)
                            print("human_result", human_result)
                            print("ai_result", ai_result)
                            print("comparison_result", comparison_result)

    return all_comparison_results


def compute_all_comparison_results_df(all_comparison_results):

    # Flatten the nested dictionary into rows
    flattened_data = []

    for mode, mode_data in all_comparison_results.items():
        for method, method_data in mode_data.items():
            for llm, llm_data in method_data.items():
                for semantic_map_basename, basename_data in llm_data.items():
                    for query_id, comparison_result in basename_data.items():
                        # Add a row for each combination of keys
                        flattened_data.append({
                            'Mode': mode,
                            'Method': method,
                            'LLM': llm,
                            'SemanticMap': semantic_map_basename,
                            'QueryID': query_id,
                            'ComparisonResult': comparison_result
                        })

    # Convert to a DataFrame
    df = pd.DataFrame(flattened_data)

    return df


def show_reflection_errors(semantic_map_basenames, queries_ids, all_comparison_results):

    self_reflection_errors = []
    multiagent_reflection_errors = []

    for mode in (constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY):
        for llm in constants.LLM_PROVIDERS:
            for semantic_map_basename in semantic_map_basenames:
                for query_id in queries_ids:

                    base_cr = all_comparison_results[mode]["base"][llm.get_provider_name(
                    )][semantic_map_basename][query_id]
                    self_reflection_cr = all_comparison_results[mode][constants.METHOD_SELF_REFLECTION][llm.get_provider_name(
                    )][semantic_map_basename][query_id]
                    multiagent_reflection_cr = all_comparison_results[mode][constants.METHOD_MULTIAGENT_REFLECTION][llm.get_provider_name(
                    )][semantic_map_basename][query_id]

                    if base_cr > self_reflection_cr:
                        self_reflection_errors.append(
                            f"{semantic_map_basename}/{query_id}")

                    if base_cr > multiagent_reflection_cr:
                        multiagent_reflection_errors.append(
                            f"{semantic_map_basename}/{query_id}")

    print(f"SELF_REFLECTION made {len(self_reflection_errors)} errors")
    for error in self_reflection_errors:
        print(f"\t{error}")
    print(f"MULTIAGENT_REFLECTION made {
          len(multiagent_reflection_errors)} errors")
    for error in multiagent_reflection_errors:
        print(f"\t{error}")


def main(args):

    semantic_map_basenames = load_semantic_maps_basenames()[:args.number_maps]
    semantic_map_sizes = load_semantic_maps_sizes()[:args.number_maps]
    queries_ids = load_queries_ids()

    ai_results = load_ai_results(
        args.reflection_iterations, semantic_map_basenames, queries_ids)

    human_results = load_human_results(semantic_map_basenames)

    all_comparison_results = compute_all_comparison_results(
        semantic_map_basenames, queries_ids, ai_results, human_results, args.reflection_iterations)

    # Create pandas dataframe
    all_comparison_results_df = compute_all_comparison_results_df(
        all_comparison_results)

    ###################################################
    ############### REFLECTION ERRORS #################
    ###################################################
    if args.evaluation == constants.EVALUATION_REFLECTION_ERRORS:
        show_reflection_errors(semantic_map_basenames,
                               queries_ids, all_comparison_results)

    ###################################################
    ########## (TABLE) WORKFLOWS COMPARISON ###########
    ###################################################
    if args.evaluation == constants.EVALUATION_TABLE_WORKFLOWS:
        table_workflows_comparison_generator = table_workflows_comparison.TableWorkflowComparisonGenerator(
            all_comparison_results_df, mode=args.mode)
        table_1_df = table_workflows_comparison_generator.generate_table()
        print(table_1_df.to_string(index=False))

    ###################################################
    ######## (CHART) NUMBER OF OBJECTS IMPACT #########
    ###################################################
    if args.evaluation == constants.EVALUATION_CHART_SIZES:
        chart_generator = chart_semantic_map_complexity.ChartSemanticMapComplexityGenerator(
            df_comparison_results=all_comparison_results_df,
            semantic_map_basenames=semantic_map_basenames,
            semantic_map_sizes=semantic_map_sizes,
            llm_label="Google_gemini-1.5-pro",
            mode="certainty")
        chart_generator.generate_chart(metric='top_any')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for evaluating the results")  # TODO

    parser.add_argument("-e", "--evaluation",
                        type=str,
                        help="Evaluation results to show",
                        choices=[constants.EVALUATION_TABLE_WORKFLOWS,
                                 constants.EVALUATION_CHART_SIZES, constants.EVALUATION_REFLECTION_ERRORS],
                        required=True)

    parser.add_argument("-n", "--number-maps",
                        type=int,
                        default=10,
                        help="TODO")  # TODO

    parser.add_argument("--mode",
                        type=str,
                        help="TODO",  # TODO
                        choices=[constants.MODE_CERTAINTY,
                                 constants.MODE_UNCERTAINTY],
                        default="certainty")

    # (only for METHODs self_reflection and multiagent_reflection)
    parser.add_argument("-i", "--reflection-iterations",
                        type=int,
                        help="Number of reflection iterations",
                        default=2)

    args = parser.parse_args()

    main(args)
