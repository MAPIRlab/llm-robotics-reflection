

import argparse
import os
import time

import tqdm

import constants
from llm.conversation_history import ConversationHistory
from llm.large_language_model import LargeLanguageModel
from prompt.chooser_prompt import ChooserPrompt
from prompt.correction_prompt import (
    PromptCorrect,
    PromptCorrectAgent,
    PromptCorrectUser,
)
from prompt.planner_prompt import PromptPlan, PromptPlanAgent, PromptPlanUser
from prompt.self_reflection_prompt import (
    PromptReflect,
    PromptReflectAgent,
    PromptReflectUser,
)
from utils import file_utils, text_utils
from voxelad import preprocess


def plan_base(mode: str, semantic_map: list[tuple], llm_provider: LargeLanguageModel, queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]

    for query_id, query_text in tqdm.tqdm(queries,
                                          desc=f"Ex. {mode} {constants.METHOD_BASE} {semantic_map_basename} {llm_provider.get_provider_name()}..."):

        # Skip if exists
        output_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                        mode,
                                        constants.METHOD_BASE,
                                        llm_provider.get_provider_name(),
                                        semantic_map_basename,
                                        query_id,
                                        "final_plan.json")
        if os.path.exists(output_file_path):
            print(f"Skipping {output_file_path}...")
            continue

        conversation_history = ConversationHistory()

        # Append prompt (user)
        prompt_plan = PromptPlan(
            semantic_map=text_utils.dict_to_json_str(semantic_map_object),
            query=query_text)
        conversation_history.append_user_message(
            prompt_plan.get_prompt_text())

        # Get response
        response = llm_provider.generate_json(conversation_history)

        # Save response
        file_utils.create_directories_for_file(output_file_path)
        file_utils.save_json_str_to_file(json_str=response,
                                         output_path=output_file_path)


def plan_self_reflection(mode: str, semantic_map: list, llm_provider: LargeLanguageModel, queries: list, reflection_iterations: int):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    plan_conversation_history = ConversationHistory()
    self_reflection_conversation_history = ConversationHistory()
    correction_conversation_history = ConversationHistory()

    for query_id, query_text in tqdm.tqdm(queries,
                                          desc=f"Ex. {mode} {constants.METHOD_SELF_REFLECTION} {semantic_map_basename} {llm_provider.get_provider_name()}..."):

        # New query -> empty conversation histories
        plan_conversation_history.clear()
        self_reflection_conversation_history.clear()
        correction_conversation_history.clear()

        ##########################################
        ################## PLAN ##################
        ##########################################
        print("Planning...")
        # Append prompt (user)
        plan_conversation_history.append_user_message(
            PromptPlan(
                semantic_map=semantic_map_object_str,
                query=query_text).get_prompt_text(),
        )

        # Skip if exists
        plan_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                               mode,
                                               constants.METHOD_SELF_REFLECTION,
                                               llm_provider.get_provider_name(),
                                               semantic_map_basename,
                                               query_id,
                                               "plan_0.json")
        # Get response
        if os.path.exists(plan_response_file_path):
            print(f"Skipping {plan_response_file_path}...")
            plan_response = text_utils.dict_to_json_str(
                file_utils.load_json(plan_response_file_path))
        else:
            plan_response = llm_provider.generate_json(
                plan_conversation_history)
            file_utils.create_directories_for_file(plan_response_file_path)
            file_utils.save_json_str_to_file(json_str=plan_response,
                                             output_path=plan_response_file_path)
        # Append response to conversation history
        plan_conversation_history.append_assistant_message(plan_response)

        # Initial plan is response to be refined
        response_to_be_refined = plan_response

        # Set reflection and correction first prompts (user)
        self_reflection_conversation_history.append_user_message(PromptReflect(
            semantic_map=semantic_map_object_str).get_prompt_text())
        correction_conversation_history.append_user_message(PromptCorrect(
            semantic_map=semantic_map_object_str).get_prompt_text())

        for reflection_iteration_idx in range(reflection_iterations):

            ##########################################
            ############## SELF-REFLECT ##############
            ##########################################
            print("Reflecting...")
            # Append prompt (user)
            self_reflection_conversation_history.append_user_message(
                PromptReflectUser(
                    query=query_text,
                    plan_response=response_to_be_refined).get_prompt_text(),
            )

            # Skip if exists
            self_reflection_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                              mode,
                                                              constants.METHOD_SELF_REFLECTION,
                                                              llm_provider.get_provider_name(),
                                                              semantic_map_basename,
                                                              query_id,
                                                              f"self_reflection_{reflection_iteration_idx}.txt")
            # Get response
            if os.path.exists(self_reflection_response_file_path):
                print(f"Skipping {self_reflection_response_file_path}...")
                self_reflection_response = text_utils.dict_to_json_str(
                    file_utils.read_text_from_file(self_reflection_response_file_path))
            else:
                self_reflection_response = llm_provider.generate_text(
                    self_reflection_conversation_history)
                file_utils.create_directories_for_file(
                    self_reflection_response_file_path)
                file_utils.save_text_to_file(text=self_reflection_response,
                                             output_path=self_reflection_response_file_path)
            # Append response (assistant)
            self_reflection_conversation_history.append_assistant_message(
                self_reflection_response)

            ##########################################
            ################ CORRECT #################
            ##########################################
            print("Correcting...")
            # Append prompt (user)
            correction_conversation_history.append_user_message(
                PromptCorrectUser(
                    plan_response=response_to_be_refined,
                    self_reflection_response=self_reflection_response).get_prompt_text())

            # Skip if exists
            correction_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                         mode,
                                                         constants.METHOD_SELF_REFLECTION,
                                                         llm_provider.get_provider_name(),
                                                         semantic_map_basename,
                                                         query_id,
                                                         f"plan_{reflection_iteration_idx+1}.json")
            # Get response
            if os.path.exists(correction_response_file_path):
                print(f"Skipping {correction_response_file_path}...")
                correction_response = text_utils.dict_to_json_str(
                    file_utils.load_json(correction_response_file_path))
            else:
                correction_response = llm_provider.generate_json(
                    correction_conversation_history)
                file_utils.create_directories_for_file(
                    correction_response_file_path)
                file_utils.save_json_str_to_file(json_str=correction_response,
                                                 output_path=correction_response_file_path)
            # Append response (assistant)
            correction_conversation_history.append_assistant_message(
                correction_response)

            # New response to be refined
            response_to_be_refined = correction_response

        # Once reflection iterations finished, new set final plan
        final_plan_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                            mode,
                                            constants.METHOD_SELF_REFLECTION,
                                            llm_provider.get_provider_name(),
                                            semantic_map_basename,
                                            query_id,
                                            f"final_plan.json")
        file_utils.create_directories_for_file(
            final_plan_file_path)
        file_utils.save_json_str_to_file(json_str=correction_response,
                                         output_path=final_plan_file_path)


def plan_multiagent_reflection(mode: str, semantic_map: list, llm_provider: LargeLanguageModel, queries: list, reflection_iterations: int):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    plan_conversation_history = ConversationHistory()
    self_reflection_conversation_history = ConversationHistory()
    correction_conversation_history = ConversationHistory()

    for query_id, query_text in tqdm.tqdm(queries,
                                          desc=f"Ex. {mode} {constants.METHOD_SELF_REFLECTION} {semantic_map_basename} {llm_provider.get_provider_name()}..."):

        # New query -> empty conversation histories
        plan_conversation_history.clear()
        self_reflection_conversation_history.clear()
        correction_conversation_history.clear()

        ##########################################
        ################## PLAN ##################
        ##########################################
        print("Planning...")
        # Append prompt (system)
        plan_conversation_history.append_system_message(
            PromptPlanAgent(
                semantic_map=semantic_map_object_str).get_prompt_text(),
        )
        # Append query (user)
        plan_conversation_history.append_user_message(
            PromptPlanUser(query=query_text).get_prompt_text()
        )

        # Skip if exists
        plan_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                               mode,
                                               constants.METHOD_MULTIAGENT_REFLECTION,
                                               llm_provider.get_provider_name(),
                                               semantic_map_basename,
                                               query_id,
                                               "plan_0.json")
        # Get response
        if os.path.exists(plan_response_file_path):
            print(f"Skipping {plan_response_file_path}...")
            plan_response = text_utils.dict_to_json_str(
                file_utils.load_json(plan_response_file_path))
        else:
            plan_response = llm_provider.generate_json(
                plan_conversation_history)
            file_utils.create_directories_for_file(plan_response_file_path)
            file_utils.save_json_str_to_file(json_str=plan_response,
                                             output_path=plan_response_file_path)
        # Append response (assistant)
        plan_conversation_history.append_assistant_message(plan_response)

        # Initial plan is response to be refined
        response_to_be_refined = plan_response

        # Set reflection and correction first prompts (system)
        self_reflection_conversation_history.append_system_message(PromptReflectAgent(
            semantic_map=semantic_map_object_str).get_prompt_text())
        correction_conversation_history.append_system_message(PromptCorrectAgent(
            semantic_map=semantic_map_object_str).get_prompt_text())

        for reflection_iteration_idx in range(reflection_iterations):

            ##########################################
            ################ REFLECT #################
            ##########################################
            print("Reflecting...")
            # Append query and plan (user)
            self_reflection_conversation_history.append_user_message(
                PromptReflectUser(
                    query=query_text,
                    plan_response=response_to_be_refined).get_prompt_text(),
            )

            # Skip if exists
            self_reflection_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                              mode,
                                                              constants.METHOD_MULTIAGENT_REFLECTION,
                                                              llm_provider.get_provider_name(),
                                                              semantic_map_basename,
                                                              query_id,
                                                              f"reflection_{reflection_iteration_idx}.txt")
            # Get response
            if os.path.exists(self_reflection_response_file_path):
                print(f"Skipping {self_reflection_response_file_path}...")
                self_reflection_response = text_utils.dict_to_json_str(
                    file_utils.read_text_from_file(self_reflection_response_file_path))
            else:
                self_reflection_response = llm_provider.generate_text(
                    self_reflection_conversation_history)
                file_utils.create_directories_for_file(
                    self_reflection_response_file_path)
                file_utils.save_text_to_file(text=self_reflection_response,
                                             output_path=self_reflection_response_file_path)
            # Append response (assistant)
            self_reflection_conversation_history.append_assistant_message(
                self_reflection_response)

            ##########################################
            ################ CORRECT #################
            ##########################################
            print("Correcting...")
            # Append prompt (user)
            correction_conversation_history.append_user_message(
                PromptCorrectUser(
                    plan_response=response_to_be_refined,
                    self_reflection_response=self_reflection_response).get_prompt_text())

            # Skip if exists
            correction_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                         mode,
                                                         constants.METHOD_MULTIAGENT_REFLECTION,
                                                         llm_provider.get_provider_name(),
                                                         semantic_map_basename,
                                                         query_id,
                                                         f"plan_{reflection_iteration_idx+1}.json")
            if os.path.exists(correction_response_file_path):
                print(f"Skipping {correction_response_file_path}...")
                correction_response = text_utils.dict_to_json_str(
                    file_utils.load_json(correction_response_file_path))
            else:
                # Get response
                correction_response = llm_provider.generate_json(
                    correction_conversation_history)
                file_utils.create_directories_for_file(
                    correction_response_file_path)
                file_utils.save_json_str_to_file(json_str=correction_response,
                                                 output_path=correction_response_file_path)
            # Append response (assistant)
            correction_conversation_history.append_assistant_message(
                correction_response)

            # New response to be refined
            response_to_be_refined = correction_response

        # Once reflection iterations finished, new set final plan
        final_plan_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                            mode,
                                            constants.METHOD_SELF_REFLECTION,
                                            llm_provider.get_provider_name(),
                                            semantic_map_basename,
                                            query_id,
                                            f"final_plan.json")
        file_utils.create_directories_for_file(
            final_plan_file_path)
        file_utils.save_json_str_to_file(json_str=correction_response,
                                         output_path=final_plan_file_path)


def plan_ensembling(mode: str, semantic_map, chooser_llm_provider: LargeLanguageModel, queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    for query_id, query_text in tqdm.tqdm(queries,
                                          desc=f"Ex. {mode} {constants.METHOD_SELF_REFLECTION} {semantic_map_basename} {chooser_llm_provider.get_provider_name()}..."):

        plan_responses = list()

        # Create N LLMs
        planner_llms = [chooser_llm_provider] * 6
        planner_llms_labels = [f"{llm_provider.get_provider_name(
        )}_{llm_index}" for llm_index, llm_provider in enumerate(planner_llms)]

        # Unique conversation history
        conversation_history = ConversationHistory()

        for llm_label, llm_provider in zip(planner_llms_labels, planner_llms):

            ##########################################
            ################## PLAN ##################
            ##########################################
            print(f"Planning {llm_label}...")
            conversation_history.clear()
            # Append prompt (user)
            conversation_history.append_user_message(
                PromptPlan(
                    semantic_map=semantic_map_object_str,
                    query=query_text).get_prompt_text())

            # Get response
            plan_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                   mode,
                                                   constants.METHOD_ENSEMBLE,
                                                   chooser_llm_provider.get_provider_name(),
                                                   semantic_map_basename,
                                                   query_id,
                                                   f"plan_{llm_label}.json")
            # Skip if exists
            if os.path.exists(plan_response_file_path):
                print(f"Skipping {plan_response_file_path}...")
                # Load response
                plan_response = text_utils.dict_to_json_str(
                    file_utils.load_json(plan_response_file_path))
            else:
                # Get response
                plan_response = llm_provider.generate_json(
                    conversation_history)
                # Save response
                file_utils.create_directories_for_file(
                    plan_response_file_path)
                file_utils.save_json_str_to_file(json_str=plan_response,
                                                 output_path=plan_response_file_path)

            plan_responses.append(plan_response)

        ##########################################
        ################# CHOOSE #################
        ##########################################
        print("Choosing...")
        conversation_history.clear()

        choice_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                 mode,
                                                 constants.METHOD_ENSEMBLE,
                                                 chooser_llm_provider.get_provider_name(),
                                                 semantic_map_basename,
                                                 str(query_id),
                                                 f"choice_{len(planner_llms)}.json")
        # Append prompt (system)
        conversation_history.append_system_message(
            ChooserPrompt(llm_responses=plan_responses,
                          semantic_map=semantic_map_object_str,
                          query=query_text).get_prompt_text())

        if os.path.exists(choice_response_file_path):
            print(f"Skipping {plan_response_file_path}...")
        else:
            # Get response
            choice_response = chooser_llm_provider.generate_json(
                conversation_history)
            # Save response
            file_utils.save_json_str_to_file(json_str=choice_response,
                                             output_path=choice_response_file_path)


def print_time_statistics(start_time: float, end_time: float, method: str, number_queries: int, reflection_iterations: int):

    execution_time = end_time - start_time
    number_llm_calls = 0

    if method == constants.METHOD_BASE:
        number_llm_calls = number_queries
    elif method == constants.METHOD_SELF_REFLECTION:
        number_llm_calls = number_queries * (1 + 2 * reflection_iterations)
    elif method == constants.METHOD_MULTIAGENT_REFLECTION:
        number_llm_calls = number_queries * (1 + 2 * reflection_iterations)
    elif method == constants.METHOD_ENSEMBLE:
        number_llm_calls = number_queries * 7

    print(f"Evaluation took {execution_time} s")
    print(f"During evaluation {number_llm_calls} calls were executed, {
          execution_time/number_llm_calls} s/call")


def main(args):
    # Load llm
    llm_provider = None
    if args.llm == constants.LLM_GEMINI_1_0_PRO:
        llm_provider = constants.GEMINI_1_0_PRO
    elif args.llm == constants.LLM_GEMINI_1_5_PRO:
        llm_provider = constants.GEMINI_1_5_PRO

    # Load and pre-process semantic map
    semantic_maps = list()
    for semantic_map_file in os.listdir(constants.SEMANTIC_MAPS_FOLDER_PATH):

        semantic_map_basename = file_utils.get_file_basename(semantic_map_file)

        # Load semantic map
        semantic_map_obj = file_utils.load_json(os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                                             semantic_map_file))

        semantic_maps.append((semantic_map_basename, semantic_map_obj))

    # Load queries
    queries = list()
    queries_dict = file_utils.load_yaml(constants.QUERIES_FILE_PATH)
    for query_id in queries_dict["queries"]:
        queries.append((query_id, queries_dict["queries"][query_id]))

    # MAIN LOOP
    for (s_m_b, s_m_o) in semantic_maps[:args.number_maps]:

        start_time = time.time()
        # Pre-process semantic map
        pre_processed_semantic_map = (s_m_b, preprocess.preprocess_semantic_map(s_m_o,
                                                                                class_uncertainty=(args.mode == constants.MODE_UNCERTAINTY)))

        # Plan actions for every method
        if args.method == constants.METHOD_BASE:
            plan_base(args.mode, pre_processed_semantic_map,
                      llm_provider, queries)
        elif args.method == constants.METHOD_SELF_REFLECTION:
            plan_self_reflection(
                args.mode, pre_processed_semantic_map, llm_provider, queries, args.reflection_iterations)
        elif args.method == constants.METHOD_MULTIAGENT_REFLECTION:
            plan_multiagent_reflection(
                args.mode, pre_processed_semantic_map, llm_provider, queries, args.reflection_iterations)
        elif args.method == constants.METHOD_ENSEMBLE:
            plan_ensembling(args.mode, pre_processed_semantic_map,
                            llm_provider, queries)

        end_time = time.time()

        print_time_statistics(start_time, end_time,
                              number_queries=len(queries),
                              method=args.method,
                              reflection_iterations=args.reflection_iterations)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TODO")  # TODO

    parser.add_argument("-n", "--number-maps",
                        help="Number of semantic maps to be processed by the pipeline",
                        type=int,
                        default=10)

    parser.add_argument("--mode",
                        help="Semantic maps input mode to LLMs, with uncertainty (uncertainty) or not (certainty)",
                        type=str,
                        choices=[constants.MODE_CERTAINTY,
                                 constants.MODE_UNCERTAINTY],
                        default=constants.MODE_CERTAINTY)

    parser.add_argument("--method",
                        type=str,
                        help="Which models from the ones tested to use",
                        choices=[constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLE])

    # in METHODS base, self_reflection and multiagent_reflection represents the main LLM
    # in METHOD ensembling represents the chooser LLM
    parser.add_argument("-l", "--llm",
                        type=str,
                        help="Which LLM to use: 0 -> Gemini 1.0 Pro; 1 -> Gemini 1.5 Pro",
                        choices=[constants.LLM_GEMINI_1_0_PRO, constants.LLM_GEMINI_1_5_PRO])

    # (only for METHODs self_reflection and multiagent_reflection)
    parser.add_argument("-i", "--reflection-iterations",
                        help="Number of reflection iterations",
                        type=int,
                        default=2)

    args = parser.parse_args()

    main(args)
