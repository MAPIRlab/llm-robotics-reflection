

import argparse
import os

import tqdm

import constants
from llm.conversation_history import ConversationHistory
from prompt.chooser_prompt import ChooserPrompt
from prompt.correction_prompt import PromptCorrect, PromptCorrectUser
from prompt.planner_prompt import PromptPlan
from prompt.self_reflection_prompt import PromptReflect, PromptReflectUser
from utils import file_utils, text_utils
from voxelad import preprocess


def plan_base(mode: str, semantic_map: list[tuple], queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]

    for llm_provider in (constants.GEMINI_1_0_PRO, constants.GEMINI_1_5_PRO):

        for query_id, query_text in tqdm.tqdm(queries,
                                              desc=f"Planning {constants.METHOD_BASE} {semantic_map_basename} {llm_provider.get_provider_name()}..."):

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


def plan_self_reflection(mode: str, semantic_map: list, queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    for llm_provider in (constants.GEMINI_1_0_PRO, constants.GEMINI_1_5_PRO):

        plan_conversation_history = ConversationHistory()
        self_reflection_conversation_history = ConversationHistory()
        correction_conversation_history = ConversationHistory()

        for query_id, query_text in tqdm.tqdm(queries,
                                              desc=f"Planning {constants.METHOD_SELF_REFLECTION} {semantic_map_basename} {llm_provider.get_provider_name()}..."):

            # New query -> empty conversation histories
            plan_conversation_history.clear()
            self_reflection_conversation_history.clear()
            correction_conversation_history.clear()

            ##########################################
            ################## PLAN ##################
            ##########################################
            print("planning")
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

            # Set reflection and correction first prompts *user()
            self_reflection_conversation_history.append_user_message(PromptReflect(
                semantic_map=semantic_map_object_str).get_prompt_text())
            correction_conversation_history.append_user_message(PromptCorrect(
                semantic_map=semantic_map_object_str).get_prompt_text())

            for reflection_iteration_idx in range(constants.SELF_REFLECTION_ITERATIONS):

                ##########################################
                ############## SELF-REFLECT ##############
                ##########################################
                print("reflecting")
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
                # Append response to conversation history
                self_reflection_conversation_history.append_assistant_message(
                    self_reflection_response)

                ##########################################
                ################ CORRECT #################
                ##########################################
                print("correcting")
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
                # Append response to conversation history
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


def plan_multiagent_reflection(mode: str, semantic_map: list, queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    for llm_provider in (constants.GEMINI_1_0_PRO, constants.GEMINI_1_5_PRO, constants.CHAT_GPT_3_5_TURBO,
                         constants.CHAT_GPT_4_O):

        # Planner
        planner_conversation_history = ConversationHistory()
        # Append system prompt
        planner_conversation_history.append_system_message(
            PromptPlan(
            ).get_prompt_text())

        # Self reflector
        self_reflector_conversation_history = ConversationHistory()
        # Append system prompt
        self_reflector_conversation_history.append_system_message(
            PromptReflect(semantic_map=semantic_map_object_str).get_prompt_text())

        # Corrector
        corrector_conversation_history = ConversationHistory()
        # Append system prompt
        corrector_conversation_history.append_system_message(
            PromptCorrect(
                semantic_map=semantic_map_object_str).get_prompt_text())

        for query_idx, query in enumerate(queries):

            # PLAN
            # Append user message
            planner_conversation_history.append_user_message(
                query)
            # Get response
            plan_response = llm_provider.generate_json(
                planner_conversation_history)
            # Append assistant message
            planner_conversation_history.append_assistant_message(
                plan_response)

            # Save plan response
            plan_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                   mode,
                                                   constants.METHOD_MULTIAGENT_REFLECTION,
                                                   llm_provider.get_provider_name(),
                                                   semantic_map_basename,
                                                   str(query_idx),
                                                   "first_plan.json")
            file_utils.create_directories_for_file(plan_response_file_path)
            file_utils.save_json_str_to_file(json_str=plan_response,
                                             output_path=plan_response_file_path)

            # SELF REFINE
            # Append user prompt
            self_reflector_conversation_history.append_user_message(
                PromptReflectUser(query=query,
                                  preliminary_response=plan_response).get_prompt_text())
            # Get response
            self_reflection_response = llm_provider.generate_text(
                self_reflector_conversation_history)
            # Append assistant prompt
            self_reflector_conversation_history.append_assistant_message(
                self_reflection_response)

            # Save self-reflection response
            self_reflection_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                              mode,
                                                              constants.METHOD_MULTIAGENT_REFLECTION,
                                                              llm_provider.get_provider_name(),
                                                              semantic_map_basename,
                                                              str(query_idx),
                                                              "self_reflection.txt")
            file_utils.save_text_to_file(text=self_reflection_response,
                                         output_path=self_reflection_response_file_path)

            # CORRECT
            # Append user prompt
            corrector_conversation_history.append_user_message(
                PromptCorrectUser(plan_response=plan_response,
                                  self_reflection_response=self_reflection_response).get_prompt_text())
            # Get response
            correction_response = llm_provider.generate_json(
                corrector_conversation_history)
            # Append assistant message
            corrector_conversation_history.append_assistant_message(
                correction_response)

            # Save correction response
            correction_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                         mode,
                                                         constants.METHOD_MULTIAGENT_REFLECTION,
                                                         llm_provider.get_provider_name(),
                                                         semantic_map_basename,
                                                         str(query_idx),
                                                         "final_plan.json")
            file_utils.save_json_str_to_file(json_str=correction_response,
                                             output_path=correction_response_file_path)


def plan_ensembling(mode: str, semantic_map, queries: list):

    semantic_map_basename = semantic_map[0]
    semantic_map_object = semantic_map[1]
    semantic_map_object_str = text_utils.dict_to_json_str(semantic_map_object)

    plan_responses = list()

    for query_idx, query in enumerate(queries):

        planner_conversation_history = ConversationHistory()

        llm_chooser = constants.GEMINI_1_5_PRO
        for llm_provider in (constants.GEMINI_1_0_PRO, constants.GEMINI_1_5_PRO, constants.CHAT_GPT_3_5_TURBO,
                             constants.CHAT_GPT_4_O):

            # PLAN
            planner_conversation_history.clear()
            # Append system prompt
            planner_conversation_history.append_system_message(
                PromptPlan(
                    semantic_map=semantic_map_object_str).get_prompt_text())
            # Append user prompt
            planner_conversation_history.append_user_message(
                query)
            # Get response
            plan_response = llm_provider.generate_json(
                planner_conversation_history)

            plan_responses.append(plan_response)

            # Save response
            plan_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                   mode,
                                                   constants.METHOD_ENSEMBLING,
                                                   semantic_map_basename,
                                                   str(query_idx),
                                                   f"plan_response_{llm_provider.get_provider_name()}.json")
            file_utils.create_directories_for_file(plan_response_file_path)
            file_utils.save_json_str_to_file(json_str=plan_response,
                                             output_path=plan_response_file_path)

        # CHOOSE
        chooser_conversation_history = ConversationHistory()
        chooser_conversation_history.clear()
        # Append system prompt
        chooser_conversation_history.append_system_message(
            ChooserPrompt(llm_responses=plan_responses,
                          semantic_map=semantic_map_object_str).get_prompt_text())
        print(chooser_conversation_history)
        choice_response = llm_chooser.generate_json(
            chooser_conversation_history)

        # Save response
        choice_response_file_path = os.path.join(constants.LLM_RESULTS_FOLDER_PATH,
                                                 mode,
                                                 constants.METHOD_ENSEMBLING,
                                                 semantic_map_basename,
                                                 str(query_idx),
                                                 f"choice.json")
        file_utils.save_json_str_to_file(json_str=choice_response,
                                         output_path=choice_response_file_path)


def main(args):
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
    for (s_m_b, s_m_o) in semantic_maps:

        # Pre-process semantic map
        pre_processed_semantic_map = (s_m_b, preprocess.preprocess_semantic_map(s_m_o,
                                                                                class_uncertainty=(args.mode == constants.MODE_UNCERTAINTY)))

        # Plan actions for every method
        if args.method == constants.METHOD_BASE:
            plan_base(args.mode, pre_processed_semantic_map, queries)
        elif args.method == constants.METHOD_SELF_REFLECTION:
            plan_self_reflection(
                args.mode, pre_processed_semantic_map, queries)
        elif args.method == constants.METHOD_MULTIAGENT_REFLECTION:
            plan_multiagent_reflection(
                args.mode, pre_processed_semantic_map, queries)
        elif args.method == constants.METHOD_ENSEMBLING:
            plan_ensembling(args.mode, pre_processed_semantic_map, queries)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="TODO")  # TODO

    parser.add_argument("-n", "--number-maps",
                        type=int,
                        help="TODO")  # TODO

    parser.add_argument("--mode",
                        type=str,
                        help="TODO",  # TODO
                        choices=[constants.MODE_CERTAINTY, constants.MODE_UNCERTAINTY])

    parser.add_argument("--method",
                        type=str,
                        help="TODO",  # TODO
                        choices=[constants.METHOD_BASE, constants.METHOD_SELF_REFLECTION, constants.METHOD_MULTIAGENT_REFLECTION, constants.METHOD_ENSEMBLING])

    args = parser.parse_args()

    main(args)
