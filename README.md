# llm-robotics-reflection

This repository contains the code for the paper *"Agentic Workflows for Improving LLM Reasoning in Robotic Object-Centered Planning"*, submitted to MDPI Robotics.

![Paper Title: Agentic Workflows for Improving LLM Reasoning in Robotic Object-Centered Planning](images/paper_title.png)

## Article

The article is currently under review in the MDPI Robotics Journal. Its preliminary (and not reviewed) version has been published in Preprints.org at [this link](https://www.preprints.org/manuscript/202501.0131).

### Abstract

The article abstract is the following

Large Language Models (LLMs) provide cognitive capabilities that enable robots to interpret and reason about their workspace, especially when paired with semantically rich representations like semantic maps. However, these models are prone to generating inaccurate or invented responses, known as hallucinations, that can produce erratic robotic operation. This can be addressed by employing agentic workflows, structured processes that guide and refine the model's output to improve response quality. This work formally defines and qualitatively analyzes the impact of three agentic workflows (LLM Ensemble, Self-Reflection, and Multi-Agent Reflection) on enhancing the reasoning capabilities of an LLM guiding a robotic system to perform object-centered planning. In this context, the LLM is provided with a pre-built semantic map of the environment and a query, to which it must respond by determining the most relevant objects for the query. This response can be used in a multitude of downstream tasks. Extensive experiments were carried out employing state-of-the-art LLMs and semantic maps generated from the widely-used datasets ScanNet and SceneNN. Results show that agentic workflows significantly enhance object retrieval performance, especially in scenarios requiring complex reasoning, with improvements averaging up to 10% over the baseline.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/MAPIRlab/llm-robotics-reflection.git
    cd mdpi-reflection
    ```

2. Set up the virtual environment:
    ```sh
    python -m venv virtual_environment
    source virtual_environment/Scripts/activate  # On Windows
    # source virtual_environment/bin/activate    # On Unix or MacOS
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Directory Structure

Maybe, the most important parts of the repository are the self-created dataset and the prompts involved in the workflows.

### Dataset

The created dataset can be found in the [data](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/data) folder, here you will find the following files and sub-folders:
- [data/semantic_maps/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/data/semantic_maps): folder containing the semantic maps used in the study, in JSON format.
- [data/queries.yaml](https://github.com/MAPIRlab/llm-robotics-reflection/blob/main/data/queries.yaml): YAML file containing all the queries considered in this work.
- [data/responses/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/data/responses): folder containing the ground truth responses for each semantic map-query pair.

### Workflows Prompts

The prompts used in each stage of the considered workflows can be found in the [src/prompts/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/data) folder. This folder contains the following Python files:
- [src/prompts/planner_prompt.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompts/planner_prompt.py): contains classes for the first response generation prompts, in the Baseline, Self-Reflection, Multi-Agent Reflection, and LLM Ensemble workflows.
- [src/prompts/self_reflection_prompt.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompts/self_reflection_prompt.py): contains classes for the feedback generation prompts, in the Self-Reflection and Multi-Agent Reflection workflows.
- [src/prompts/correction_prompt.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompts/self_reflection_prompt.py): contains classes for the response refinement process, in the Self-Reflection and Multi-Agent Reflection workflows.
- [src/prompts/chooser_prompt.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompts/self_reflection_prompt.py): contain classes for the evaluator prompt in the LLM Ensemble workflow.
- [src/prompts/prompt.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompts/prompt.py): abstract class for a prompt.

### General structure

The directory structure is as follows:

- [credentials](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/credentials): contains JSON files needed for API authentication, see "Usage" section (explained above).
- [data](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/data): Contains the dataset created for this work, with semantic maps, object-centered planning queries, and annotated responses.
- [results](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/results): Contains the outputs results generated by the repository scripts.
- [src](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src): Contains the source code for the project.
  - [annotate.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/annotate.py): Script that launches the GUI for annotating the responses and creating the ground truth.
  - [compare/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/compare): Directory with utils for the evaluation and the comparison results.
  - [constants.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/constants.py): Python file with constants involved in all the processes.
  - [evaluate.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/evaluate.py): Script for the evaluation of the results.
  - [llm/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/llm): Directory with LLM implementations and interfaces.
  - `llm_test.py`: Script for testing the LLMs generation text capabilities.
  - [main.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/main.py): Main script of the project, evaluates every query on every semantic map, for all the considered workflows.
  - [preprocess.py](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/preprocess.py): Script for pre-processing a Voxeland semantic map.
  - [prompt/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/prompt): Directory with prompt related classes (explained above).
  - [results](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/results): Directory where the results of each script are saved.
  - [utils/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/utils): Directory for utility scripts.
  - [voxelad/](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/voxeland): Directory with utils for pre-processing Voxeland semantic maps.
- [virtual_environment](https://github.com/MAPIRlab/llm-robotics-reflection/tree/main/src/virtual_environment): Contains the virtual environment setup.

## Usage

1. Activate the virtual environment:
    ```sh
    source virtual_environment/Scripts/activate  # On Windows
    # source virtual_environment/bin/activate    # On Unix or MacOS
    ```

2. Run some of the Python scripts, for example `main.py`:
    ```sh
    python src/main.py
    ```

## Credentials

As it is explained in the paper, the considered workflows have been implemented using LLMs from the Google Gemini family. Furthermore, in this repository we present the option of running the considered workflows using LLMs from the OpenIA family.
For running these proprietary models, we need to properly configure their credentials.

### Google Gemini LLMs

The Google Vertex AI credentials should be placed in the `credentials` folder. Then, the `constants.py` file constants related to this model should be modified with the corresponding information, i.e., modifying the `GOOGLE_GEMINI_CREDENTIALS_FILENAME`, `GOOGLE_GEMINI_PROJECT_ID`, and `GOOGLE_GEMINI_PROJECT_LOCATION` constants.

### OpenAI LLMs

For using the OpenAI LLMs it is needed to create a `.env` file in the root directory of this repository, containing the environment variable `OPENAI_API_KEY`, which should include the OpenAI account API key.

## Main scripts

This repository main scripts are the following:
- `main.py`: executes each query on each semantic map, for every considered workflow.
- `evaluate.py`: evaluates the results generated in the previous step, generating tables and charts presented in the paper.
- `annotate.py`: GUI useful for generating the ground truth of annotated responses.

The functioning of each script is the following:

### `main.py`

This is the main script to run the project. It orchestrates the entire workflow from loading data to generating results. The script processes semantic maps and queries, applies the specified agentic workflows, and generates the final output.

**Parameters:**
- `-q`, `--queries`: Path to the file containing the queries (required).
- `-m`, `--maps`: Path to the directory containing the semantic maps (required).
- `-w`, `--workflow`: The agentic workflow to be applied (required).
  - Choices: `LLM_ENSEMBLE`, `SELF_REFLECTION`, `MULTI_AGENT_REFLECTION`
- `-o`, `--output`: Path to the directory where the results will be saved (required).
- `-n`, `--number-queries`: Number of queries to be processed (default: 10).
- `--mode`: Semantic maps input mode to LLMs, with uncertainty (`uncertainty`) or not (`certainty`) (default: `certainty`).

**Usage:**
```sh
python src/main.py -q path/to/queries.json -m path/to/semantic_maps/ -w LLM_ENSEMBLE -o path/to/output/ --number-queries 10 --mode certainty
