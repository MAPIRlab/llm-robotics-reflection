

import os

import constants
from utils import file_utils


def reduce_float_precision(data: dict, precision: int = 2) -> dict:
    """
    Reduces the precision of float numbers in the given dictionary to the specified number of decimal places.

    Args:
        data (dict): The dictionary containing nested float values.
        precision (int): The number of decimal digits to round floats to. Default is 2.

    Returns:
        dict: The dictionary with rounded float values.
    """
    for key, value in data.items():
        if isinstance(value, float):
            data[key] = round(value, precision)
        elif isinstance(value, dict):
            data[key] = reduce_float_precision(value, precision)
        elif isinstance(value, list):
            data[key] = [round(v, precision) if isinstance(
                v, float) else v for v in value]
    return data


def reduce_class_uncertainty(data: dict) -> dict:
    """
    Reduces uncertainty in class instances by filtering out those with a maximum classification result of "unknown".

    This function iterates through the instances in the input dictionary `data` and selects only the instances where 
    the maximum classification result is not labeled as "unknown". For each instance, it determines the classification 
    result with the highest associated value and includes the instance in the resulting dictionary if the maximum result 
    is different from "unknown".

    Args:
        data (dict): A dictionary containing class instances. The expected structure is:
            {
                "instances": {
                    object_key: {
                        "results": [
                            (result_key, result_value),
                            ...
                        ]
                    },
                    ...
                }
            }

    Returns:
        dict: A new dictionary containing only the filtered instances. The returned dictionary has the same structure as 
        the input but excludes instances with a maximum classification result of "unknown".
    """
    new_data = dict()
    new_data["instances"] = dict()

    for object_key, object_value in data["instances"].items():

        max_result_key = None
        max_result_value = None

        # Get (max_result_key, max_result_value)
        for result_key, result_value in object_value["results"].items():

            if max_result_key is None or max_result_value < result_value:
                max_result_key = result_key
                max_result_value = result_value

        # Include object if max result was not unknown
        if max_result_key != "unknown":
            new_data["instances"][object_key] = object_value

    return new_data


def preprocess_semantic_map(semantic_map: dict, class_uncertainty: bool = True):
    """
    TODO: documentation
    """
    semantic_map = reduce_float_precision(semantic_map, 2)
    if not class_uncertainty:
        semantic_map = reduce_class_uncertainty(semantic_map)
    return semantic_map
