
# TODO: replace and use search_dict_list
import json


def search_dict_by_key_value(dictionary_list: list, key: str, value: str):
    """
    Searches for a dictionary that has a specific key-value pair in a list of dictionaries.
    """
    for doc_dict in dictionary_list:
        if key in doc_dict and doc_dict[key] == value:
            return doc_dict
    return None


def search_dict_list(dict_list, search_criteria) -> list:
    """
    Search for dictionaries in a list based on a list of key-value pairs.

    Args:
        dict_list: List of dictionaries to search in.
        search_criteria: Dictionary of key-value pairs to match.

    Returns:
        list: List of dictionaries that match all key-value pairs.
    """
    results = []
    for d in dict_list:
        if all(d.get(k) == v for k, v in search_criteria.items()):
            results.append(d)
    return results


def delete_keys(dictionary: dict, keys: list) -> dict:
    """
    Returns a new dictionary with specified keys removed.

    This function creates and returns a new dictionary that contains only those items from the
    original dictionary whose keys are not in the provided list of keys to remove.

    Args:
        dictionary (dict): The original dictionary from which keys are to be removed.
        keys (list): A list of keys that should be removed from the dictionary.

    Returns:
        dict: A new dictionary with the specified keys removed.
    """
    return {k: v for k, v in dictionary.items() if k not in keys}


def all_values_none_except_keys(dictionary, keys):
    """
    Checks if all values in a dictionary are None, except for those keys specified.

    This function iterates over each item in the dictionary. If it encounters a key not listed
    in the specified `keys` and its value is not None, it returns False. If all non-specified keys
    have values of None, it returns True.

    Args:
        dictionary (dict): The dictionary to check.
        keys (list): A list of keys that are allowed to have non-None values.

    Returns:
        bool: True if all values in the dictionary are None except for those associated with the specified keys,
              otherwise False.
    """
    for k, v in dictionary.items():
        if k in keys:
            continue
        if v is not None:
            return False
    return True


def load_dict(text: str):
    return json.loads(text)


def dict_to_json_str(obj, indent):
    return json.dumps(obj, indent=indent, ensure_ascii=False)