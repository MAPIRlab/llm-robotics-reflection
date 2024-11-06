import base64
import json
import os
import pickle
import re
import shutil

import PIL
import PIL.Image
import yaml


def is_pdf(file_path: str) -> bool:
    """Checks if the file at the specified path has a PDF extension.

    Args:
        file_path (str): The path of the file to be checked.

    Returns:
        bool: True if the file has a PDF extension, False otherwise.
    """
    _, extension = os.path.splitext(file_path)
    return extension.lower() == '.pdf'


def save_text_to_file(text: str, output_path: str):
    """
    Saves text to a file.

    Args:
        text (str): The extracted text from the document.
        output_path (str): The path of the processed document

    Returns:
        str: The path to the saved text file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def read_text_from_file(file_path: str) -> str:
    """
    Reads and returns the entire content of a text file.

    Args:
        file_path (str): The path to the file that needs to be read.

    Returns:
        str: The content of the file as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_binary_from_file(file_path: str):
    """
    Reads and returns the content of a file in binary mode.

    This function opens a file in binary read mode ('rb') and reads its entire content.
    The content is returned as a bytes object, which is suitable for binary data handling.

    Args:
        file_path (str): The path to the file that needs to be read.

    Returns:
        bytes: The content of the file as a bytes object.
    """
    with open(file_path, "rb") as file:
        return file.read()


def load_json(file_path: str):
    """
    Loads and returns the content of a JSON file.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict: The content of the JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)


def load_yaml(file_path):
    """
    TODO
    """
    with open(file_path, 'r', encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_png_image(image: PIL.Image, output_path: str):
    """
    Saves a PNG image to a file.

    Args:
        image (PIL.Image.Image): The image to be saved to a file.
        output_path (str): The path of the image

    Returns:
        str: The path to the saved text file.
    """
    image.save(output_path, "PNG")
    # print(f"Imagen guardada en: {output_path}")


def save_json_str_to_file(json_str: str, output_path: str):
    """
    Saves a JSON string to a file at a specified path, overwriting the file if it exists.

    Args:
        json_str (str): The JSON string to save.
        output_path (str): Path to the file where the JSON will be saved.

    Raises:
        json.JSONDecodeError: If `json_str` is not valid JSON.
        OSError: If writing to the file fails.
    """
    json_obj = json.loads(json_str)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(json_obj, json_file, indent=4, ensure_ascii=False)
    # print(f"JSON file saved: {output_path}")


def save_dict_to_json_file(dictionary, file_path):
    """
    Saves a dictionary to a file as JSON.

    Parameters:
    dictionary (dict): The dictionary to save.
    file_path (str): The path to the file where the JSON will be saved.
    """
    with open(file_path, 'w', encoding="utf-8") as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)


def create_directories_for_file(file_path):
    """
    Creates all necessary parent directories for the given file path if they do not already exist.

    This function takes a file path, extracts its directory path, and ensures that this directory
    path exists by creating any missing directories in the path. It uses the `os.makedirs()` function
    with `exist_ok=True` to prevent raising an error if the directory already exists.

    Args:
        file_path (str): The full path to a file for which directories need to be created.
    """
    directory_path = os.path.dirname(file_path)
    os.makedirs(directory_path, exist_ok=True)


def create_directories(path):
    """
    Creates all directories in the specified path if they do not already exist.

    Args:
        path (str): The full path of directories to be created.
    """

    os.makedirs(path, exist_ok=True)


def directory_exists_and_contains_files(directory_path: str):
    """
    Check if the given directory path exists and contains at least one file.

    Args:
    directory_path (str): The path to the directory.

    Returns:
    bool: True if the directory exists and contains at least one file, False otherwise.
    """
    if os.path.isdir(directory_path):
        for entry in os.listdir(directory_path):
            if os.path.isfile(os.path.join(directory_path, entry)):
                return True
    return False


def encode_file_base64(self, file_path: str) -> str:
    """
    Encodes a file to a Base64 string.

    Reads the file from the specified path, encodes its binary content into Base64, and returns it as a UTF-8 string.

    Args:
        file_path(str): The path to the file.

    Returns:
        str: The Base64-encoded string of the file's contents.
    """
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode('utf-8')


def get_file_basename(file_path: str, include_extension: bool = False) -> str:
    """
    Extracts and returns the basename (filename without the extension) from a given file path.

    Args:
        file_path (str): The complete path to the file from which to extract the basename.

    Returns:
        str: The basename of the file, without the file extension.
    """
    # Extract the full file name with extension
    full_file_name = os.path.basename(file_path)
    # Split the full file name into the file name and extension and return the file name part
    if include_extension:
        return full_file_name
    else:
        return os.path.splitext(full_file_name)[0]


def find_matching_files(base_path: str, pattern: str) -> list:
    """
    Searches for files within the base_path whose names match the given regular expression pattern.

    Args:
        base_path (str): The directory path to search in.
        pattern (str): The regular expression pattern to match file names against.

    Returns:
        list: A list of paths to files that match the pattern.
    """
    matching_files = []
    regex = re.compile(pattern)

    filenames = os.listdir(base_path)

    for filename in filenames:
        full_path = os.path.join(base_path, filename)
        if os.path.isfile(full_path) and regex.match(filename):
            matching_files.append(full_path)

    return matching_files


def save_as_pickle(obj, file_path: str):
    """
    Saves an object to a file using pickle.

    Args:
        obj (any): The Python object to be pickled.
        file_path (str): The path to the file where the object should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def load_from_pickle(file_path: str):
    """
    Load and return the data from a pickle file.

    This function opens a pickle file in binary read mode, deserializes the pickle data using the pickle module, 
    and returns the resulting Python object.

    Args:
        file_path (str): The path to the pickle file that is to be loaded.

    Returns:
        object: The Python object loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def count_files(path: str) -> int:
    """
    Counts the total number of files in the specified directory and its subdirectories.

    Args:
        path (str): The path of the directory to count documents in.

    Returns:
        int: The total number of files found in the directory and its subdirectories.

    """
    total_documents = 0
    for _, _, files in os.walk(path):
        total_documents += len(files)
    return total_documents


def list_subdirectories(directory_path):
    """
    List all subdirectories inside the given directory path.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        list: A list of subdirectory names.
    """
    entries = os.listdir(directory_path)
    subdirectories = [entry for entry in entries if os.path.isdir(
        os.path.join(directory_path, entry))]
    return subdirectories


def copy_file_to_dir(src_file, dst_folder, new_name):
    """
    Copies a file to a specified folder with a new name.

    Args:
        src_file (str): The path to the source file.
        dst_folder (str): The path to the destination folder.
        new_name (str): The new name for the copied file.

    Returns:
        str: The path to the copied file with the new name.
    """
    if not os.path.isfile(src_file):
        raise FileNotFoundError(
            f"The source file '{src_file}' does not exist.")

    if not os.path.isdir(dst_folder):
        raise NotADirectoryError(f"The destination folder '{
                                 dst_folder}' does not exist.")

    # Copy the file with the new name
    dst_file = os.path.join(dst_folder, new_name)
    shutil.copy(src_file, dst_file)

    return dst_file


def pdf_to_base64(pdf_path) -> str:
    """
    Convert a PDF file to a base64 string.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        str: Base64 encoded string of the PDF file
    """
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
        base64_string = base64.b64encode(pdf_bytes).decode("utf-8")
    return base64_string