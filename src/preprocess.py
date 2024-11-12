import os
import pprint

import constants
from utils import file_utils
from voxelad.preprocess import preprocess_semantic_map

if __name__ == "__main__":

    semantic_map_file_path = os.path.join(constants.SEMANTIC_MAPS_FOLDER_PATH,
                                          input("Introduce semantic map file name: "))
    semantic_map_obj = file_utils.load_json(semantic_map_file_path)
    pprint.pprint(preprocess_semantic_map(semantic_map_obj, False))
