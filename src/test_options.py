from src.options import Options
import sys
import os
import json


if __name__ == "__main__":
    print("Testing option class and file")
    options = Options()
    options.display()

    base_options_file = "mock_config.json"
    with open(base_options_file, 'r') as json_file:
        options.update_options(json.load(json_file))
    options.display()
