from src.options import Options
import sys
import os


if __name__ == "__main__":
    print("Testing option class and file")
    options = Options()
    options.load("src/test_options.json")
    options.display()
