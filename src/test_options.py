from src.options import Options

if __name__ == "__main__":
    print("Testing option class and file")
    options = Options()
    options.load("test_options.json")
    options.display()

