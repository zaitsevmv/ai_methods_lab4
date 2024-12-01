from json import load, JSONDecodeError


# Reading config file
# Input: json file path
# Output: json
def load_config(filename: str):
    try:
        with open(filename, 'r') as config_file:
            config = load(config_file)
            return config
    except FileNotFoundError:
        print(f"Configuration file '{filename}' not found. Using default settings.")
        return None
    except JSONDecodeError:
        print(f"Error decoding JSON from the file '{filename}'.")
        return None