import json

def read_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file