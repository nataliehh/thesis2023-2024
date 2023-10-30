import json

def read_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)