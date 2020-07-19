import json

def load_settings(filepath="settings.json"):

    with open(filepath) as json_file:
        settings = json.load(json_file)

    return settings

