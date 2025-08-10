import json
import unicodedata

MATCH_DATA_PATH="data/raw/data_match.json"
SAVE_MATCH_DATA="data/processed/match_data.jsonl"
TARGET_KEY = ["team1", "team2", "score1", "score2"]

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def remove_accents(input_str: str):
    nfd_form_str = unicodedata.normalize("NFD", input_str)
    remove_list = []
    for char in nfd_form_str:
        if unicodedata.category(char) != "Mn":
            remove_list.append(char)
    return "".join(remove_list)

def data_processing():
    match_data_all = load_json(MATCH_DATA_PATH)
    match_data = match_data_all.get("data")
    match_data_segments = match_data.get("segments")

    match_data_segments_processed = []
    for original_dict in match_data_segments:
        new_dict = {}
        for key in TARGET_KEY:
            value = original_dict.get(key)
            if isinstance(value, str):
                new_dict[key] = remove_accents(value)
            else:
                new_dict[key] = value
        match_data_segments_processed.append(new_dict)

    with open(SAVE_MATCH_DATA, "w", encoding="utf-8") as f:
        for data in match_data_segments_processed:
            json.dump(data, f)
            f.write("\n")

if __name__ == "__main__":
    data_processing()