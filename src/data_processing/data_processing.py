import json
import unicodedata
import os

MATCH_DATA_PATH_1="data/raw/data_match_1-10.json"
MATCH_DATA_PATH_2="data/raw/data_match_11-20.json"
MATCH_DATA_PATH_3="data/raw/data_match_21-30.json"

SAVE_MATCH_DATA="data/processed/match_data.jsonl"
TARGET_KEY = ["team1", "team2", "score1", "score2"]

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def remove_accents(input_str: str) -> str:
    nfd_form_str = unicodedata.normalize("NFD", input_str)
    remove_list = []
    for char in nfd_form_str:
        if unicodedata.category(char) != "Mn":
            remove_list.append(char)
    return "".join(remove_list)

def data_processing(input_file_path: list) -> None:
    match_data_segments_processed = []
    for file_path in input_file_path:
        match_data_all = load_json(file_path)
        match_data = match_data_all.get("data")
        match_data_segments = match_data.get("segments")

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

def main():
    if(os.path.isfile(SAVE_MATCH_DATA)):
        os.remove(SAVE_MATCH_DATA)
    
    input_file_paths = [
        "data/raw/data_match_1-10.json",
        "data/raw/data_match_11-20.json",
        "data/raw/data_match_21-30.json"
    ]

    data_processing(input_file_paths)


if __name__ == "__main__":
    main()