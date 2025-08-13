import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import logging
from sklearn.model_selection import train_test_split


SAVE_MATCH_DATA="data/processed/match_data.jsonl"
TEST_SPLIT_RATIO=0.3


class MatchOutcomePredictor(nn.Module):
    def __init__(self, num_teams, embedding_dim=32):
        super().__init__()
        self.team_embedding = nn.Embedding(num_embeddings=num_teams, embedding_dim=embedding_dim)
        input_dim = embedding_dim * 2 + 2
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, team1_id, team2_id, score1, score2):
        team1_embed = self.team_embedding(team1_id)
        team2_embed = self.team_embedding(team2_id)
        scores = torch.stack([score1, score2], dim=1).float()
        x = torch.cat([team1_embed, team2_embed, scores], dim=1)
        x = self.layers(x)
        output = self.sigmoid(x)
        return output.squeeze()


def load_data_and_preprocess(filepath: str):
    raw_data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    all_teams = set()
    for item in raw_data:
        all_teams.add(item["team1"])
        all_teams.add(item["team2"])

    team_to_idx = {
        team: i for i, team in enumerate(sorted(list(all_teams)))
    }
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    processed_data = []
    for item in raw_data:
        s1 = int(item["score1"])
        s2 = int(item["score2"])
        processed_data.append({
            "team1_id": team_to_idx[item["team1"]],
            "team2_id": team_to_idx[item["team2"]],
            "score1": s1,
            "score2": s2,
            "label": 1.0 if s1 > s2 else 0.0
        })
    return processed_data, team_to_idx, idx_to_team

class MatchDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "team1_id": torch.tensor(item["team1_id"], dtype=torch.long),
            "team2_id": torch.tensor(item["team2_id"], dtype=torch.long),
            "score1": torch.tensor(item["score1"], dtype=torch.float),
            "score2": torch.tensor(item["score2"], dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }


def debug():
    processed_data, team_to_idx, idx_to_team = load_data_and_preprocess(SAVE_MATCH_DATA)
    print(processed_data[0])
    print(f"team_to_idx: {team_to_idx}")
    print(f"idx_to_team: {idx_to_team}")


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Using device] {device}")

#     processed_data, team_to_idx, idx_to_team = load_data_and_preprocess(SAVE_MATCH_DATA)
#     num_teams = len(team_to_idx)
#     logging.info(f"Total unique teams: {num_teams}")
#     logging.info(f"Total matches loaded: {len(processed_data)}")

#     train_data, valid_data = train_test_split(processed_data, test_size=TEST_SPLIT_RATIO, random_state=33)
#     train_dataset = MatchDataset(train_data)
#     valid_dataset = MatchDataset(valid_data)

#     train_loader = DataLoader()


if __name__ == "__main__":
    debug()
