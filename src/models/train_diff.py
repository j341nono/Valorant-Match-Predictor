import torch
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

MODEL_TYPE="mlp_model_diff"
SAVE_MATCH_DATA="data/processed/match_data.jsonl"
VALID_SPLIT_RATIO=0.1
TEST_SPLIT_RATIO=1/9
BATCH_SIZE=64
EMBEDDING_DIM=32
LEARNING_RATE=1e-3
EPOCHS=10
MODEL_SAVE_PATH="outputs/"+MODEL_TYPE
RESULT_SAVE_PATH="results/"+MODEL_TYPE+"/test_results.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_team_stats(data):
    stats = {}
    for item in data:
        t1, t2 = item["team1"], item["team2"]
        s1, s2 = int(item["score1"]), int(item["score2"])
        for team, score_for, score_against, win in [
            (t1, s1, s2, s1 > s2),
            (t2, s2, s1, s2 > s1),
        ]:
            if team not in stats:
                stats[team] = {"points_for":0, "points_against":0, "games":0, "wins":0}
            stats[team]["points_for"] += score_for
            stats[team]["points_against"] += score_against
            stats[team]["games"] += 1
            stats[team]["wins"] += int(win)

    for team, st in stats.items():
        st["avg_points_for"] = st["points_for"] / st["games"]
        st["avg_points_against"] = st["points_against"] / st["games"]
        st["avg_diff"] = (st["points_for"] - st["points_against"]) / st["games"]
        st["win_rate"] = st["wins"] / st["games"]

    return stats


def load_data_and_preprocess(filepath: str):
    raw_data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))

    # チームID辞書作成
    all_teams = set()
    for item in raw_data:
        all_teams.add(item["team1"])
        all_teams.add(item["team2"])
    team_to_idx = {team: i for i, team in enumerate(sorted(list(all_teams)))}
    idx_to_team = {i: team for team, i in team_to_idx.items()}

    # チーム統計量
    stats = build_team_stats(raw_data)

    processed_data = []
    for item in raw_data:
        s1 = int(item["score1"])
        s2 = int(item["score2"])
        processed_data.append({
            "team1_id": team_to_idx[item["team1"]],
            "team2_id": team_to_idx[item["team2"]],
            # チーム1の特徴
            "t1_avg_for": stats[item["team1"]]["avg_points_for"],
            "t1_avg_against": stats[item["team1"]]["avg_points_against"],
            "t1_diff": stats[item["team1"]]["avg_diff"],
            "t1_winrate": stats[item["team1"]]["win_rate"],
            # チーム2の特徴
            "t2_avg_for": stats[item["team2"]]["avg_points_for"],
            "t2_avg_against": stats[item["team2"]]["avg_points_against"],
            "t2_diff": stats[item["team2"]]["avg_diff"],
            "t2_winrate": stats[item["team2"]]["win_rate"],
            # ラベル
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
        extra_features = [
            item["t1_avg_for"], item["t1_avg_against"], item["t1_diff"], item["t1_winrate"],
            item["t2_avg_for"], item["t2_avg_against"], item["t2_diff"], item["t2_winrate"]
        ]
        return {
            "team1_id": torch.tensor(item["team1_id"], dtype=torch.long),
            "team2_id": torch.tensor(item["team2_id"], dtype=torch.long),
            "extra_features": torch.tensor(extra_features, dtype=torch.float),
            "label": torch.tensor(item["label"], dtype=torch.float)
        }


class MatchOutcomePredictor(nn.Module):
    def __init__(self, num_teams, embedding_dim, hidden_dims, dropout_rate, extra_features_dim=8):
        super().__init__()
        self.team_embedding = nn.Embedding(num_embeddings=num_teams, embedding_dim=embedding_dim)
        
        input_dim = embedding_dim * 2 + extra_features_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, team1_id, team2_id, extra_features):
        team1_embed = self.team_embedding(team1_id)
        team2_embed = self.team_embedding(team2_id)
        x = torch.cat([team1_embed, team2_embed, extra_features], dim=1)
        x = self.layers(x)
        return self.sigmoid(x).squeeze()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    processed_data, team_to_idx, idx_to_team = load_data_and_preprocess(SAVE_MATCH_DATA)
    
    num_teams = len(team_to_idx)
    logging.info(f"Total unique teams: {num_teams}")
    logging.info(f"Total matches loaded: {len(processed_data)}")

    train_data, test_data = train_test_split(processed_data, test_size=TEST_SPLIT_RATIO, random_state=33)
    train_data, valid_data = train_test_split(train_data, test_size=VALID_SPLIT_RATIO, random_state=33)

    train_dataset = MatchDataset(train_data)
    valid_dataset = MatchDataset(valid_data)
    test_dataset = MatchDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = MatchOutcomePredictor(
        num_teams=num_teams, embedding_dim=EMBEDDING_DIM, hidden_dims=[64, 32], dropout_rate=0.3
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader):
            team1_ids = batch["team1_id"].to(device)
            team2_ids = batch["team2_id"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(team1_ids, team2_ids, extra_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        
        model.eval()
        total_valid_loss = 0
        correct_predictions = 0
        with torch.no_grad():
            for batch in valid_loader:
                team1_ids = batch["team1_id"].to(device)
                team2_ids = batch["team2_id"].to(device)
                extra_features = batch["extra_features"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(team1_ids, team2_ids, extra_features)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item()

                predicted = (outputs > 0.5).float()
                correct_predictions += (predicted == labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_acc = correct_predictions / len(valid_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_valid_loss:.4f}, Val Accuracy: {valid_acc:.4f}")


    logging.info("Starting Test Phase")
    model.eval()
    total_test_loss = 0
    correct_test_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            team1_ids = batch["team1_id"].to(device)
            team2_ids = batch["team2_id"].to(device)
            extra_features = batch["extra_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(team1_ids, team2_ids, extra_features)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()

            predicted = (outputs > 0.5).float()
            correct_test_predictions += (predicted == labels).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_acc = correct_test_predictions / len(test_dataset)
    print(f"Test Results -> Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "team_to_idx": team_to_idx,
        "embedding_dim": EMBEDDING_DIM
    }, MODEL_SAVE_PATH)
    logging.info(f"Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
