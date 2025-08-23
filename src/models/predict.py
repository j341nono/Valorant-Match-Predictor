import torch
import json
import torch.nn as nn

MODEL_SAVE_PATH="outputs/mlp_model"
SAVE_MATCH_DATA="data/processed/match_data.jsonl"


# =========================
# モデル定義
# =========================
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


# =========================
# チーム統計量を計算
# =========================
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


# =========================
# 予測関数
# =========================
def predict_match(team1_name, team2_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルと辞書をロード
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    team_to_idx = checkpoint["team_to_idx"]
    embedding_dim = checkpoint["embedding_dim"]

    num_teams = len(team_to_idx)
    model = MatchOutcomePredictor(
        num_teams=num_teams, embedding_dim=embedding_dim, hidden_dims=[64,32], dropout_rate=0.3
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 試合データから統計量を計算
    raw_data = []
    with open(SAVE_MATCH_DATA, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(json.loads(line))
    stats = build_team_stats(raw_data)

    if team1_name not in team_to_idx or team2_name not in team_to_idx:
        raise ValueError("指定したチームが辞書に存在しません")

    # 特徴量作成
    extra_features = [
        stats[team1_name]["avg_points_for"], stats[team1_name]["avg_points_against"],
        stats[team1_name]["avg_diff"], stats[team1_name]["win_rate"],
        stats[team2_name]["avg_points_for"], stats[team2_name]["avg_points_against"],
        stats[team2_name]["avg_diff"], stats[team2_name]["win_rate"],
    ]

    team1_id = torch.tensor([team_to_idx[team1_name]], dtype=torch.long).to(device)
    team2_id = torch.tensor([team_to_idx[team2_name]], dtype=torch.long).to(device)
    extra_features = torch.tensor([extra_features], dtype=torch.float).to(device)

    with torch.no_grad():
        prob = model(team1_id, team2_id, extra_features).item()

    print(f"予測: {team1_name} が勝つ確率 = {prob:.3f}, {team2_name} が勝つ確率 = {1-prob:.3f}")
    return prob


# =========================
# 実行例
# =========================
if __name__ == "__main__":
    predict_match("TeamA", "TeamB")
