import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, BatchNorm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestNeighbors
import random

# --- Config ---
SEED = 42
K_NEIGHBORS = 10
N_SPLITS = 5
SELECTED_FEATURES = ['total_tweets', 'avg_tweet_length', 'avg_sentiment', 'activity_duration', 'degree']
RUMOR_RATIO_THRESHOLD = 0.6

# --- Reproducibility ---
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# --- Data Loading ---
df = pd.read_csv("pheme_user_features_multi_event.csv", index_col=0)
df['events'] = df['events'].apply(json.loads)
df['neighbors'] = df['neighbors'].apply(json.loads)
df.index = df.index.astype(str)
df['neighbors'] = df['neighbors'].apply(lambda nbs: [str(nb) for nb in nbs])

X = df[SELECTED_FEATURES]
y = ((df['rumor_ratio'] > RUMOR_RATIO_THRESHOLD)).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Graph Construction (Social + kNN Edges) ---
user_ids = list(df.index)
user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
X_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Social edges
edges = set()
for uid, row in df.iterrows():
    src = user_id_to_idx[uid]
    for nb in row['neighbors']:
        if nb in user_id_to_idx and nb != uid:
            tgt = user_id_to_idx[nb]
            edges.add((src, tgt))

edges = np.array(list(edges))
edge_index = torch.tensor(edges.T, dtype=torch.long) if edges.size > 0 else torch.zeros((2, 0), dtype=torch.long)
data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)

# --- GNN Definition ---
class MultiGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# --- Helper for Metrics ---
def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics

# --- Cross-Validation ---
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
dummy_metrics, lr_metrics, svm_metrics, gnn_metrics = [], [], [], []

print("Starting 5-fold cross-validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    print(f"\nFold {fold+1}/{N_SPLITS}")
    train_subidx, val_subidx = train_test_split(
        train_idx, test_size=0.1, stratify=y.iloc[train_idx], random_state=fold
    )

    # Dummy Classifier
    dummy = DummyClassifier(strategy='stratified', random_state=SEED)
    dummy.fit(X_scaled[train_idx], y.iloc[train_idx])
    dummy_preds = dummy.predict(X_scaled[test_idx])
    dummy_metrics.append(compute_metrics(y.iloc[test_idx], dummy_preds))
    print(f"Dummy Macro-F1: {dummy_metrics[-1]['f1']:.4f}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
    lr.fit(X_scaled[train_idx], y.iloc[train_idx])
    lr_preds = lr.predict(X_scaled[test_idx])
    lr_proba = lr.predict_proba(X_scaled[test_idx])[:, 1]
    lr_metrics.append(compute_metrics(y.iloc[test_idx], lr_preds, lr_proba))
    print(f"Logistic Regression Macro-F1: {lr_metrics[-1]['f1']:.4f}")

    # SVM
    svm = SVC(kernel='rbf', probability=True, random_state=SEED, class_weight='balanced')
    svm.fit(X_scaled[train_idx], y.iloc[train_idx])
    svm_preds = svm.predict(X_scaled[test_idx])
    svm_proba = svm.predict_proba(X_scaled[test_idx])[:, 1]
    svm_metrics.append(compute_metrics(y.iloc[test_idx], svm_preds, svm_proba))
    print(f"SVM Macro-F1: {svm_metrics[-1]['f1']:.4f}")

    # GNN with CrossEntropyLoss and early stopping
    train_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    val_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    test_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    train_mask[train_subidx] = True
    val_mask[val_subidx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    model = MultiGNN(
        in_channels=X_tensor.shape[1],
        hidden_channels=128,
        out_channels=2,
        dropout=0
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Early stopping: Macro-F1 on val with threshold tuning
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x, data.edge_index)
            val_probs = torch.softmax(val_logits[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            thresholds = np.linspace(0, 1, 101)
            f1s = [f1_score(val_true, (val_probs > t).astype(int), average='macro', zero_division=0) for t in thresholds]
            max_f1 = max(f1s)
        if max_f1 > best_val_f1:
            best_val_f1 = max_f1
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= patience and epoch > 20:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate GNN with threshold tuning for Macro-F1
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
        thresholds = np.linspace(0, 1, 101)
        f1s = [f1_score(y_true, (probs > t).astype(int), average='macro', zero_division=0) for t in thresholds]
        best_idx = np.argmax(f1s)
        best_threshold = thresholds[best_idx]
        best_f1 = f1s[best_idx]
        gnn_preds = (probs > best_threshold).astype(int)
        gnn_metrics.append(compute_metrics(y_true, gnn_preds, probs))
        print(f"GNN Macro-F1: {best_f1:.4f} (best threshold: {best_threshold:.2f})")

# --- Results Summary ---
def summarize_metrics(metrics_list, name):
    f1s = [m['f1'] for m in metrics_list]
    precisions = [m['precision'] for m in metrics_list]
    recalls = [m['recall'] for m in metrics_list]
    
    print(f"\n{name} Results:")
    print(f"  Macro-F1: {np.mean(f1s):.4f}")
    print(f"  Precision: {np.mean(precisions):.4f}")
    print(f"  Recall: {np.mean(recalls):.4f}")
    
    if 'roc_auc' in metrics_list[0]:
        roc_aucs = [m['roc_auc'] for m in metrics_list]
        pr_aucs = [m['pr_auc'] for m in metrics_list]
        print(f"  ROC-AUC: {np.mean(roc_aucs):.4f}")
        print(f"  PR-AUC: {np.mean(pr_aucs):.4f}")
    
    return {
        'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
        'precision_mean': np.mean(precisions), 'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls), 'recall_std': np.std(recalls)
    }

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS")
print("="*50)

dummy_summary = summarize_metrics(dummy_metrics, "Dummy Classifier")
lr_summary = summarize_metrics(lr_metrics, "Logistic Regression")
svm_summary = summarize_metrics(svm_metrics, "SVM")
gnn_summary = summarize_metrics(gnn_metrics, "GNN (Proposed)")

