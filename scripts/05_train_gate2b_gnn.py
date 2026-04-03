"""
Script 05: Gate 2B — Cascade Risk Graph Neural Network (GCN)
=============================================================
Models default contagion across the merchant transaction network.
A merchant's risk is influenced by the risk of their trading partners.

Architecture: 3-Layer Graph Convolutional Network (GCN)
  - Nodes: merchants (with tabular features)
  - Edges: transaction relationships (weighted by volume)
  - Task: Node-level binary classification (default / no default)
  - Message Passing: aggregates risk signals from 1st/2nd/3rd degree neighbors

Uses: PyTorch Geometric (torch_geometric)

Saves: models/gate2b_gnn.pt  +  models/gate2b_gnn_scaler.joblib

Run: python scripts/05_train_gate2b_gnn.py

NOTE: If PyTorch Geometric is not installed, this script falls back
to a NetworkX + gradient boosting approximation (still powerful).
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, MODELS_DIR, GNN_CONFIG

GNN_NODE_FEATURES = [
    "bsi_score", "revenue_cv", "transaction_entropy",
    "settlement_regularity", "active_days_ratio", "avg_daily_revenue",
    "revenue_trend_slope", "device_session_entropy", "business_age_days",
    "merchant_category", "city_tier", "has_soundbox",
]


# ─── Try PyTorch Geometric first, fallback to GraphSAGE-equivalent ────────────
def try_torch_geometric():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        return torch, F, Data, GCNConv
    except ImportError:
        return None, None, None, None


class GCNModel:
    """Wrapper for 3-layer GCN using PyTorch Geometric."""

    def __init__(self, in_channels, hidden_channels, dropout):
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GCNConv
        import torch.nn.functional as F

        self.torch = torch
        self.F = F

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_channels // 2, 1)

            def forward(self, x, edge_index, edge_weight=None):
                x = F.relu(self.conv1(x, edge_index, edge_weight))
                x = self.dropout(x)
                x = F.relu(self.conv2(x, edge_index, edge_weight))
                x = self.dropout(x)
                x = F.relu(self.conv3(x, edge_index, edge_weight))
                x = self.classifier(x)
                return x.squeeze(-1)

        self.model = Net()

    def train_model(self, data, epochs=100, lr=0.01):
        import torch
        import torch.nn.functional as F

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        
        self.model.train()
        losses = []
        aucs = []

        print(f"\n  Training GCN ({epochs} epochs)...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.edge_weight)
            
            # Weighted BCE for class imbalance
            pos_weight = torch.tensor([5.0])
            loss = F.binary_cross_entropy_with_logits(
                out[data.train_mask], data.y[data.train_mask].float(),
                pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(
                        self.model(data.x, data.edge_index, data.edge_weight)
                    )
                    val_probs = probs[data.val_mask].numpy()
                    val_labels = data.y[data.val_mask].numpy()
                    try:
                        auc = roc_auc_score(val_labels, val_probs)
                    except Exception:
                        auc = 0.5
                    aucs.append(auc)
                    losses.append(loss.item())
                self.model.train()
                print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val AUC: {auc:.4f}")

        return losses, aucs


class GraphSAGEFallback:
    """
    Fallback when PyTorch Geometric is not available.
    Implements neighbor feature aggregation manually using
    NetworkX + gradient boosting — a powerful GNN approximation.
    """

    def __init__(self):
        from sklearn.ensemble import GradientBoostingClassifier
        self.clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            random_state=42
        )

    def aggregate_neighbors(self, df, edges, node_features, depth=2):
        """Aggregate neighbor features up to `depth` hops."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(df["merchant_id"])
        for _, row in edges.iterrows():
            G.add_edge(row["src"], row["dst"], weight=row["weight"])

        id_to_idx = {mid: i for i, mid in enumerate(df["merchant_id"])}
        X = node_features.copy()
        X_aug = [X]

        for d in range(1, depth + 1):
            agg_features = np.zeros_like(X)
            for mid, i in id_to_idx.items():
                neighbors = list(G.predecessors(mid)) + list(G.successors(mid))
                if neighbors:
                    nb_idx = [id_to_idx[nb] for nb in neighbors if nb in id_to_idx]
                    if nb_idx:
                        agg_features[i] = X[nb_idx].mean(axis=0)
            X_aug.append(agg_features)

        return np.hstack(X_aug)

    def fit(self, X_aug, y, X_val, y_val):
        from sklearn.utils import class_weight
        cw = class_weight.compute_sample_weight("balanced", y)
        self.clf.fit(X_aug, y, sample_weight=cw)
        val_probs = self.clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        print(f"\n  GraphSAGE-Fallback Validation AUC: {auc:.4f}")
        return auc

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


def prepare_torch_data(df, edges, node_features_scaled):
    import torch
    from torch_geometric.data import Data

    # Build index mappings
    id_to_idx = {mid: i for i, mid in enumerate(df["merchant_id"])}

    # Edge index
    edge_src, edge_dst, edge_weights = [], [], []
    for _, row in edges.iterrows():
        if row["src"] in id_to_idx and row["dst"] in id_to_idx:
            edge_src.append(id_to_idx[row["src"]])
            edge_dst.append(id_to_idx[row["dst"]])
            edge_weights.append(row["weight"])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    # Normalize edge weights
    edge_weight = edge_weight / (edge_weight.max() + 1e-6)

    x = torch.tensor(node_features_scaled, dtype=torch.float)
    y = torch.tensor(df["is_default"].values, dtype=torch.long)

    # Train/val/test split (70/15/15)
    n = len(df)
    idx = np.random.permutation(n)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:train_size]] = True
    val_mask[idx[train_size:train_size + val_size]] = True
    test_mask[idx[train_size + val_size:]] = True

    data = Data(
        x=x, y=y,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return data


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Gate 2B: Cascade Risk GNN")
    print("=" * 60)

    # Load data (non-ghost only — ghosts were caught at Gate 1)
    df = pd.read_csv(PROCESSED_DIR / "full_features.csv")
    df_model = df[df["is_ghost"] == 0].reset_index(drop=True).copy()
    edges = pd.read_csv(PROCESSED_DIR / "graph_edges.csv")

    print(f"  Merchants: {len(df_model):,}")
    print(f"  Graph edges: {len(edges):,}")
    print(f"  Default rate: {df_model['is_default'].mean():.1%}")

    # Prepare node features
    X = df_model[GNN_NODE_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    torch, F, Data, GCNConv = try_torch_geometric()

    if torch is not None:
        print("\n  ✓ PyTorch Geometric available — using GCN")

        data = prepare_torch_data(df_model, edges, X_scaled)

        gcn = GCNModel(
            in_channels=len(GNN_NODE_FEATURES),
            hidden_channels=GNN_CONFIG["hidden_channels"],
            dropout=GNN_CONFIG["dropout"]
        )
        losses, aucs = gcn.train_model(data, epochs=GNN_CONFIG["epochs"],
                                        lr=GNN_CONFIG["learning_rate"])

        # Final evaluation
        gcn.model.eval()
        with torch.no_grad():
            all_probs = torch.sigmoid(
                gcn.model(data.x, data.edge_index, data.edge_weight)
            ).numpy()
        test_labels = data.y[data.test_mask].numpy()
        test_probs = all_probs[data.test_mask.numpy()]
        final_auc = roc_auc_score(test_labels, test_probs)
        print(f"\n  ── Final Test Results ──────────────────────────")
        print(f"  Test ROC-AUC: {final_auc:.4f}")

        # Save
        model_path = MODELS_DIR / "gate2b_gnn.pt"
        torch.save(gcn.model.state_dict(), model_path)
        meta = {
            "scaler": scaler,
            "features": GNN_NODE_FEATURES,
            "hidden_channels": GNN_CONFIG["hidden_channels"],
            "dropout": GNN_CONFIG["dropout"],
            "final_auc": final_auc,
            "mode": "gcn",
        }
        joblib.dump(meta, MODELS_DIR / "gate2b_gnn_meta.joblib")
        print(f"  ✓ GCN model saved: {model_path}")

    else:
        print("\n  ⚠ PyTorch Geometric not found — using GraphSAGE Fallback")
        print("    (Install torch-geometric for full GCN capability)")

        gnn_fallback = GraphSAGEFallback()

        print("  Aggregating neighbor features (2-hop)...")
        X_aug = gnn_fallback.aggregate_neighbors(df_model, edges, X_scaled, depth=2)

        # Train/val split
        n = len(df_model)
        idx = np.random.permutation(n)
        train_idx = idx[:int(0.8 * n)]
        val_idx = idx[int(0.8 * n):]

        y = df_model["is_default"].values
        auc = gnn_fallback.fit(
            X_aug[train_idx], y[train_idx],
            X_aug[val_idx], y[val_idx]
        )

        # All predictions
        all_probs = gnn_fallback.predict_proba(X_aug)
        print(f"  Final ROC-AUC: {auc:.4f}")

        model_path = MODELS_DIR / "gate2b_gnn_fallback.joblib"
        meta = {
            "model": gnn_fallback,
            "scaler": scaler,
            "features": GNN_NODE_FEATURES,
            "auc": auc,
            "mode": "fallback",
        }
        joblib.dump(meta, model_path)
        print(f"  ✓ Fallback GNN saved: {model_path}")

    # Save GNN risk scores back to features
    # (Use neighbor_avg_default_rate as proxy if GNN not fully trained)
    print("\n✅ Gate 2B (GNN) training complete!")
    print("Next: python scripts/06_train_gate3_ensemble.py")
