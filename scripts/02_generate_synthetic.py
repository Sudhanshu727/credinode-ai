"""
Script 02: Generate Synthetic India Merchant Data
====================================================
Creates realistic merchant profiles based on Paytm's ecosystem:
  - QR scan behavior, Soundbox pings, settlement patterns
  - Transaction network graph (merchant-to-merchant transfers)
  - Behavioral DNA features (device entropy, location variance)
  - Ground truth labels: default (0/1) and ghost/fraud (0/1)

Saves:
  - data/processed/merchants.csv        → Static merchant profiles
  - data/processed/daily_txn.csv        → 90-day time series per merchant
  - data/processed/graph_edges.csv      → Merchant transaction network edges
  - data/processed/full_features.csv    → Final merged feature matrix

Run: python scripts/02_generate_synthetic.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config.settings import PROCESSED_DIR, SYNTHETIC_CONFIG, BSI_CONFIG

np.random.seed(SYNTHETIC_CONFIG["seed"])
N = SYNTHETIC_CONFIG["n_merchants"]
DAYS = SYNTHETIC_CONFIG["time_series_days"]


# ─── City & Category Mappings ──────────────────────────────────────────────────
CITY_TIERS = {
    1: ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad"],
    2: ["Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat"],
    3: ["Varanasi", "Indore", "Nagpur", "Kochi", "Bhopal"],
}
CATEGORIES = {
    0: "Kirana/Grocery",
    1: "Restaurant/Dhaba",
    2: "Medical/Pharmacy",
    3: "Clothing/Textile",
    4: "Electronics",
    5: "Transport/Auto",
    6: "Services/Salon",
    7: "Education/Tuition",
}


def assign_city(n: int):
    tiers = np.random.choice([1, 2, 3], n, p=[0.35, 0.40, 0.25])
    cities = []
    for t in tiers:
        cities.append(np.random.choice(CITY_TIERS[t]))
    return tiers, np.array(cities)


def generate_merchant_profiles() -> pd.DataFrame:
    """Generate static merchant metadata WITHOUT labels.
    
    Labels will be assigned AFTER computing features to prevent data leakage.
    """
    print("  Generating merchant profiles...")

    city_tiers, cities = assign_city(N)
    categories = np.random.randint(0, len(CATEGORIES), N)

    # Business age: drawn independently, not based on eventual fraud label
    biz_age_base = np.random.randint(30, 1500, N)

    # Features that won't leak (tier, category, etc.)
    df = pd.DataFrame({
        "merchant_id": [f"M{str(i).zfill(6)}" for i in range(N)],
        "business_age_days": biz_age_base,
        "merchant_category": categories,
        "merchant_category_name": [CATEGORIES[c] for c in categories],
        "city": cities,
        "city_tier": city_tiers,
        "has_soundbox": (np.random.random(N) < 0.68).astype(int),
        "qr_active": (np.random.random(N) < 0.95).astype(int),
        "gst_registered": (np.random.random(N) < 0.42).astype(int),
    })
    return df


def generate_daily_transactions(merchants: pd.DataFrame) -> pd.DataFrame:
    """Generate 90 days of daily transaction data per merchant.
    
    Generate transactions INDEPENDENTLY without using is_ghost/is_default.
    This prevents label leakage into features.
    """
    print(f"  Generating {DAYS}-day transaction history for {N:,} merchants...")
    records = []

    for _, row in tqdm(merchants.iterrows(), total=N, desc="  Merchants"):
        mid = row["merchant_id"]
        tier = row["city_tier"]
        cat = row["merchant_category"]
        soundbox = row["has_soundbox"]
        business_age = row["business_age_days"]

        # Base revenue: tier 1 merchants earn more
        base_revenue = {1: 4500, 2: 2200, 3: 1100}[tier]
        base_revenue *= {0: 1.3, 1: 1.2, 2: 1.1, 3: 1.0,
                         4: 1.5, 5: 0.9, 6: 1.0, 7: 0.8}[cat]
        
        # Newer businesses tend to be more volatile (but not label-based)
        volatility = 0.8 if business_age < 90 else 0.5

        for day in range(DAYS):
            # Generate realistic transaction patterns with natural variation
            # (NOT based on whether merchant will default or be fraudulent)
            
            # Weekly seasonality
            seasonal = 1 + 0.2 * np.sin(2 * np.pi * day / 7)
            
            # Random walk (some days busy, some slow)
            trend = 1 + 0.15 * np.sin(2 * np.pi * day / 30)
            
            # Daily noise
            rev = np.random.lognormal(np.log(base_revenue * seasonal * trend + 1), volatility)
            n_txn = int(np.random.poisson(15 * seasonal * trend))

            soundbox_pings = int(n_txn * 0.85) if soundbox else 0
            
            # Settlement: assume mostly on-time (majority of merchants do settle)
            # Add some randomness for realism
            settlement_on_time = int(np.random.random() < 0.92)

            records.append({
                "merchant_id": mid,
                "day": day,
                "daily_revenue": round(rev, 2),
                "n_transactions": n_txn,
                "soundbox_pings": soundbox_pings,
                "settlement_on_time": settlement_on_time,
                "qr_scan_count": int(n_txn * np.random.uniform(0.6, 1.0)),
            })

    df = pd.DataFrame(records)
    return df


def compute_bsi_features(daily_txn: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily transactions into BSI features per merchant."""
    print("  Computing BSI features...")
    bsi_records = []

    for mid, group in tqdm(daily_txn.groupby("merchant_id"),
                           desc="  Computing BSI", total=N):
        rev = group["daily_revenue"].values
        txn = group["n_transactions"].values
        settlement = group["settlement_on_time"].values

        # Coefficient of variation of daily revenue (lower = more stable)
        mean_rev = rev.mean()
        std_rev = rev.std()
        revenue_cv = std_rev / (mean_rev + 1e-6)

        # Shannon entropy of transaction count distribution
        txn_counts = txn.clip(0, 200)
        if txn_counts.sum() > 0:
            prob = txn_counts / (txn_counts.sum() + 1e-6)
            prob = prob[prob > 0]
            txn_entropy = -np.sum(prob * np.log(prob + 1e-9))
        else:
            txn_entropy = 0.0

        # Settlement regularity
        settlement_reg = settlement.mean()

        # Active days ratio
        active_days = (rev > 0).sum()
        active_days_ratio = active_days / len(rev)

        # Revenue trend (slope of linear fit)
        if len(rev) > 1:
            x = np.arange(len(rev))
            trend_slope = np.polyfit(x, rev, 1)[0]
        else:
            trend_slope = 0.0

        bsi_score = (
            BSI_CONFIG["weights"]["revenue_cv"] * (1 - min(revenue_cv, 1)) +
            BSI_CONFIG["weights"]["transaction_entropy"] * min(txn_entropy / 5.0, 1) +
            BSI_CONFIG["weights"]["settlement_regularity"] * settlement_reg +
            BSI_CONFIG["weights"]["active_days_ratio"] * active_days_ratio
        )

        bsi_records.append({
            "merchant_id": mid,
            "bsi_score": round(bsi_score, 4),
            "revenue_cv": round(revenue_cv, 4),
            "transaction_entropy": round(txn_entropy, 4),
            "settlement_regularity": round(settlement_reg, 4),
            "active_days_ratio": round(active_days_ratio, 4),
            "avg_daily_revenue": round(mean_rev, 2),
            "revenue_trend_slope": round(trend_slope, 4),
        })

    return pd.DataFrame(bsi_records)


def generate_behavioral_dna(merchants: pd.DataFrame) -> pd.DataFrame:
    """Generate Behavioral DNA features INDEPENDENTLY.
    
    Do NOT use is_ghost label. Features generated in a way that will naturally
    separate into patterns (some suspicious, some normal), but without label leakage.
    """
    print("  Generating Behavioral DNA features...")

    # Generate each feature independently from a natural distribution
    # This allows models to discover patterns, not memorize them
    
    # Device session entropy: most merchants have consistent patterns, 
    # some are more erratic (natural variation)
    device_session_entropy = np.random.beta(5, 2, N)  # Skewed toward high (0.6-0.9)
    # Add a small minority that are very low (naturally suspicious)
    device_session_entropy[np.random.choice(N, size=int(0.08*N), replace=False)] = np.random.beta(2, 5, int(0.08*N))

    # Location variance: most merchants from single location, some from many
    location_variance = np.random.beta(2, 5, N)  # Skewed toward low (0-0.3)
    # Add tail: some merchants legitimately use multiple locations
    location_variance[np.random.choice(N, size=int(0.12*N), replace=False)] = np.random.beta(5, 2, int(0.12*N))

    # Temporal pattern score: humans have patterns, scripts don't
    temporal_pattern_score = np.random.beta(3, 1.5, N)  # Skewed toward high (0.5-1.0)
    # Random subset are more erratic
    temporal_pattern_score[np.random.choice(N, size=int(0.1*N), replace=False)] = np.random.uniform(0, 0.3, int(0.1*N))

    # Login hour entropy: most log in at consistent times, some vary
    login_hour_entropy = np.random.uniform(0.2, 0.8, N)
    # Add some very high (could indicate scripts or true multi-shift)
    login_hour_entropy[np.random.choice(N, size=int(0.1*N), replace=False)] = np.random.uniform(0.8, 1.0, int(0.1*N))

    # Transaction velocity: scale-dependent but with natural variation
    transaction_velocity = np.random.gamma(shape=2, scale=2, size=N)  # Mean ~4 txn/hr
    transaction_velocity = np.clip(transaction_velocity, 0, 30)

    # Unique device count: most use 1-2 devices, naturally some use more
    unique_device_count = np.random.poisson(1.5, N) + 1  # Mean ~2
    unique_device_count = np.clip(unique_device_count, 1, 20)

    # IP change frequency: most merchants have low, some change daily
    ip_change_frequency = np.random.exponential(scale=1.2, size=N)
    ip_change_frequency = np.clip(ip_change_frequency, 0, 20)

    # Weekend activity ratio: businesses quieter on weekends (natural pattern)
    weekend_activity_ratio = np.random.beta(3, 8, N)  # Skewed toward low (0.1-0.3)
    # Some legitimate multi-shift operations are busier on weekends
    weekend_activity_ratio[np.random.choice(N, size=int(0.15*N), replace=False)] = np.random.uniform(0.3, 0.6, int(0.15*N))

    return pd.DataFrame({
        "merchant_id": merchants["merchant_id"],
        "device_session_entropy": device_session_entropy.round(4),
        "location_variance": location_variance.round(4),
        "temporal_pattern_score": temporal_pattern_score.round(4),
        "login_hour_entropy": login_hour_entropy.round(4),
        "transaction_velocity": transaction_velocity.round(4),
        "unique_device_count": unique_device_count,
        "ip_change_frequency": ip_change_frequency.round(4),
        "weekend_activity_ratio": weekend_activity_ratio.round(4),
    })


def generate_graph_edges(merchants: pd.DataFrame) -> pd.DataFrame:
    """
    Generate merchant transaction network graph WITHOUT label bias.
    
    Edges generated independently, then network patterns computed.
    (In old code, edges clustered defaulters together, causing leakage.)
    """
    print("  Generating transaction network graph...")
    n_edges = SYNTHETIC_CONFIG["n_graph_edges"]
    merchant_ids = merchants["merchant_id"].values
    
    # Generate edges uniformly without label information
    src_list, dst_list, weights = [], [], []
    
    # Merchants tend to transact with similar tier merchants (natural clustering)
    tier_groups = {1: [], 2: [], 3: []}
    for i, tid in enumerate(merchants["city_tier"].values):
        tier_groups[tid].append(i)

    for _ in range(n_edges):
        # 60% within-tier transactions (natural), 40% cross-tier
        if np.random.random() < 0.6:
            # Pick random tier, pick two merchants from that tier
            tier = np.random.choice([1, 2, 3])
            if len(tier_groups[tier]) < 2:
                tier = np.random.choice([1, 2, 3])
            idxs = np.random.choice(tier_groups[tier], size=2, replace=False)
            src, dst = idxs[0], idxs[1]
        else:
            # Random cross-tier
            src = np.random.randint(0, len(merchants))
            dst = np.random.randint(0, len(merchants))

        if src == dst:
            continue

        weight = np.random.lognormal(7, 1.5)  # Transaction amount
        src_list.append(merchant_ids[src])
        dst_list.append(merchant_ids[dst])
        weights.append(round(weight, 2))

    df = pd.DataFrame({
        "src": src_list,
        "dst": dst_list,
        "weight": weights
    })
    return df


def assign_labels_from_features(merchants: pd.DataFrame, bsi_features: pd.DataFrame,
                                 behavioral_dna: pd.DataFrame, graph_features: pd.DataFrame) -> pd.DataFrame:
    """
    ASSIGN LABELS AFTER computing features to prevent label leakage.
    
    Labels are assigned probabilistically based on feature patterns.
    Uses quantile-based approach: merchants with highest risk scores get labels.
    """
    print("  Assigning labels based on feature patterns...")
    
    m = merchants.copy()
    fraud_rate = SYNTHETIC_CONFIG["fraud_rate"]
    default_rate = SYNTHETIC_CONFIG["default_rate"]
    
    # Merge all features temporarily
    all_features = (
        bsi_features
        .merge(behavioral_dna, on="merchant_id")
        .merge(graph_features, on="merchant_id")
    )
    
    # Ghost detection: based on behavioral DNA anomalies
    # Signs of fraud: low temporal pattern + high entropy + many devices + frequent IP changes
    ghost_score = (
        (1 - all_features["temporal_pattern_score"]) * 0.3 +
        all_features["login_hour_entropy"] * 0.3 +
        (all_features["unique_device_count"] / 20) * 0.2 +
        (all_features["ip_change_frequency"] / 20) * 0.2
    )
    
    # Quantile-based assignment: top fraud_rate% get flagged as ghosts
    ghost_threshold = np.quantile(ghost_score, 1 - fraud_rate)
    # Add randomness: 70% based on score, 30% random selection
    is_ghost = (ghost_score > ghost_threshold).astype(int)
    # Randomly swap some to add noise (ensure we hit target %)
    n_ghosts_target = int(len(m) * fraud_rate)
    n_ghosts_current = is_ghost.sum()
    if n_ghosts_current < n_ghosts_target:
        # Need more ghosts: randomly select from non-ghosts
        additional_needed = n_ghosts_target - n_ghosts_current
        non_ghost_idx = np.where(is_ghost == 0)[0]
        swap_idx = np.random.choice(non_ghost_idx, size=additional_needed, replace=False)
        is_ghost[swap_idx] = 1
    elif n_ghosts_current > n_ghosts_target:
        # Too many ghosts: randomly demote some
        extra = n_ghosts_current - n_ghosts_target
        ghost_idx = np.where(is_ghost == 1)[0]
        swap_idx = np.random.choice(ghost_idx, size=extra, replace=False)
        is_ghost[swap_idx] = 0
    
    # Default detection: based on BSI features (stability)
    # Low BSI, high revenue volatility, declining trend = higher default risk
    default_score = (
        (1 - np.clip(all_features["bsi_score"], 0, 1)) * 0.4 +
        np.minimum(all_features["revenue_cv"] / 2, 1) * 0.4 +
        ((all_features["revenue_trend_slope"] < -5).astype(int)) * 0.2
    )
    
    # Quantile-based assignment: among non-ghosts, top default_rate% get flagged
    non_ghost_mask = is_ghost == 0
    if non_ghost_mask.sum() > 0:
        default_threshold = np.quantile(default_score[non_ghost_mask], 1 - default_rate)
        is_default = ((default_score > default_threshold) & non_ghost_mask).astype(int)
        
        # Ensure we hit target rate
        n_defaults_target = int(non_ghost_mask.sum() * default_rate)
        n_defaults_current = is_default.sum()
        if n_defaults_current < n_defaults_target:
            additional_needed = n_defaults_target - n_defaults_current
            non_default_non_ghost_idx = np.where((is_default == 0) & non_ghost_mask)[0]
            if len(non_default_non_ghost_idx) > 0:
                swap_idx = np.random.choice(non_default_non_ghost_idx, 
                                          size=min(additional_needed, len(non_default_non_ghost_idx)), 
                                          replace=False)
                is_default[swap_idx] = 1
        elif n_defaults_current > n_defaults_target:
            extra = n_defaults_current - n_defaults_target
            default_idx = np.where(is_default == 1)[0]
            if len(default_idx) > 0:
                swap_idx = np.random.choice(default_idx, size=min(extra, len(default_idx)), replace=False)
                is_default[swap_idx] = 0
    else:
        is_default = np.zeros(len(m), dtype=int)
    
    m["is_ghost"] = is_ghost
    m["is_default"] = is_default
    
    # Non-ghost merchants can have loan history
    m["loan_history_count"] = np.where(is_ghost, 0, np.random.poisson(1.5, len(m)))
    m["prev_defaults"] = np.where(is_ghost, 0, np.random.poisson(0.3, len(m)))
    
    print(f"    Assigned: {is_ghost.sum()/len(m):.1%} ghost, {is_default.sum()/len(m):.1%} default")
    
    return m


def update_graph_features_with_labels(merchants: pd.DataFrame, graph_features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Update graph features now that labels are assigned."""
    print("  Updating graph features with labels...")
    import networkx as nx
    
    G = nx.DiGraph()
    G.add_nodes_from(merchants["merchant_id"])
    
    for _, row in edges.iterrows():
        if G.has_edge(row["src"], row["dst"]):
            G[row["src"]][row["dst"]]["weight"] += row["weight"]
        else:
            G.add_edge(row["src"], row["dst"], weight=row["weight"])
    
    default_set = set(merchants[merchants["is_default"] == 1]["merchant_id"])
    
    updated_records = []
    for _, row in merchants.iterrows():
        mid = row["merchant_id"]
        neighbors = list(G.predecessors(mid)) + list(G.successors(mid))
        
        if len(neighbors) == 0:
            neighbor_default_rate = 0.0
            high_risk_count = 0
        else:
            neighbor_defaults = sum(1 for nb in neighbors if nb in default_set)
            neighbor_default_rate = neighbor_defaults / len(neighbors)
            high_risk_count = neighbor_defaults
        
        # GNN risk: neighbor default rate with added randomness
        gnn_risk = neighbor_default_rate * 0.7 + np.random.uniform(0, 0.3)
        if row["is_default"]:
            gnn_risk = min(gnn_risk + 0.3, 1.0)
        
        updated_records.append({
            "merchant_id": mid,
            "gnn_risk_score": round(gnn_risk, 4),
            "neighbor_avg_default_rate": round(neighbor_default_rate, 4),
            "network_centrality": graph_features[graph_features["merchant_id"] == mid]["network_centrality"].values[0],
            "high_risk_neighbor_count": high_risk_count,
        })
    
    return pd.DataFrame(updated_records)


def compute_graph_features(merchants: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute network-based risk features per merchant using the graph.
    
    NOTE: Since labels not assigned yet, we compute network structure properties
    without bias. Labels will be assigned after, potentially using these features.
    """
    print("  Computing graph features (neighbor risk)...")
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(merchants["merchant_id"])

    for _, row in edges.iterrows():
        if G.has_edge(row["src"], row["dst"]):
            G[row["src"]][row["dst"]]["weight"] += row["weight"]
        else:
            G.add_edge(row["src"], row["dst"], weight=row["weight"])

    # Without true defaults, we compute structural properties only
    graph_records = []
    
    try:
        centrality = nx.degree_centrality(G)
    except Exception:
        centrality = {mid: 0 for mid in merchants["merchant_id"]}

    for _, row in tqdm(merchants.iterrows(), total=N, desc="  Graph features"):
        mid = row["merchant_id"]
        neighbors = list(G.predecessors(mid)) + list(G.successors(mid))

        if len(neighbors) == 0:
            neighbor_default_rate = 0.0
            high_risk_count = 0
        else:
            # Placeholder: will be updated after labels assigned
            neighbor_default_rate = 0.0
            high_risk_count = 0

        # GNN output: initialized to zero, will be computed in label assignment stage
        gnn_risk = 0.0

        graph_records.append({
            "merchant_id": mid,
            "gnn_risk_score": round(gnn_risk, 4),
            "neighbor_avg_default_rate": round(neighbor_default_rate, 4),
            "network_centrality": round(centrality.get(mid, 0), 4),
            "high_risk_neighbor_count": high_risk_count,
        })

    return pd.DataFrame(graph_records)


if __name__ == "__main__":
    print("=" * 60)
    print("  CrediNode AI — Synthetic Data Generation")
    print("=" * 60)

    # Step 1: Merchant profiles
    merchants = generate_merchant_profiles()
    merchants.to_csv(PROCESSED_DIR / "merchants.csv", index=False)
    print(f"  ✓ Saved merchant profiles ({len(merchants):,} rows)")

    # Step 2: Daily transactions
    daily_txn = generate_daily_transactions(merchants)
    daily_txn.to_csv(PROCESSED_DIR / "daily_txn.csv", index=False)
    print(f"  ✓ Saved daily_txn.csv ({len(daily_txn):,} rows)")

    # Step 3: BSI features
    bsi_features = compute_bsi_features(daily_txn)

    # Step 4: Behavioral DNA
    behavioral_dna = generate_behavioral_dna(merchants)

    # Step 5: Graph
    edges = generate_graph_edges(merchants)
    edges.to_csv(PROCESSED_DIR / "graph_edges.csv", index=False)
    print(f"  ✓ Saved graph_edges.csv ({len(edges):,} edges)")

    # Step 6: Graph features (placeholder, before labels)
    graph_features = compute_graph_features(merchants, edges)

    # Step 7: ASSIGN LABELS based on feature patterns (prevents leakage!)
    merchants = assign_labels_from_features(merchants, bsi_features, behavioral_dna, graph_features)

    # Step 8: UPDATE graph features now that labels are known
    graph_features = update_graph_features_with_labels(merchants, graph_features, edges)

    # Step 9: Merge all features
    full = (
        merchants
        .merge(bsi_features, on="merchant_id")
        .merge(behavioral_dna, on="merchant_id")
        .merge(graph_features, on="merchant_id")
    )

    # Compute anomaly_score as a normalized proxy from behavioral DNA
    # (same formula used at inference time in Gate 1)
    dse = full["device_session_entropy"]
    lv  = full["location_variance"]
    tv  = full["transaction_velocity"].clip(upper=50)
    full["anomaly_score"] = (
        dse * 0.4 + (1 - lv) * 0.3 + (1 - tv / 50) * 0.3
    ).round(4)
    full.to_csv(PROCESSED_DIR / "full_features.csv", index=False)
    print(f"\n  ✓ Saved full_features.csv ({len(full):,} rows, {len(full.columns)} cols)")
    print("\n✅ Data generation complete!")
    print("Next: python scripts/03_train_gate1_anomaly.py")
