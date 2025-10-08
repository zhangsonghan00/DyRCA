import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pickle
import os
from typing import Optional
import typer
import dgl

def bidirectional_pairwise_ranking_loss(logits, root_idx, graph, margin=0.1, max_depth=2):
    """
    logits: (N,), prediction score for each node
    root_idx: index of the root cause node
    graph: DGLGraph, single call graph
    margin: minimum score margin for pairwise
    max_depth: maximum propagation chain length
    """
    device = logits.device
    loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    visited = set()
    frontier = {root_idx}
    affected_nodes = set()

    for _ in range(max_depth):
        next_frontier = set()

        for node in frontier:
            down = graph.successors(node).tolist()
            up = graph.predecessors(node).tolist()

            for nei in down + up:
                if nei not in visited:
                    affected_nodes.add(nei)
                    next_frontier.add(nei)

        visited |= frontier
        frontier = next_frontier

    # Prevent the root cause node from being treated as a negative sample
    affected_nodes.discard(root_idx)

    for neg_idx in affected_nodes:
        diff = margin - logits[root_idx] + logits[neg_idx]
        loss += torch.relu(diff) 
        total_pairs += 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)
    return loss / total_pairs

def cross_period_discrepancy_loss(anomaly_emb, normal_emb, root_idx, margin=0.1, distance_type="euclidean"):
    """
    CPD Loss based on high-dimensional intermediate representations
    anomaly_emb: GAT representation during anomaly period (B, N, E2) → single sample is (N, E2)
    normal_emb: GAT representation during normal period (B, N, E2) → single sample is (N, E2)
    root_idx: index of root cause node (B,) → single sample is scalar
    margin: minimum margin of discrepancy between root cause and other nodes
    distance_type: method for calculating discrepancy (euclidean/cosine/l1)
    """
    device = anomaly_emb.device
    batch_size, num_nodes, _ = anomaly_emb.size()
    loss = torch.tensor(0.0, device=device)
    total_pairs = 0

    for i in range(batch_size):
    # Abnormal/normal representation for a single sample
        a_emb = anomaly_emb[i]  # (N, E2)
        n_emb = normal_emb[i]  # (N, E2)
        root = root_idx[i].item()  # 单个样本的根因索引

    # 1. Calculate anomaly-normal discrepancy for each node
        if distance_type == "euclidean":
            discrepancy = torch.norm(a_emb - n_emb, dim=1)  # (N,), Euclidean distance per node
        elif distance_type == "cosine":
            discrepancy = 1 - F.cosine_similarity(a_emb, n_emb, dim=1)  # (N,), cosine distance
        elif distance_type == "l1":
            discrepancy = torch.norm(a_emb - n_emb, p=1, dim=1)  # (N,), L1 distance
        else:
            raise ValueError("distance_type must be euclidean/cosine/l1")

    # 2. Discrepancy of root cause node vs other nodes
        root_dis = discrepancy[root]
        other_nodes = [j for j in range(num_nodes) if j != root]
        if not other_nodes:
            continue

    # 3. Pairwise comparison: root cause discrepancy should be >= other node discrepancy + margin
        for node_j in other_nodes:
            diff = margin - (root_dis - discrepancy[node_j])
            loss += torch.relu(diff)
            total_pairs += 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)
    return loss / total_pairs


def train_model(model, output_model, train_loader, val_loader, device, num_epochs=20, lr=1e-3):
    # criterion = nn.BCEWithLogitsLoss()  # With sigmoid, suitable for one-hot labels
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # Record best validation loss
    best_val_loss = float("inf")
    best_model_path = output_model

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

    # Training process
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")

            for features, graphs, normal_features, normal_graphs, labels in tepoch:
                # Move data to device
                features = features.to(device)
                graphs = graphs.to(device)
                normal_features = normal_features.to(device)
                normal_graphs = normal_graphs.to(device)
                # labels = labels.to(device)
                labels = torch.argmax(labels, dim=1).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs, anomaly_gat_emb = model(features, graphs)
                normal_outputs, normal_gat_emb = model(normal_features, normal_graphs)

                # Compute loss
                loss = criterion(outputs, labels)

                # # add pairwise-ranking loss
                batched_graphs = dgl.unbatch(graphs)
                pairwise_losses = []
                for i in range(len(batched_graphs)):
                    g = batched_graphs[i]
                    root = labels[i].item()
                    logits = outputs[i]
                    loss_r = bidirectional_pairwise_ranking_loss(logits, root, g, margin=0.1, max_depth=3)
                    pairwise_losses.append(loss_r)
                pairwise_loss = torch.stack(pairwise_losses).mean()

                # add cross-period discrepancy loss
                cpd_loss = cross_period_discrepancy_loss(
                    anomaly_emb=anomaly_gat_emb,
                    normal_emb=normal_gat_emb,
                    root_idx=labels,
                    margin=0.1,
                    distance_type="cosine"  # Try cosine first, then adjust based on results
                )

                loss = loss + 0.2 * pairwise_loss + 0.1 * cpd_loss
                train_loss += loss.item() * features.size(0)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Update progress bar
                tepoch.set_postfix(loss=loss.item())

    # Calculate average training loss
        train_loss /= len(train_loader.dataset)

    # Validation process
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, graphs, normal_features, normal_graphs, labels in val_loader:
                features = features.to(device)
                graphs = graphs.to(device)
                # labels = labels.to(device)
                labels = torch.argmax(labels, dim=1).to(device)

                outputs, _ = model(features, graphs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

    # Learning rate scheduling
        scheduler.step()

    # Print training info
        typer.echo(f"Epoch {epoch + 1}/{num_epochs}")
        typer.echo(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        typer.echo(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

    # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            typer.echo(f"Saved best model (Val Loss: {best_val_loss:.6f})")
    # else:
    #     typer.echo(f"Early stopping triggered at epoch {epoch}")
    #     break

    # Print test details every 2 epochs
        if (epoch + 1) % 1 == 0:
            typer.echo("-" * 50)
            test_results,_ = test_model(model, val_loader, device)
            for metric, value in test_results.items():
                typer.echo(f"{metric}: {value:.4f}")

    return model


def test_model(model, test_loader, device, top_k_list=[1, 3, 5]):
    alpha=0.985
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    all_rerank_results = []
    top_k_results = []

    with torch.no_grad():
        for features, graphs, normal_features, normal_graphs, labels in test_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = torch.argmax(labels, dim=1).to(device)

            logits,_ = model(features, graphs)
            outputs = torch.sigmoid(logits)

            all_preds.append(outputs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Support batch multiple graphs
            graph_list = dgl.unbatch(graphs)
            pred_batch = outputs.cpu().numpy()
            label_batch = labels.cpu().numpy()

            for i, g in enumerate(graph_list):
                pred = pred_batch[i]
                label = int(label_batch[i])

                for k in top_k_list:
                    k = min(k, pred.shape[0])
                    top_k_nodes = np.argsort(pred)[-k:][::-1]
                    true_root = label

                    # baseline
                    hit_raw = (true_root in top_k_nodes)
                    all_rerank_results.append({
                        f"Top-{k} Accuracy": hit_raw,
                    })

                    # Collect top_k_nodes for all k values, not just k=5
                    if k == 5:
                        top_k_results.append(top_k_nodes)

    # Aggregate statistics
    results = {}
    for k in top_k_list:
    # Filter out all results related to the current k value
        k_specific_results = [res for res in all_rerank_results 
                             if any(f"Top-{k} Accuracy" in key for key in res)]
        if not k_specific_results:
            print(f"Warning: No results found for Top-{k}")
            continue
            
        raw_acc = np.mean([res[f"Top-{k} Accuracy"] for res in k_specific_results])
        results[f"Top-{k} Accuracy"] = raw_acc

    # print("=== RCA Top-K Accuracy with Soft Re-ranking (Inbound) ===")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")

    return results, top_k_results




def rerank_nodes_with_coverage_inbound(top_k_nodes, pred_scores, g, alpha=0.7, max_depth=None):
    """
    Among the top-k nodes, use structural propagation relationships for re-ranking.
    For each node, calculate the number of top-k nodes contained along the predecessor (predecessors) chain.

    Args:
        top_k_nodes: initial list of top-k nodes
        pred_scores: prediction scores
        g: graph structure
        alpha: weight of prediction scores
        max_depth: maximum traversal depth to prevent too deep traversal, None means no limit
    """
    influence_counts = []
    top_k_set = set(top_k_nodes)

    for node in top_k_nodes:
    # Store visited nodes to prevent circular traversal
        visited = set()
    # Queue of nodes to be traversed, format: (current node, current depth)
        queue = [(node, 0)]
    # Count the number of top-k nodes contained in the chain
        count = 0
        
        while queue:
            current_node, depth = queue.pop(0)  # Breadth-first traversal
            
            # Check if maximum depth is exceeded
            if max_depth is not None and depth > max_depth:
                continue
                
            # If the current node is in top_k_set and has not been counted
            if current_node in top_k_set and current_node not in visited:
                count += 1
                visited.add(current_node)
            # Get predecessor nodes and continue traversal
            predecessors = g.predecessors(current_node).numpy()
            for pred in predecessors:
                if pred not in visited:
                    queue.append((pred, depth + 1))
        
    # Subtract 1 because it includes the node itself, decide whether to keep it as needed
        influence_counts.append(count - 1 if count > 0 else 0)

    coverages = influence_counts
    
    # Extract prediction scores corresponding to top_k nodes
    node_scores = [pred_scores[node] for node in top_k_nodes]
    
    # Normalize prediction scores (min-max normalization to [0,1] range)
    min_score, max_score = min(node_scores), max(node_scores)
    if max_score > min_score:  # Avoid division by zero
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in node_scores]
    else:  # All scores are the same
        normalized_scores = [0.5 for _ in node_scores]  # All assigned middle value
    
    # Normalize coverage (min-max normalization to [0,1] range)
    min_cov, max_cov = min(coverages), max(coverages)
    if max_cov > min_cov:  # Avoid division by zero
        normalized_coverages = [(c - min_cov) / (max_cov - min_cov) for c in coverages]
    else:  # All coverages are the same
        normalized_coverages = [0.5 for _ in coverages]  # All assigned middle value
    
    # Use normalized values to calculate the final score
    final_scores = [
        alpha * norm_score + (1 - alpha) * norm_cov
        for norm_score, norm_cov in zip(normalized_scores, normalized_coverages)
    ]

    reranked = [x for _, x in sorted(zip(final_scores, top_k_nodes), reverse=True)]
    return reranked




def predict_single_case(model, features, graph, device, node_names=None):
    """Perform root cause analysis inference for a single case"""
    model.eval()

    with torch.no_grad():
    # Ensure input shape is [1, time_window, nodes, metrics]
        if features.dim() == 3:
            features = features.unsqueeze(0)

        features = features.to(device)
        graph = graph.to(device)

    # Model prediction
        outputs = torch.sigmoid(model(features, graph))
        probs = outputs.squeeze(0).cpu().numpy()  # [num_nodes]

    # Get prediction results
        top_k_indices = np.argsort(probs)[::-1]  # Descending order by probability

        results = []
        for i, node_idx in enumerate(top_k_indices):
            node_name = node_names[node_idx] if node_names else f"Node_{node_idx}"
            results.append(
                {
                    "rank": i + 1,
                    "node_id": int(node_idx),
                    "node_name": node_name,
                    "probability": float(probs[node_idx]),
                }
            )

        return results


def load_model_and_metadata(model_path: str, metadata_path: Optional[str] = None):
    """Load trained model and metadata"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # If there is a metadata file, load metadata
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

    return metadata

def prepare_data_loaders(train_dataset, test_dataset, batch_size=32, train_ratio=0.7, val_ratio=0.3, seed=42):
    """Prepare train, validation, and test data loaders with a fixed random seed"""
    from .dataset import collate_fn
    import torch

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Split train, validation, and test sets
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset1, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    return train_loader, val_loader, test_loader
