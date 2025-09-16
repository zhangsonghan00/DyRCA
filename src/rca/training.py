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
    logits: (N,), 每个节点的预测得分
    root_idx: 根因节点的索引
    graph: DGLGraph，单个调用图
    margin: pairwise 的最小分数间隔
    max_depth: 最大传播链长度
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

    # 防止根因节点被当作负样本
    affected_nodes.discard(root_idx)

    for neg_idx in affected_nodes:
        diff = margin - logits[root_idx] + logits[neg_idx]
        loss += torch.relu(diff) 
        total_pairs += 1

    if total_pairs == 0:
        return torch.tensor(0.0, device=device)
    return loss / total_pairs


def train_model(model, output_model, train_loader, val_loader, device, num_epochs=50, lr=1e-3):
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

            for features, graphs, labels in tepoch:
                # Move data to device
                features = features.to(device)
                graphs = graphs.to(device)
                # labels = labels.to(device)
                labels = torch.argmax(labels, dim=1).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features, graphs)

                # Compute loss
                loss = criterion(outputs, labels)

                # add pairwise-ranking loss
                batched_graphs = dgl.unbatch(graphs)
                pairwise_losses = []
                for i in range(len(batched_graphs)):
                    g = batched_graphs[i]
                    root = labels[i].item()
                    logits = outputs[i]
                    loss_r = bidirectional_pairwise_ranking_loss(logits, root, g, margin=0.1, max_depth=3)
                    pairwise_losses.append(loss_r)
                pairwise_loss = torch.stack(pairwise_losses).mean()

                loss = loss + 0.5 * pairwise_loss
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
            for features, graphs, labels in val_loader:
                features = features.to(device)
                graphs = graphs.to(device)
                # labels = labels.to(device)
                labels = torch.argmax(labels, dim=1).to(device)

                outputs = model(features, graphs)
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


# def test_model(model, test_loader, device, top_k_list=[1, 3, 5]):
#     """Model test function, calculate Top-K accuracy for RCA task"""
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for features, graphs, labels in test_loader:
#             features = features.to(device)
#             graphs = graphs.to(device)
#             # labels = labels.to(device)
#             labels = torch.argmax(labels, dim=1).to(device)

#             # Model prediction (use sigmoid to get probabilities)
#             outputs = torch.sigmoid(model(features, graphs))

#             all_preds.append(outputs.cpu().numpy())
#             all_labels.append(labels.cpu().numpy())

#     # # Concatenate all predictions and labels
#     # all_preds = np.concatenate(all_preds, axis=0)
#     # all_labels = np.concatenate(all_labels, axis=0)

#     # # Calculate Top-K accuracy
#     # results = {}
#     # top_k_results=[]
#     # for k in top_k_list:
#     #     # Ensure k does not exceed number of nodes
#     #     k = min(k, all_preds.shape[1])
#     #     correct = 0

#     #     for pred, label in zip(all_preds, all_labels):
#     #         # Get true root cause positions
#     #         true_root = np.where(label == 1)[0]
#     #         if len(true_root) == 0:
#     #             continue  # Skip samples without label

#     #         # Get top-k nodes with highest predicted probabilities
#     #         top_k_pred = np.argsort(pred)[-k:][::-1]

#     #         # Check if true root cause is in Top-K
#     #         if np.any(np.isin(true_root, top_k_pred)):
#     #             correct += 1
#     #         if k == 5:
#     #             top_k_results.append(top_k_pred)
#     #     accuracy = correct / len(all_preds)
#     #     results[f"Top-{k} Accuracy"] = accuracy
#     # print(f"Test Results: {results}")
#     # return results, top_k_results
#     all_preds = np.concatenate(all_preds, axis=0)
#     all_labels = np.concatenate(all_labels, axis=0)

#     # Calculate Top-K accuracy
#     results = {}
#     top_k_results = []
#     for k in top_k_list:
#         # Ensure k does not exceed number of nodes
#         k = min(k, all_preds.shape[1])
#         correct = 0

#         for pred, label in zip(all_preds, all_labels):
#             # Get true root cause (now a single integer class)
#             true_root = int(label)  # 直接使用标签值作为类别
            
#             # Get top-k nodes with highest predicted probabilities
#             top_k_pred = np.argsort(pred)[-k:][::-1]
            
#             # Check if true root cause is in Top-K predictions
#             if true_root in top_k_pred:
#                 correct += 1
                
#             if k == 5:
#                 top_k_results.append(top_k_pred)
        
#         accuracy = correct / len(all_preds)
#         results[f"Top-{k} Accuracy"] = accuracy

#     print(f"Test Results: {results}")
#     return results, top_k_results


def test_model(model, test_loader, device, top_k_list=[1, 3, 5]):
    alpha=0.985
    model.eval()
    all_preds = []
    all_logits = []
    all_labels = []
    all_rerank_results = []
    top_k_results = []

    with torch.no_grad():
        for features, graphs, labels in test_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = torch.argmax(labels, dim=1).to(device)

            logits = model(features, graphs)
            outputs = torch.sigmoid(logits)

            all_preds.append(outputs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # 支持 batch 多个图
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

                    # 收集所有 k 值的 top_k_nodes，而不仅是 k=5
                    if k == 5:
                        top_k_results.append(top_k_nodes)

    # 聚合统计
    results = {}
    for k in top_k_list:
        # 过滤出所有与当前 k 值相关的结果
        k_specific_results = [res for res in all_rerank_results 
                             if any(f"Top-{k} Accuracy" in key for key in res)]
        if not k_specific_results:
            print(f"警告: 没有找到 Top-{k} 的结果")
            continue
            
        raw_acc = np.mean([res[f"Top-{k} Accuracy"] for res in k_specific_results])
        results[f"Top-{k} Accuracy"] = raw_acc

    # print("=== RCA Top-K Accuracy with Soft Re-ranking (Inbound) ===")
    # for k, v in results.items():
    #     print(f"{k}: {v:.4f}")

    return results, top_k_results




def rerank_nodes_with_coverage_inbound(top_k_nodes, pred_scores, g, alpha=0.7, max_depth=None):
    """
    在 top-k 节点中，利用结构性传播关系进行 re-ranking。
    对于每个节点，计算它沿着前驱边(predecessors)链路中包含的top-k节点数量。
    
    参数:
        top_k_nodes: 初始的top-k节点列表
        pred_scores: 预测分数
        g: 图结构
        alpha: 预测分数的权重
        max_depth: 最大遍历深度，防止过深遍历，None表示不限制
    """
    influence_counts = []
    top_k_set = set(top_k_nodes)

    for node in top_k_nodes:
        # 存储已访问过的节点，防止循环遍历
        visited = set()
        # 待遍历的节点队列，格式为(当前节点, 当前深度)
        queue = [(node, 0)]
        # 计数链路中包含的top-k节点
        count = 0
        
        while queue:
            current_node, depth = queue.pop(0)  # 广度优先遍历
            
            # 检查是否超过最大深度
            if max_depth is not None and depth > max_depth:
                continue
                
            # 如果当前节点在top_k_set中且未被计数过
            if current_node in top_k_set and current_node not in visited:
                count += 1
                visited.add(current_node)
            # 获取前驱节点并继续遍历
            predecessors = g.predecessors(current_node).numpy()
            for pred in predecessors:
                if pred not in visited:
                    queue.append((pred, depth + 1))
        
        # 减去1是因为包含了节点本身，根据需求决定是否保留
        influence_counts.append(count - 1 if count > 0 else 0)

    coverages = influence_counts
    
    # 提取top_k节点对应的预测分数
    node_scores = [pred_scores[node] for node in top_k_nodes]
    
    # 标准化预测分数 (min-max标准化到[0,1]范围)
    min_score, max_score = min(node_scores), max(node_scores)
    if max_score > min_score:  # 避免除以零
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in node_scores]
    else:  # 所有分数都相同的情况
        normalized_scores = [0.5 for _ in node_scores]  # 都赋予中间值
    
    # 标准化覆盖率 (min-max标准化到[0,1]范围)
    min_cov, max_cov = min(coverages), max(coverages)
    if max_cov > min_cov:  # 避免除以零
        normalized_coverages = [(c - min_cov) / (max_cov - min_cov) for c in coverages]
    else:  # 所有覆盖率都相同的情况
        normalized_coverages = [0.5 for _ in coverages]  # 都赋予中间值
    
    # 使用标准化后的值计算最终分数
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
    # from torch.utils.data import Subset

    # # 按顺序分割数据集（假设 dataset 是 RCADataset 实例）
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.3 * len(dataset))

    # # 创建连续索引的子集
    # train_dataset = Subset(dataset, indices=range(0, train_size))
    # val_dataset = Subset(dataset, indices=range(train_size, train_size + val_size))
    # test_dataset = Subset(dataset, indices=range(train_size + val_size, len(dataset)))

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
