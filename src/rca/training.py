import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import pickle
import os
from typing import Optional


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()  # With sigmoid, suitable for one-hot labels
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Record best validation loss
    best_val_loss = float("inf")
    best_model_path = "best_rca_model.pth"

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
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features, graphs)

                # Compute loss
                loss = criterion(outputs, labels)
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
                labels = labels.to(device)

                outputs = model(features, graphs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        # Learning rate scheduling
        scheduler.step()

        # Print training info
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (Val Loss: {best_val_loss:.6f})")

        # Print test details every 2 epochs
        if (epoch + 1) % 2 == 0:
            print("-" * 50)
            test_results = test_model(model, val_loader, device)
            for metric, value in test_results.items():
                print(f"{metric}: {value:.4f}")

    return model


def test_model(model, test_loader, device, top_k_list=[1, 3, 5]):
    """Model test function, calculate Top-K accuracy for RCA task"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, graphs, labels in test_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = labels.to(device)

            # Model prediction (use sigmoid to get probabilities)
            outputs = torch.sigmoid(model(features, graphs))

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate Top-K accuracy
    results = {}
    for k in top_k_list:
        # Ensure k does not exceed number of nodes
        k = min(k, all_preds.shape[1])
        correct = 0

        for pred, label in zip(all_preds, all_labels):
            # Get true root cause positions
            true_root = np.where(label == 1)[0]
            if len(true_root) == 0:
                continue  # Skip samples without label

            # Get top-k nodes with highest predicted probabilities
            top_k_pred = np.argsort(pred)[-k:][::-1]

            # Check if true root cause is in Top-K
            if np.any(np.isin(true_root, top_k_pred)):
                correct += 1

        accuracy = correct / len(all_preds)
        results[f"Top-{k} Accuracy"] = accuracy
        print(f"Top-{k} Accuracy: {accuracy:.4f}")

    return results


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


def prepare_data_loaders(dataset, batch_size=32, train_ratio=0.6, val_ratio=0.2):
    """Prepare train, validation, and test data loaders"""
    from .dataset import collate_fn

    # Split train, validation, and test sets
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
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
