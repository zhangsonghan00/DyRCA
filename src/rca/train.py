import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from model import RCAModel


class RCATrainer:
    """RCA model trainer"""

    def __init__(
        self,
        model: RCAModel,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.anomaly_criterion = nn.CrossEntropyLoss()
        self.root_cause_criterion = nn.CrossEntropyLoss(
            ignore_index=-1
        )  # -1 means normal sample

    def train_step(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        anomaly_labels: torch.Tensor,
        root_cause_labels: torch.Tensor,
    ) -> float:
        """Perform one training step"""
        self.model.train()

        # Move to device
        node_features = node_features.to(
            self.device
        )  # [batch_size, num_services, time_steps, features]
        adj_matrix = adj_matrix.to(self.device)  # [num_services, num_services]
        anomaly_labels = anomaly_labels.to(self.device)  # [batch_size]
        root_cause_labels = root_cause_labels.to(self.device)  # [batch_size]

        # Forward pass
        outputs = self.model(node_features, adj_matrix)
        root_cause_probs = outputs["root_cause_probs"]  # [batch_size, num_services]
        anomaly_score = outputs["anomaly_score"]  # [batch_size, 2]

        # Compute loss
        anomaly_loss = self.anomaly_criterion(anomaly_score, anomaly_labels)

        # Only compute root cause loss for abnormal samples
        anomaly_indices = (anomaly_labels == 1).nonzero(as_tuple=True)[0]
        if len(anomaly_indices) > 0:
            root_cause_loss = self.root_cause_criterion(
                root_cause_probs[anomaly_indices], root_cause_labels[anomaly_indices]
            )
            loss = anomaly_loss + root_cause_loss
        else:
            loss = anomaly_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        anomaly_labels: torch.Tensor,
        root_cause_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()

        # Move to device
        node_features = node_features.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        anomaly_labels = anomaly_labels.to(self.device)
        root_cause_labels = root_cause_labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(node_features, adj_matrix)
            root_cause_probs = outputs["root_cause_probs"]
            anomaly_score = outputs["anomaly_score"]

            # Compute accuracy
            anomaly_preds = anomaly_score.argmax(dim=1)
            anomaly_acc = (anomaly_preds == anomaly_labels).float().mean().item()

            # Only compute root cause accuracy for abnormal samples
            anomaly_indices = (anomaly_labels == 1).nonzero(as_tuple=True)[0]
            if len(anomaly_indices) > 0:
                root_cause_preds = root_cause_probs[anomaly_indices].argmax(dim=1)
                root_cause_acc = (
                    (root_cause_preds == root_cause_labels[anomaly_indices])
                    .float()
                    .mean()
                    .item()
                )
            else:
                root_cause_acc = 0.0

        return {"anomaly_accuracy": anomaly_acc, "root_cause_accuracy": root_cause_acc}

    def predict(
        self, node_features: torch.Tensor, adj_matrix: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Predict root cause and anomaly"""
        self.model.eval()

        # Move to device
        node_features = node_features.to(self.device)
        adj_matrix = adj_matrix.to(self.device)

        with torch.no_grad():
            outputs = self.model(node_features, adj_matrix)
            root_cause_probs = (
                outputs["root_cause_probs"].cpu().numpy()
            )  # [batch_size, num_services]
            anomaly_probs = (
                torch.softmax(outputs["anomaly_score"], dim=1).cpu().numpy()
            )  # [batch_size, 2]
            node_features = (
                outputs["node_features"].cpu().numpy()
            )  # [batch_size, num_services, hidden_dim]

        return {
            "root_cause_probs": root_cause_probs,
            "anomaly_probs": anomaly_probs,
            "node_features": node_features,
        }
