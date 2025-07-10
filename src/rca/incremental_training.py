import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GATConv
import pandas as pd
from typing import List, Dict, Tuple, Optional
from .model import RCAModel
from .train import RCATrainer


# 增量训练策略
class IncrementalTrainer:
    """RCA模型增量训练器"""

    def __init__(
        self,
        base_model: RCAModel,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.base_model = base_model
        self.device = device
        self.trainer = RCATrainer(base_model, device, learning_rate, weight_decay)

        # 存储旧模型参数（用于知识蒸馏）
        self.old_model = RCAModel(
            num_services=base_model.num_services,
            feature_dim=base_model.temporal_encoder.encoder.d_model
            if base_model.temporal_encoder.use_transformer
            else base_model.temporal_encoder.encoder.hidden_size,
            hidden_dim=base_model.graph_encoder.gat1.out_feats,
        ).to(device)
        self.old_model.load_state_dict(base_model.state_dict())
        self.old_model.eval()

        # 增量训练参数
        self.kd_weight = 0.5  # 知识蒸馏权重

    def train_incremental(
        self,
        new_node_features: torch.Tensor,
        new_adj_matrix: torch.Tensor,
        new_anomaly_labels: torch.Tensor,
        new_root_cause_labels: torch.Tensor,
        old_samples_ratio: float = 0.2,
    ) -> float:
        """执行增量训练步骤，结合新知识和旧知识"""
        self.base_model.train()

        # 1. 使用旧模型生成旧样本的预测（知识蒸馏）
        with torch.no_grad():
            old_outputs = self.old_model(
                new_node_features.to(self.device), new_adj_matrix.to(self.device)
            )
            old_root_cause_probs = old_outputs[
                "root_cause_probs"
            ]  # [batch_size, num_services]
            old_anomaly_probs = torch.softmax(
                old_outputs["anomaly_score"], dim=1
            )  # [batch_size, 2]

        # 2. 计算新数据上的损失
        new_outputs = self.base_model(
            new_node_features.to(self.device), new_adj_matrix.to(self.device)
        )
        new_root_cause_probs = new_outputs[
            "root_cause_probs"
        ]  # [batch_size, num_services]
        new_anomaly_probs = torch.softmax(
            new_outputs["anomaly_score"], dim=1
        )  # [batch_size, 2]

        # 知识蒸馏损失（旧模型与新模型输出的KL散度）
        kd_root_cause_loss = F.kl_div(
            torch.log_softmax(new_root_cause_probs, dim=1),
            torch.softmax(old_root_cause_probs, dim=1),
            reduction="batchmean",
        )

        kd_anomaly_loss = F.kl_div(
            torch.log_softmax(new_anomaly_probs, dim=1),
            torch.softmax(old_anomaly_probs, dim=1),
            reduction="batchmean",
        )

        kd_loss = kd_root_cause_loss + kd_anomaly_loss

        # 3. 计算监督损失（使用新标签）
        new_anomaly_labels = new_anomaly_labels.to(self.device)  # [batch_size]
        new_root_cause_labels = new_root_cause_labels.to(self.device)  # [batch_size]

        anomaly_loss = self.trainer.anomaly_criterion(
            new_outputs["anomaly_score"], new_anomaly_labels
        )

        # 只对异常样本计算根因损失
        anomaly_indices = (new_anomaly_labels == 1).nonzero(as_tuple=True)[0]
        if len(anomaly_indices) > 0:
            root_cause_loss = self.trainer.root_cause_criterion(
                new_root_cause_probs[anomaly_indices],
                new_root_cause_labels[anomaly_indices],
            )
            supervised_loss = anomaly_loss + root_cause_loss
        else:
            supervised_loss = anomaly_loss

        # 4. 总损失（监督损失 + 知识蒸馏损失）
        total_loss = (1 - self.kd_weight) * supervised_loss + self.kd_weight * kd_loss

        # 5. 反向传播
        self.trainer.optimizer.zero_grad()
        total_loss.backward()
        self.trainer.optimizer.step()

        return total_loss.item()

    def update_old_model(self):
        """更新旧模型为当前模型状态"""
        self.old_model.load_state_dict(self.base_model.state_dict())
        self.old_model.eval()

    # 示例：增量训练
    def incremental_train_demo(self):
        # 创建增量训练器
        incremental_trainer = IncrementalTrainer(self.model, self.device)

        # 假设我们有新数据
        new_batch_size = 16
        new_node_features = torch.randn(
            new_batch_size, num_services, window_size, feature_dim
        )
        new_anomaly_labels = torch.randint(0, 2, (new_batch_size,))
        new_root_cause_labels = torch.full((new_batch_size,), -1, dtype=torch.long)
        for i in range(new_batch_size):
            if new_anomaly_labels[i] == 1:
                new_root_cause_labels[i] = np.random.randint(0, num_services)

        # 使用相同的邻接矩阵（静态图）
        adj_matrix = torch.zeros(num_services, num_services)
        for i in range(num_services):
            for j in range(num_services):
                if i != j and np.random.random() < 0.3:
                    adj_matrix[i, j] = np.random.uniform(0.1, 1.0)

        # 增量训练
        incremental_epochs = 20
        for epoch in range(incremental_epochs):
            loss = incremental_trainer.train_incremental(
                new_node_features, adj_matrix, new_anomaly_labels, new_root_cause_labels
            )
            if (epoch + 1) % 5 == 0:
                print(
                    f"Incremental Epoch {epoch + 1}/{incremental_epochs}, Loss: {loss:.4f}"
                )

        # 更新旧模型
        incremental_trainer.update_old_model()
