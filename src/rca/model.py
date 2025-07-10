import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GATConv
import pandas as pd
from typing import List, Dict, Tuple, Optional

class TemporalEncoder(nn.Module):
    """时序特征编码器：使用Transformer或GRU提取时序依赖"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, 
                 use_transformer: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_transformer = use_transformer
        self.hidden_dim = hidden_dim
        
        if use_transformer:
            # Transformer编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim, 
                nhead=2, 
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projection = nn.Linear(input_dim, hidden_dim)
        else:
            # GRU编码器
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: x [batch_size, num_nodes, time_steps, features]
        输出: [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, time_steps, features = x.shape
        
        # 合并批次和节点维度
        x_reshaped = x.reshape(batch_size * num_nodes, time_steps, features)
        
        if self.use_transformer:
            # Transformer处理
            encoded = self.encoder(x_reshaped)  # [batch_size*num_nodes, time_steps, input_dim]
            # 使用最后一个时间步的输出作为表示
            out = encoded[:, -1, :]  # [batch_size*num_nodes, input_dim]
        else:
            # GRU处理
            out, _ = self.encoder(x_reshaped)  # [batch_size*num_nodes, time_steps, hidden_dim*2]
            # 使用最后一个时间步的输出作为表示
            out = out[:, -1, :]  # [batch_size*num_nodes, hidden_dim*2]
        
        # 投影到统一维度
        projected = self.projection(out)  # [batch_size*num_nodes, hidden_dim]
        
        # 分离批次和节点维度
        return projected.reshape(batch_size, num_nodes, self.hidden_dim)


class FaultPropagationGAT(nn.Module):
    """基于GAT的故障传播图神经网络（修复维度匹配问题）"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads  # 保存多头数，用于后续维度计算
        self.gat1 = GATConv(
            input_dim, hidden_dim, num_heads=num_heads,
            feat_drop=dropout, attn_drop=dropout, residual=True,
            allow_zero_in_degree=True
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads, hidden_dim, num_heads=1,
            feat_drop=dropout, attn_drop=dropout, residual=True,
            allow_zero_in_degree=True
        )
        self.prop_coef = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)  # 增加dropout提升泛化性

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor) -> torch.Tensor:
        edge_weights = g.edata['weight']  # 边权重形状: [num_edges]

        # # 手动实现带边权重的消息传递（替换默认GAT的消息函数）
        # def message_func(edges):
        #     # edges.src['h']: [num_edges, hidden_dim*num_heads]
        #     # edges.data['weight']: [num_edges] → 扩展为 [num_edges, 1]
        #     return {'msg': edges.src['h'] * edges.data['weight'].unsqueeze(1)}

        # # 修正聚合函数：明确在邻居维度聚合
        # def reduce_func(nodes):
        #     # nodes.mailbox['msg']: [num_nodes, num_neighbors, hidden_dim*num_heads]
        #     # 仅在邻居维度（dim=1）取平均，输出形状: [num_nodes, hidden_dim*num_heads]
        #     return {'h': torch.mean(nodes.mailbox['msg'], dim=1)}
        
        # 第一层GAT：多头特征提取
        h1 = self.gat1(g, node_features)  # 形状: [num_nodes, num_heads, hidden_dim]
        
        # 合并多头维度 → [num_nodes, hidden_dim*num_heads]
        h1 = h1.flatten(1)  # 关键：消除num_heads维度

        # 消息传递准备
        g = g.local_var()
        g.ndata['h'] = h1  # 节点特征形状: [num_nodes, 512]
        g.edata['weight'] = torch.sigmoid(self.prop_coef * edge_weights)  # 权重归一化
        
        # 执行消息传递
        # g.update_all(message_func, reduce_func)
        g.update_all(
        dgl.function.u_mul_e('h', 'weight', 'msg'),  # 源节点特征×边权重
        dgl.function.mean('msg', 'h')  # 邻居聚合（平均）
        )
        h1_weighted = g.ndata['h']  # 形状: [num_nodes, 512]

        # 第二层GAT：压缩特征维度
        h2 = self.gat2(g, h1_weighted)  # 形状: [num_nodes, 1, hidden_dim]
        h2 = h2.squeeze(1)  # 消除单头维度 → [num_nodes, hidden_dim]

        # # 3. 结合原始特征和传播后的特征（残差连接，保留自身信息）
        # final_features = node_features + h2  # [num_nodes, hidden_dim]
        return h2


class RCAModel(nn.Module):
    """根因分析主模型：整合时序特征和图特征"""
    
    def __init__(self, num_services: int, feature_dim: int, hidden_dim: int = 128, 
                 use_transformer: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_services = num_services
        self.hidden_dim = hidden_dim
        
        # 1. 时序特征编码器
        self.temporal_encoder = TemporalEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_transformer=use_transformer,
            dropout=dropout
        )
        
        # 2. 图特征编码器
        self.graph_encoder = FaultPropagationGAT(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # 3. 根因预测器
        self.root_cause_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.rca=nn.Sequential(
            nn.Linear(num_services, num_services*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_services*2, num_services)
        )

        
    def forward(self, features: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        输入:
            features: [window_size, num_nodes, feature_dim] 时序特征
            graph: DGL图对象
        输出:
            root_cause_scores: [num_nodes] 每个节点的根因分数
        """
        # 1. 时序特征提取 - 需要转换维度
        # 从 [batch_size, window_size, num_nodes, feature_dim] 转为 [batch_size, num_nodes, window_size, feature_dim]
        features_transposed = features.permute(0, 2, 1, 3)  # [batch_size, num_nodes, window_size, feature_dim]

        # 对每个节点提取时序特征
        temporal_features = self.temporal_encoder(features_transposed)  # [batch_size, num_nodes, hidden_dim]
        

        # 2. 图特征提取
        # temporal_features形状：[batch_size, num_nodes, hidden_dim]
        # 将batch和node维度合并：[batch_size*num_nodes, hidden_dim]
        batch_size, num_nodes = temporal_features.shape[0], temporal_features.shape[1]
        flat_features = temporal_features.reshape(-1, self.hidden_dim)  # [B*N, D]
        
        # 直接对批处理图计算（无需unbatch）
        flat_graph_features = self.graph_encoder(graph, flat_features)  # [B*N, D]
        
        # 恢复维度：[batch_size, num_nodes, hidden_dim]
        graph_features = flat_graph_features.reshape(batch_size, num_nodes, self.hidden_dim)
        
        # 3. 根因预测
        # 调整维度以适应预测器
        graph_features_flat = graph_features.reshape(-1, self.hidden_dim)  # [batch_size*num_nodes, hidden_dim]
        root_cause_scores_flat = self.root_cause_predictor(graph_features_flat)  # [batch_size*num_nodes, 1]
        
        # 恢复批次和节点维度
        root_cause_scores = root_cause_scores_flat.reshape(batch_size, self.num_services)  # [batch_size, num_services]
        
        root_cause_scores=self.rca(root_cause_scores)  # [batch_size, num_services]
        return root_cause_scores
    

if __name__ == "__main__":
    # 模拟输入特征和图
    num_services = 5
    feature_dim = 12
    window_size = 7
    hidden_dim = 128

    # 创建模型实例
    model = RCAModel(num_services=num_services, feature_dim=feature_dim, hidden_dim=hidden_dim)

    # 创建模拟输入特征
    features = torch.randn(window_size, num_services, feature_dim)  # [window_size, num_nodes, feature_dim]

    # 创建模拟图
    graph = dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])))
    graph.edata['weight'] = torch.rand(graph.num_edges())  # 边权重

    # 测试模型
    root_cause_scores = model(features, graph)
    print(root_cause_scores)
    print(root_cause_scores.shape)  # 应该是 [num_nodes]


