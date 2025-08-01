import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GATConv


class TemporalEncoder(nn.Module):
    """Temporal feature encoder: Extracts temporal dependencies using Transformer or GRU"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        use_transformer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.hidden_dim = hidden_dim

        if use_transformer:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=2,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projection = nn.Linear(input_dim, hidden_dim)
        else:
            # GRU encoder
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [batch_size, num_nodes, time_steps, features]
        Output: [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, time_steps, features = x.shape

        # Merge batch and node dimensions
        x_reshaped = x.reshape(batch_size * num_nodes, time_steps, features)

        if self.use_transformer:
            # Transformer processing
            encoded = self.encoder(
                x_reshaped
            )  # [batch_size*num_nodes, time_steps, input_dim]
            # Use the output of the last time step as the representation
            out = encoded[:, -1, :]  # [batch_size*num_nodes, input_dim]
        else:
            # GRU processing
            out, _ = self.encoder(
                x_reshaped
            )  # [batch_size*num_nodes, time_steps, hidden_dim*2]
            # Use the output of the last time step as the representation
            out = out[:, -1, :]  # [batch_size*num_nodes, hidden_dim*2]

        # Project to unified dimension
        projected = self.projection(out)  # [batch_size*num_nodes, hidden_dim]

        # Separate batch and node dimensions
        return projected.reshape(batch_size, num_nodes, self.hidden_dim)


class FaultPropagationGAT(nn.Module):
    """Fault propagation graph neural network based on GAT (fixes dimension matching issues)"""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_heads: int = 2, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = (
            num_heads  # Save number of heads for later dimension calculation
        )
        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            allow_zero_in_degree=True,
        )
        self.gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=True,
            allow_zero_in_degree=True,
        )
        self.prop_coef = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)  # Add dropout to improve generalization

    def forward(self, g: dgl.DGLGraph, node_features: torch.Tensor) -> torch.Tensor:
        edge_weights = g.edata["weight"]  # Edge weights shape: [num_edges]

        # First GAT layer: multi-head feature extraction
        h1 = self.gat1(g, node_features)  # Shape: [num_nodes, num_heads, hidden_dim]

        # Merge multi-head dimension → [num_nodes, hidden_dim*num_heads]
        h1 = h1.flatten(1)  # Key: remove num_heads dimension

        # Prepare for message passing
        g = g.local_var()
        g.ndata["h"] = h1  # Node feature shape: [num_nodes, 512]
        g.edata["weight"] = torch.sigmoid(
            self.prop_coef * edge_weights
        )  # Normalize weights

        # Perform message passing
        # g.update_all(message_func, reduce_func)
        g.update_all(
            dgl.function.u_mul_e(
                "h", "weight", "msg"
            ),  # Source node feature × edge weight
            dgl.function.mean("msg", "h"),  # Neighbor aggregation (mean)
        )
        h1_weighted = g.ndata["h"]  # Shape: [num_nodes, 512]

        # Second GAT layer: compress feature dimension
        h2 = self.gat2(g, h1_weighted)  # Shape: [num_nodes, 1, hidden_dim]
        h2 = h2.squeeze(1)  # Remove single-head dimension → [num_nodes, hidden_dim]

        return h2


class RCAModel(nn.Module):
    """Root cause analysis main model: integrates temporal and graph features"""

    def __init__(
        self,
        num_services: int,
        feature_dim: int,
        hidden_dim: int = 128,
        use_transformer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_services = num_services
        self.hidden_dim = hidden_dim

        # 1. Temporal feature encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_transformer=use_transformer,
            dropout=dropout,
        )

        # 2. Graph feature encoder
        self.graph_encoder = FaultPropagationGAT(
            input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        # 3. Root cause predictor
        self.root_cause_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.rca = nn.Sequential(
            nn.Linear(num_services, num_services * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_services * 2, num_services),
        )

    def forward(self, features: torch.Tensor, graph: dgl.DGLGraph) -> torch.Tensor:
        """
        Input:
            features: [window_size, num_nodes, feature_dim] temporal features
            graph: DGL graph object
        Output:
            root_cause_scores: [num_nodes] root cause score for each node
        """
        # 1. Temporal feature extraction - need to transpose dimensions
        # From [batch_size, window_size, num_nodes, feature_dim] to [batch_size, num_nodes, window_size, feature_dim]
        features_transposed = features.permute(
            0, 2, 1, 3
        )  # [batch_size, num_nodes, window_size, feature_dim]

        # Extract temporal features for each node
        temporal_features = self.temporal_encoder(
            features_transposed
        )  # [batch_size, num_nodes, hidden_dim]

        # 2. Graph feature extraction
        # temporal_features shape: [batch_size, num_nodes, hidden_dim]
        # Merge batch and node dimensions: [batch_size*num_nodes, hidden_dim]
        batch_size, num_nodes = temporal_features.shape[0], temporal_features.shape[1]
        flat_features = temporal_features.reshape(-1, self.hidden_dim)  # [B*N, D]

        # Directly compute on batched graph (no need to unbatch)
        flat_graph_features = self.graph_encoder(graph, flat_features)  # [B*N, D]

        # Restore dimensions: [batch_size, num_nodes, hidden_dim]
        graph_features = flat_graph_features.reshape(
            batch_size, num_nodes, self.hidden_dim
        )

        # 3. Root cause prediction
        # Adjust dimensions for predictor
        graph_features_flat = graph_features.reshape(
            -1, self.hidden_dim
        )  # [batch_size*num_nodes, hidden_dim]
        root_cause_scores_flat = self.root_cause_predictor(
            graph_features_flat
        )  # [batch_size*num_nodes, 1]

        # Restore batch and node dimensions
        root_cause_scores = root_cause_scores_flat.reshape(
            batch_size, self.num_services
        )  # [batch_size, num_services]

        root_cause_scores = self.rca(root_cause_scores)  # [batch_size, num_services]
        return root_cause_scores


if __name__ == "__main__":
    # Simulate input features and graph
    num_services = 5
    feature_dim = 12
    window_size = 7
    hidden_dim = 128

    # Create model instance
    model = RCAModel(
        num_services=num_services, feature_dim=feature_dim, hidden_dim=hidden_dim
    )

    # Create simulated input features
    features = torch.randn(
        window_size, num_services, feature_dim
    )  # [window_size, num_nodes, feature_dim]

    # Create simulated graph
    graph = dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 4])))
    graph.edata["weight"] = torch.rand(graph.num_edges())  # Edge weights

    # Test model
    root_cause_scores = model(features, graph)
    print(root_cause_scores)
    print(root_cause_scores.shape)  # Should be [num_nodes]
