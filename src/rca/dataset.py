import torch
from torch.utils.data import Dataset
import pickle
import dgl


class RCADataset(Dataset):
    """Root Cause Analysis dataset class, handling graph data and temporal features"""

    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        # Check if data lengths match
        assert len(self.samples) == len(self.labels), (
            "Number of samples and labels do not match"
        )
        self.num_nodes = self.labels[0].shape[0]  # Get number of nodes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get temporal features and graph structure
        features, g = self.samples[idx]
        features = torch.tensor(
            features, dtype=torch.float32
        )  # [time_window, nodes, metrics]
        label = self.labels[idx].float()  # Convert to float type

        return features, g, label


def collate_fn(batch):
    """Custom batch function to handle DGL graph batching"""
    features_list, graphs_list, labels_list = zip(*batch)

    # Stack temporal features [batch_size, time_window, nodes, metrics]
    features = torch.stack(features_list, dim=0)

    # Merge graphs using dgl.batch
    batched_graph = dgl.batch(graphs_list)

    # Stack labels [batch_size, num_nodes]
    labels = torch.stack(labels_list, dim=0)

    return features, batched_graph, labels
