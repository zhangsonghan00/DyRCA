import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import dgl
import pickle
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from model import RCAModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='dgl')



class RCADataset(Dataset):
    """根因分析数据集类，处理图数据和时序特征"""
    def __init__(self, samples_path, labels_path):
        # 加载数据
        with open(samples_path, 'rb') as f:
            self.samples = pickle.load(f)  # list of (features, graph) tuples
        with open(labels_path, 'rb') as f:
            self.labels = pickle.load(f)   # list of tensors (length = num_nodes)
        
        # 检查数据长度是否匹配
        assert len(self.samples) == len(self.labels), "样本和标签数量不匹配"
        self.num_nodes = self.labels[0].shape[0]  # 获取节点数量

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 获取时序特征和图结构
        features, g = self.samples[idx]
        features = torch.tensor(features, dtype=torch.float32)  # [time_window, nodes, metrics]
        label = self.labels[idx].float()  # 转换为float类型
        
        return features, g, label


def collate_fn(batch):
    """自定义批处理函数，解决DGL图的批处理问题"""
    features_list, graphs_list, labels_list = zip(*batch)
    
    # 堆叠时序特征 [batch_size, time_window, nodes, metrics]
    features = torch.stack(features_list, dim=0)
    
    # 使用dgl.batch合并图
    batched_graph = dgl.batch(graphs_list)
    
    # 堆叠标签 [batch_size, num_nodes]
    labels = torch.stack(labels_list, dim=0)
    
    return features, batched_graph, labels


def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3):
    """模型训练函数"""
    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 自带sigmoid，适合one-hot标签
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 记录最佳验证损失
    best_val_loss = float('inf')
    best_model_path = 'best_rca_model.pth'
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # 训练过程
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for features, graphs, labels in tepoch:
                # 数据移至设备
                features = features.to(device)
                graphs = graphs.to(device)
                labels = labels.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(features, graphs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                train_loss += loss.item() * features.size(0)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 更新进度条
                tepoch.set_postfix(loss=loss.item())
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        
        # 验证过程
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
        
        # 学习率调整
        scheduler.step()
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (Val Loss: {best_val_loss:.6f})")
        
        # 每3个epoch打印一次test详细信息
        if (epoch + 1) % 2 == 0:
            print("-" * 50)
            test_results=test_model(model, val_loader, device)
            for metric, value in test_results.items():
                print(f"{metric}: {value:.4f}")
    
    return model


def test_model(model, test_loader, device, top_k_list=[1, 3, 5]):
    """模型测试函数，计算RCA任务的Top-K准确率"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, graphs, labels in test_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = labels.to(device)
            
            # 模型预测（使用sigmoid转换为概率）
            outputs = torch.sigmoid(model(features, graphs))
            
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并所有预测和标签
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算Top-K准确率
    results = {}
    for k in top_k_list:
        # 确保k不超过节点数量
        k = min(k, all_preds.shape[1])
        correct = 0
        
        for pred, label in zip(all_preds, all_labels):
            # 获取真实根因位置
            true_root = np.where(label == 1)[0]
            if len(true_root) == 0:
                continue  # 跳过无标签样本
            
            # 获取预测概率最高的k个节点
            top_k_pred = np.argsort(pred)[-k:][::-1]
            
            # 检查真实根因是否在Top-K中
            if np.any(np.isin(true_root, top_k_pred)):
                correct += 1
        
        accuracy = correct / len(all_preds)
        results[f"Top-{k} Accuracy"] = accuracy
        print(f"Top-{k} Accuracy: {accuracy:.4f}")
    
    return results


def main():
    """主函数：数据加载、模型初始化、训练和测试"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据路径
    data_path = "data/RCABENCH/samples/"
    samples_path = os.path.join(data_path, "abnormal_samples.pkl")
    labels_path = os.path.join(data_path, "abnormal_labels.pkl")

    # 加载数据集
    full_dataset = RCADataset(samples_path, labels_path)
    
    num_nodes = full_dataset.num_nodes
    feature_dim = full_dataset.samples[0][0].shape[-1]  # 从样本中获取特征维度
    
    # 划分训练集、验证集和测试集（6:2:2）
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    # 初始化模型
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=32,
        use_transformer=True
    ).to(device)
    print(model)
    
    # 训练模型
    print("Starting training...")
    train_model(model, train_loader, val_loader, device, num_epochs=20)
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load("best_rca_model.pth"))
    print("\nStarting testing...")
    test_results = test_model(model, test_loader, device)
    
    # 打印最终测试结果
    print("\nFinal Test Results:")
    for metric, value in test_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()