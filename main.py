import torch
import typer
import os
from typing import Optional
from src.rca.model import RCAModel
from src.rca.dataset import RCADataset
from src.rca.training import (
    train_model,
    test_model,
    predict_single_case,
    load_model_and_metadata,
    prepare_data_loaders,
)
from src.rca.dataset import collate_fn
from src.rca.incremental_training import incremental_train_model
from src.rca.preprocess import run_preprocessing, run_single_pack_preprocessing
from torch.utils.data import DataLoader
import pickle
import warnings
from pathlib import Path

# 过滤特定的 PyTorch 警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = typer.Typer(help="PaDi-RCA: Root Cause Analysis Command Line Tool")


@app.command()
def preprocess(
    data_root: str = typer.Option(
        "/mnt/jfs/rcabench-platform-v2/data",
        help="Root directory containing case data",
    ),
    output_dir: str = typer.Option(
        "data/RCABENCH", help="Output dir/ectory for processed data"
    ),
    max_cases: Optional[int] = typer.Option(
        None, help="Maximum number of cases to process"
    ),
    dataset_type: str = typer.Option(
        "RCABENCH_r1", help="Dataset type: RCABENCH_r1, RCABENCH_filtered, or other"
    ),
):
    """Run data preprocessing"""
    typer.echo("Starting data preprocessing...")
    typer.echo(f"Data root: {data_root}")
    typer.echo(f"Output directory: {output_dir}")

    if max_cases:
        typer.echo(f"Max cases: {max_cases}")

    try:
        result = run_preprocessing(
            data_root=data_root,
            output_dir=output_dir,
            max_cases=max_cases,
            ds=dataset_type,
        )

        typer.echo("\nPreprocessing completed successfully!")
        typer.echo(
            f"Found {result['num_services']} services and {result['num_metrics']} metrics"
        )

    except Exception as e:
        typer.echo(f"Error during preprocessing: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    train_samples_path: str = typer.Option(
        "data/RCABENCH/samples/train_samples.pkl", help="Path to samples pickle file"
    ),
    train_labels_path: str = typer.Option(
        "data/RCABENCH/samples/train_labels.pkl", help="Path to labels pickle file"
    ),
    test_samples_path: str = typer.Option(
        "data/RCABENCH/samples/test_samples.pkl", help="Path to samples pickle file"
    ),
    test_labels_path: str = typer.Option(
        "data/RCABENCH/samples/test_labels.pkl", help="Path to labels pickle file"
    ),
    seed: int = typer.Option(42, help="Random seed for data splitting"),
    output_model: str = typer.Option("best_rca_model.pth", help="Output model path"),
    epochs: int = typer.Option(20, help="Number of training epochs"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    hidden_dim: int = typer.Option(32, help="Hidden dimension"),
    use_transformer: bool = typer.Option(True, help="Use transformer architecture"),
    device: Optional[str] = typer.Option(None, help="Device to use (cuda/cpu)"),
):
    """Train RCA model"""
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)
    typer.echo(f"Using device: {device_obj}")

    # Check if data files exist
    if not os.path.exists(train_samples_path):
        typer.echo(f"Error: Samples file not found: {train_samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(train_labels_path):
        typer.echo(f"Error: Labels file not found: {train_labels_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(test_samples_path):
        typer.echo(f"Error: Samples file not found: {test_samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(test_labels_path):
        typer.echo(f"Error: Labels file not found: {test_labels_path}", err=True)
        raise typer.Exit(1)
    
    # Load dataset
    typer.echo("Loading dataset...")
        # Load data
    with open(train_samples_path, "rb") as f:
        train_samples = pickle.load(f)  # list of (features, graph) tuples
    with open(train_labels_path, "rb") as f:
        train_labels = pickle.load(f)  # list of tensors (length = num_nodes)
    with open(test_samples_path, "rb") as f:
        test_samples = pickle.load(f)  # list of (features, graph) tuples
    with open(test_labels_path, "rb") as f:
        test_labels = pickle.load(f)  # list of tensors (length = num_nodes)

    ###让label type一致
    def filter_samples_by_labels(train_labels, train_samples, test_labels):
        # 1. 统计测试集中出现的非零类别
        test_labels_tensor = torch.stack(test_labels)
        test_classes = set(torch.where(test_labels_tensor.sum(dim=0) > 0)[0].tolist())
        
        # 2. 筛选训练集中的样本
        filtered_train_labels = []
        filtered_train_samples = []
        
        for label, sample in zip(train_labels, train_samples):
            # 获取当前标签的类别索引（假设是one-hot编码）
            class_idx = torch.argmax(label).item()
            
            # 保留测试集中存在的类别样本
            if class_idx in test_classes:
                filtered_train_labels.append(label)
                filtered_train_samples.append(sample)
        
        return filtered_train_labels, filtered_train_samples

    train_labels, train_samples = filter_samples_by_labels(
        train_labels, train_samples, test_labels
    )

    # 验证过滤后的训练集类别是否都在测试集中
    train_classes = set(torch.argmax(torch.stack(train_labels), dim=1).tolist())
    test_classes = set(torch.argmax(torch.stack(test_labels), dim=1).tolist())

    print("过滤后的训练集类别:", train_classes)
    print("测试集类别:", test_classes)
    print("训练集类别是否都在测试集中:", train_classes.issubset(test_classes))

    train_dataset = RCADataset(train_samples, train_labels)
    test_dataset = RCADataset(test_samples, test_labels)

    num_nodes = train_dataset.num_nodes
    feature_dim = train_dataset.samples[0][0].shape[-1]

    typer.echo(
        f"Train Dataset loaded: {len(train_dataset)} samples, {num_nodes} nodes, {feature_dim} features"
    )

    # Create data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        train_dataset, test_dataset, batch_size=batch_size, seed=seed
    )

    # Initialize model
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
    ).to(device_obj)

    typer.echo(
        f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Train model
    typer.echo("Starting training...")
    train_model(
        model, output_model, train_loader, test_loader, device_obj, num_epochs=epochs, lr=learning_rate
    )

    # Save model metadata
    metadata = {
        "num_nodes": num_nodes,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "use_transformer": use_transformer,
    }
    metadata_path = output_model.replace(".pth", "_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    typer.echo(f"Training completed! Model saved to {output_model}")
    typer.echo(f"Metadata saved to {metadata_path}")


@app.command()
def test(
    samples_path: str = typer.Option(
        "data/RCABENCH/samples/test_samples.pkl", help="Path to samples pickle file"
    ),
    labels_path: str = typer.Option(
        "data/RCABENCH/samples/test_labels.pkl", help="Path to labels pickle file"
    ),
    seed: int = typer.Option(42, help="Random seed for data splitting"),
    model_path: str = typer.Option("best_rca_model.pth", help="Path to trained model"),
    batch_size: int = typer.Option(32, help="Batch size for testing"),
    device: Optional[str] = typer.Option(None, help="Device to use (cuda/cpu)"),
):
    """Test RCA model"""
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)
    typer.echo(f"Using device: {device_obj}")

    # Check if files exist
    if not os.path.exists(model_path):
        typer.echo(f"Error: Model file not found: {model_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(samples_path):
        typer.echo(f"Error: Samples file not found: {samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(labels_path):
        typer.echo(f"Error: Labels file not found: {labels_path}", err=True)
        raise typer.Exit(1)

    # Load metadata
    metadata_path = model_path.replace(".pth", "_metadata.pkl")
    metadata = load_model_and_metadata(model_path, metadata_path)

    # Load dataset
    typer.echo("Loading dataset...")

    with open(samples_path, "rb") as f:
        samples = pickle.load(f)  # list of (features, graph) tuples
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)  # list of tensors (length = num_nodes)
    test_dataset = RCADataset(samples, labels)

    # Get model parameters from metadata or dataset
    num_nodes = metadata.get("num_nodes", test_dataset.num_nodes)
    feature_dim = metadata.get("feature_dim", test_dataset.samples[0][0].shape[-1])
    hidden_dim = metadata.get("hidden_dim", 32)
    use_transformer = metadata.get("use_transformer", True)

    # Create test data loader
    _, _, test_loader = prepare_data_loaders(
        test_dataset, test_dataset, batch_size=batch_size, seed=seed
    )

    # Initialize and load model
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
    ).to(device_obj)

    model.load_state_dict(torch.load(model_path, map_location=device_obj))
    typer.echo("Model loaded successfully")

    # Run test
    typer.echo("Starting testing...")
    test_results,_ = test_model(model, test_loader, device_obj)

    # Print test results
    typer.echo("\nTest Results:")
    typer.echo("-" * 40)
    for metric, value in test_results.items():
        typer.echo(f"{metric}: {value:.4f}")


@app.command()
def run_inference(
    datapack_path: Path,
    model_path: str = typer.Option("best_rca_model.pth", help="Path to trained model"),
    output_dir: str = typer.Option(
        "data/RCABENCH", help="Output directory for inference results"
    ),
    service_map_path: str = typer.Option(
        "data/RCABENCH/service_to_idx.pkl", help="Path to service_to_idx.pkl mapping file" 
    ),
    batch_size: int = typer.Option(32, help="Batch size for testing"),
):
    # Set device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device_obj = torch.device(device_str)
    typer.echo(f"Using device: {device_obj}")

    """Run inference on a single datapack"""
    preprocessed_data = run_single_pack_preprocessing(
        datapack_path=datapack_path,
        output_dir=output_dir,
        )
    samples=preprocessed_data["samples"]
    labels=preprocessed_data["sample_labels"]

    full_dataset = RCADataset(samples, labels)

    # Load metadata
    metadata_path = model_path.replace(".pth", "_metadata.pkl")
    metadata = load_model_and_metadata(model_path, metadata_path)

    # Get model parameters from metadata or dataset
    num_nodes = metadata.get("num_nodes", full_dataset.num_nodes)
    feature_dim = metadata.get("feature_dim", full_dataset.samples[0][0].shape[-1])
    hidden_dim = metadata.get("hidden_dim", 32)
    use_transformer = metadata.get("use_transformer", True)

    # Create test data loader
    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Initialize and load model
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
    ).to(device_obj)

    model.load_state_dict(torch.load(model_path, map_location=device_obj))
    typer.echo("Model loaded successfully")

    # Run test
    typer.echo("Starting inference...")
    test_results,top_k_results = test_model(model, test_loader, device_obj)

    top_k_result=top_k_results[0]

    with open(service_map_path, "rb") as f:
        service_to_idx = pickle.load(f)  # 原始映射: {service_name: node_id}
    idx_to_service = {v: k for k, v in service_to_idx.items()}  # 反转后: {node_id: service_name}

    # Get top-k service names
    top_k_service_names = [idx_to_service[idx] for idx in top_k_result]

    return top_k_service_names

@app.command()
def predict(
    sample_index: int = typer.Argument(..., help="Index of the sample to predict"),
    samples_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_samples.pkl", help="Path to samples pickle file"
    ),
    model_path: str = typer.Option("best_rca_model.pth", help="Path to trained model"),
    service_map_path: str = typer.Option(
        "data/RCABENCH/service_to_idx.pkl", help="Path to service_to_idx.pkl mapping file" 
    ),
    top_k: int = typer.Option(5, help="Number of top predictions to show"),
    device: Optional[str] = typer.Option(None, help="Device to use (cuda/cpu)"),
):
    """Run root cause analysis inference for a single sample"""
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)

    # Check if files exist
    if not os.path.exists(model_path):
        typer.echo(f"Error: Model file not found: {model_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(samples_path):
        typer.echo(f"Error: Samples file not found: {samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(service_map_path):
        typer.echo(f"Error: Service map file not found: {service_map_path}", err=True)
        raise typer.Exit(1)
    
    # 加载服务名映射并反转（得到 {node_id: service_name}）
    with open(service_map_path, "rb") as f:
        service_to_idx = pickle.load(f)  # 原始映射: {service_name: node_id}
    idx_to_service = {v: k for k, v in service_to_idx.items()}  # 反转后: {node_id: service_name}

    # Load metadata
    metadata_path = model_path.replace(".pth", "_metadata.pkl")
    metadata = load_model_and_metadata(model_path, metadata_path)

    # Load sample data
    typer.echo("Loading samples...")
    with open(samples_path, "rb") as f:
        samples = pickle.load(f)

    if sample_index >= len(samples) or sample_index < 0:
        typer.echo(
            f"Error: Sample index {sample_index} out of range (0-{len(samples) - 1})",
            err=True,
        )
        raise typer.Exit(1)

    # Get specified sample
    features, graph = samples[sample_index]
    features = torch.tensor(features, dtype=torch.float32)

    # Get model parameters from metadata
    num_nodes = metadata.get("num_nodes", graph.num_nodes())
    feature_dim = metadata.get("feature_dim", features.shape[-1])
    hidden_dim = metadata.get("hidden_dim", 32)
    use_transformer = metadata.get("use_transformer", True)

    # Initialize and load model
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
    ).to(device_obj)

    model.load_state_dict(torch.load(model_path, map_location=device_obj))
    typer.echo("Model loaded successfully")

    # Run prediction
    typer.echo(f"Predicting root cause for sample {sample_index}...")
    results = predict_single_case(model, features, graph, device_obj)
    # 5. 转换
    answers = []
    for result in results[:top_k]:  # 取 top_k 结果
        node_id = result["node_id"]
        service_name = idx_to_service[node_id]  # 通过 node_id 映射服务名
        answers.append({
            "level": "service",  # 固定为 "service"
            "name": service_name,  # 服务名（如 "mysql"、"rabbitmq"）
            "rank": result["rank"]  # 保持原排名
        })


    # Show Top-K results
    typer.echo(f"\nTop-{top_k} Root Cause Predictions:")
    typer.echo("-" * 50)
    typer.echo(f"{'Rank':<10} {'Node ID':<10} {'Node Name':<15}")
    typer.echo("-" * 50)

    for result in answers[:top_k]:
        typer.echo(
            f"{result['level']:<10} {result['rank']:<10} {result['name']:<10}"
        )

@app.command()
def incremental_train(
    samples_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_samples1.pkl",
        help="Path to new samples pickle file for incremental training"
    ),
    labels_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_labels1.pkl",
        help="Path to new labels pickle file for incremental training"
    ),
    pretrained_model_path: str = typer.Option(
        "best_rca_model.pth",
        help="Path to pretrained model checkpoint"
    ),
    output_model: str = typer.Option(
        "best_rca_model.pth",
        help="Output path for the incrementally trained model (will overwrite if exists)"
    ),
    seed: int = typer.Option(42, help="Random seed for data splitting"),
    epochs: int = typer.Option(20, help="Number of incremental training epochs"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    learning_rate: float = typer.Option(1e-3, 
        help="Learning rate for incremental training (usually smaller than initial training)"),
    hidden_dim: int = typer.Option(32, help="Hidden dimension (must match pretrained model)"),
    use_transformer: bool = typer.Option(True, 
        help="Use transformer architecture (must match pretrained model)"),
    device: Optional[str] = typer.Option(None, help="Device to use (cuda/cpu)"),
):
    """Incrementally train the RCA model with new data"""
    # Set device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_obj = torch.device(device_str)
    typer.echo(f"Using device: {device_obj}")

    # Check files
    if not os.path.exists(samples_path):
        typer.echo(f"Error: Samples file not found: {samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(labels_path):
        typer.echo(f"Error: Labels file not found: {labels_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(pretrained_model_path):
        typer.echo(f"Error: Pretrained model not found: {pretrained_model_path}", err=True)
        raise typer.Exit(1)

    # Load dataset
    typer.echo("Loading new dataset for incremental training...")
    with open(samples_path, "rb") as f:
        samples = pickle.load(f)  # list of (features, graph) tuples
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)  # list of tensors (length = num_nodes)
    full_dataset = RCADataset(samples, labels)

    # Get model parameters
    num_nodes = full_dataset.num_nodes
    feature_dim = full_dataset.samples[0][0].shape[-1]
    typer.echo(
        f"Dataset loaded: {len(full_dataset)} samples, {num_nodes} nodes, {feature_dim} features"
    )

    # Create data loaders
    train_loader, val_loader, _ = prepare_data_loaders(
        full_dataset, batch_size=batch_size, seed=seed
    )

    # Initialize model (must match pretrained model architecture)
    model = RCAModel(
        num_services=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        use_transformer=use_transformer,
    ).to(device_obj)

    # Perform incremental training
    typer.echo("Starting incremental training...")
    model = incremental_train_model(
        model,
        train_loader,
        val_loader,
        device_obj,
        num_epochs=epochs,
        lr=learning_rate,
        pretrained_model_path=pretrained_model_path,
        output_model_path=output_model,
    )

    typer.echo(f"Incremental training completed! Model saved to {output_model}")


if __name__ == "__main__":
    app()
