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
from src.rca.preprocess import run_preprocessing
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="dgl")

app = typer.Typer(help="PaDi-RCA: Root Cause Analysis Command Line Tool")


@app.command()
def preprocess(
    data_root: str = typer.Option(
        "/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered",
        help="Root directory containing case data",
    ),
    output_dir: str = typer.Option(
        "data/RCABENCH", help="Output directory for processed data"
    ),
    max_cases: Optional[int] = typer.Option(
        None, help="Maximum number of cases to process"
    ),
    dataset_type: str = typer.Option(
        "rcabench_filtered", help="Dataset type: rcabench, rcabench_filtered, or other"
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
        typer.echo(f"Processed {result['num_samples']} samples")
        typer.echo(
            f"Found {result['num_services']} services and {result['num_metrics']} metrics"
        )
        typer.echo(f"Samples saved to: {result['samples_path']}")
        typer.echo(f"Labels saved to: {result['labels_path']}")

    except Exception as e:
        typer.echo(f"Error during preprocessing: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def train(
    samples_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_samples.pkl", help="Path to samples pickle file"
    ),
    labels_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_labels.pkl", help="Path to labels pickle file"
    ),
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
    if not os.path.exists(samples_path):
        typer.echo(f"Error: Samples file not found: {samples_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(labels_path):
        typer.echo(f"Error: Labels file not found: {labels_path}", err=True)
        raise typer.Exit(1)

    # Load dataset
    typer.echo("Loading dataset...")
    full_dataset = RCADataset(samples_path, labels_path)

    num_nodes = full_dataset.num_nodes
    feature_dim = full_dataset.samples[0][0].shape[-1]

    typer.echo(
        f"Dataset loaded: {len(full_dataset)} samples, {num_nodes} nodes, {feature_dim} features"
    )

    # Create data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        full_dataset, batch_size=batch_size
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
        model, train_loader, val_loader, device_obj, num_epochs=epochs, lr=learning_rate
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
        "data/RCABENCH/samples/abnormal_samples.pkl", help="Path to samples pickle file"
    ),
    labels_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_labels.pkl", help="Path to labels pickle file"
    ),
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
    full_dataset = RCADataset(samples_path, labels_path)

    # Get model parameters from metadata or dataset
    num_nodes = metadata.get("num_nodes", full_dataset.num_nodes)
    feature_dim = metadata.get("feature_dim", full_dataset.samples[0][0].shape[-1])
    hidden_dim = metadata.get("hidden_dim", 32)
    use_transformer = metadata.get("use_transformer", True)

    # Create test data loader (use all data)
    from torch.utils.data import DataLoader
    from src.rca.dataset import collate_fn

    test_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    typer.echo("Starting testing...")
    test_results = test_model(model, test_loader, device_obj)

    # Print test results
    typer.echo("\nTest Results:")
    typer.echo("-" * 40)
    for metric, value in test_results.items():
        typer.echo(f"{metric}: {value:.4f}")


@app.command()
def predict(
    sample_index: int = typer.Argument(..., help="Index of the sample to predict"),
    samples_path: str = typer.Option(
        "data/RCABENCH/samples/abnormal_samples.pkl", help="Path to samples pickle file"
    ),
    model_path: str = typer.Option("best_rca_model.pth", help="Path to trained model"),
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

    # Show Top-K results
    typer.echo(f"\nTop-{top_k} Root Cause Predictions:")
    typer.echo("-" * 50)
    typer.echo(f"{'Rank':<6} {'Node ID':<10} {'Node Name':<15} {'Probability':<12}")
    typer.echo("-" * 50)

    for result in results[:top_k]:
        typer.echo(
            f"{result['rank']:<6} {result['node_id']:<10} {result['node_name']:<15} {result['probability']:<12.4f}"
        )


if __name__ == "__main__":
    app()
