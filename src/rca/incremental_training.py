import torch
import torch.nn as nn
import torch.optim as optim
import os
import typer
from tqdm import tqdm
from .training import test_model

def incremental_train_model(
    model, 
    train_loader, 
    val_loader, 
    device, 
    num_epochs=20, 
    lr=1e-3,
    pretrained_model_path="best_rca_model.pth",
    output_model_path="best_rca_model.pth", ## using the same path or other path 
):
    """
    Incrementally train the model using a pretrained model as initialization.
    
    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the model (e.g., 'cuda' or 'cpu').
        num_epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        pretrained_model_path: Path to the pretrained model checkpoint.
        output_model_path: Path to save the incrementally trained model.
    """
    # 1. Load pretrained weights (if provided)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        typer.echo(f"Loaded pretrained model from {pretrained_model_path}")
    
    # 2. Initialize optimizer and scheduler (same as original training)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 3. Evaluate initial performance of the pretrained model
    best_val_loss = float("inf")
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for features, graphs, labels in val_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = labels.to(device)
            outputs = model(features, graphs)
            val_loss += criterion(outputs, labels).item() * features.size(0)
        best_val_loss = val_loss / len(val_loader.dataset)
    typer.echo(f"Initial validation loss (pretrained model): {best_val_loss:.6f}")

    # 4. Incremental training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Incremental Epoch {epoch + 1}/{num_epochs}")

            for features, graphs, labels in tepoch:
                features = features.to(device)
                graphs = graphs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(features, graphs)
                loss = criterion(outputs, labels)
                train_loss += loss.item() * features.size(0)

                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, graphs, labels in val_loader:
                features = features.to(device)
                graphs = graphs.to(device)
                labels = labels.to(device)
                outputs = model(features, graphs)
                val_loss += criterion(outputs, labels).item() * features.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()

        typer.echo(f"Epoch {epoch + 1}/{num_epochs}")
        typer.echo(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        typer.echo(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)
            typer.echo(f"Saved best incremental model to {output_model_path} (Val Loss: {best_val_loss:.6f})")
            
        # Evaluate on validation set every 2 epochs
        if (epoch + 1) % 2 == 0:
            typer.echo("-" * 50)
            test_results, _ = test_model(model, val_loader, device)
            for metric, value in test_results.items():
                typer.echo(f"{metric}: {value:.4f}")

    return model
