import torch
import torch.nn as nn
import torch.optim as optim
import os
import typer
from tqdm import tqdm
from .training import test_model
import dgl
from .training import bidirectional_pairwise_ranking_loss, cross_period_discrepancy_loss

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 3. Evaluate initial performance of the pretrained model
    best_val_loss = float("inf")
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for features, graphs,_,_, labels in val_loader:
            features = features.to(device)
            graphs = graphs.to(device)
            labels = labels.to(device)
            outputs,_ = model(features, graphs)
            val_loss += criterion(outputs, labels).item() * features.size(0)
        best_val_loss = val_loss / len(val_loader.dataset)
    typer.echo(f"Initial validation loss (pretrained model): {best_val_loss:.6f}")

    # 4. Incremental training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training process
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Incremental Epoch {epoch + 1}/{num_epochs}")

            for features, graphs, normal_features, normal_graphs, labels in tepoch:
                # Move data to device
                features = features.to(device)
                graphs = graphs.to(device)
                normal_features = normal_features.to(device)
                normal_graphs = normal_graphs.to(device)
                # labels = labels.to(device)
                labels = torch.argmax(labels, dim=1).to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs, anomaly_gat_emb = model(features, graphs)
                normal_outputs, normal_gat_emb = model(normal_features, normal_graphs)

                # Compute loss
                loss = criterion(outputs, labels)

                # # add pairwise-ranking loss
                batched_graphs = dgl.unbatch(graphs)
                pairwise_losses = []
                for i in range(len(batched_graphs)):
                    g = batched_graphs[i]
                    root = labels[i].item()
                    logits = outputs[i]
                    loss_r = bidirectional_pairwise_ranking_loss(logits, root, g, margin=0.1, max_depth=3)
                    pairwise_losses.append(loss_r)
                pairwise_loss = torch.stack(pairwise_losses).mean()

                # add cross-period discrepancy loss
                cpd_loss = cross_period_discrepancy_loss(
                    anomaly_emb=anomaly_gat_emb,
                    normal_emb=normal_gat_emb,
                    root_idx=labels,
                    margin=0.1,
                    distance_type="cosine"  # 先试cosine，再根据效果调整
                )

                loss = loss + 0.5 * pairwise_loss + 0.15 * cpd_loss
                train_loss += loss.item() * features.size(0)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Update progress bar
                tepoch.set_postfix(loss=loss.item())

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, graphs, normal_features, normal_graphs, labels in val_loader:
                features = features.to(device)
                graphs = graphs.to(device)
                labels = labels.to(device)
                outputs,_ = model(features, graphs)
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

        # Evaluate on validation set every 1 epochs
        if (epoch + 1) % 1 == 0:
            typer.echo("-" * 50)
            test_results, _ = test_model(model, val_loader, device)
            for metric, value in test_results.items():
                typer.echo(f"{metric}: {value:.4f}")

    return model
