#!/usr/bin/env python
# coding: utf-8

"""
Train a GNN model for TSPDL.

This script trains a GNN model to predict edge properties for TSPDL problems.
"""

import argparse
import datetime
import json
import os
import pathlib
import uuid

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm.auto as tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from gnngls import TSPDLDataset, TSPDLEdgeModel


def train(model, data_loader, criterion, optimizer, device, draft_violation_weight=2.0):
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        data_loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        draft_violation_weight: Weight for draft violation loss

    Returns:
        epoch_loss: Average loss for the epoch
        path_loss: Average path optimization loss
        violation_loss: Average draft violation loss
    """
    model.train()

    epoch_loss = 0
    path_loss_sum = 0
    violation_loss_sum = 0

    for batch_i, batch in enumerate(data_loader):
        batch_graphs = batch[0].to(device)
        batch_labels = batch[1].to(device)

        # Forward pass
        edge_feats = batch_graphs.edata['features']
        node_feats = torch.cat([
            batch_graphs.ndata['coords'],
            batch_graphs.ndata['demand'],
            batch_graphs.ndata['draft_limit']
        ], dim=1)

        outputs = model(batch_graphs, edge_feats, node_feats).squeeze()

        # Get reachability mask
        reachability_mask = batch_graphs.edata['reachability_mask'].squeeze()

        # Calculate path optimization loss (for edges that are reachable)
        # Only consider reachable edges (mask=1) for path optimization
        reachable_indices = reachability_mask > 0.5
        if reachable_indices.sum() > 0:
            path_loss = criterion(
                outputs[reachable_indices],
                batch_labels[reachable_indices]
            )
        else:
            path_loss = torch.tensor(0.0, device=device)

        # Calculate draft violation loss
        # For unreachable edges (mask=0), predict low scores (close to 0)
        unreachable_indices = reachability_mask < 0.5
        if unreachable_indices.sum() > 0:
            # Use MSE loss to push predictions for unreachable edges towards 0
            violation_loss = torch.mean(outputs[unreachable_indices] ** 2)
        else:
            violation_loss = torch.tensor(0.0, device=device)

        # Combine losses with weighting
        loss = path_loss + draft_violation_weight * violation_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        path_loss_sum += path_loss.item()
        violation_loss_sum += violation_loss.item()

    epoch_loss /= (batch_i + 1)
    path_loss_avg = path_loss_sum / (batch_i + 1)
    violation_loss_avg = violation_loss_sum / (batch_i + 1)

    return epoch_loss, path_loss_avg, violation_loss_avg


def validate(model, data_loader, criterion, device, draft_violation_weight=2.0):
    """
    Validate the model.

    Args:
        model: Model to validate
        data_loader: Data loader
        criterion: Loss function
        device: Device to use
        draft_violation_weight: Weight for draft violation loss

    Returns:
        val_loss: Average validation loss
        path_loss: Average path optimization loss
        violation_loss: Average draft violation loss
    """
    model.eval()

    val_loss = 0
    path_loss_sum = 0
    violation_loss_sum = 0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            batch_graphs = batch[0].to(device)
            batch_labels = batch[1].to(device)

            # Forward pass
            edge_feats = batch_graphs.edata['features']
            node_feats = torch.cat([
                batch_graphs.ndata['coords'],
                batch_graphs.ndata['demand'],
                batch_graphs.ndata['draft_limit']
            ], dim=1)

            outputs = model(batch_graphs, edge_feats, node_feats).squeeze()

            # Get reachability mask
            reachability_mask = batch_graphs.edata['reachability_mask'].squeeze()

            # Calculate path optimization loss (for edges that are reachable)
            reachable_indices = reachability_mask > 0.5
            if reachable_indices.sum() > 0:
                path_loss = criterion(
                    outputs[reachable_indices],
                    batch_labels[reachable_indices]
                )
            else:
                path_loss = torch.tensor(0.0, device=device)

            # Calculate draft violation loss
            unreachable_indices = reachability_mask < 0.5
            if unreachable_indices.sum() > 0:
                violation_loss = torch.mean(outputs[unreachable_indices] ** 2)
            else:
                violation_loss = torch.tensor(0.0, device=device)

            # Combine losses
            loss = path_loss + draft_violation_weight * violation_loss

            val_loss += loss.item()
            path_loss_sum += path_loss.item()
            violation_loss_sum += violation_loss.item()

    val_loss /= (batch_i + 1)
    path_loss_avg = path_loss_sum / (batch_i + 1)
    violation_loss_avg = violation_loss_sum / (batch_i + 1)

    return val_loss, path_loss_avg, violation_loss_avg


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        save_path: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN model for TSPDL")
    parser.add_argument("data_dir", type=str, help="Directory containing the dataset")
    parser.add_argument("output_dir", type=str, help="Directory to save the model")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--draft_violation_weight", type=float, default=2.0,
                        help="Weight for draft violation loss")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_dataset = TSPDLDataset(os.path.join(args.data_dir, "train", "train.txt"))
    val_dataset = TSPDLDataset(os.path.join(args.data_dir, "val", "val.txt"))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: (dgl.batch([x[0] for x in batch]), torch.cat([x[1] for x in batch]))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: (dgl.batch([x[0] for x in batch]), torch.cat([x[1] for x in batch]))
    )

    # Create model
    model = TSPDLEdgeModel(
        in_dim=3,  # [distance, in_tour, draft_limit_respected]
        embed_dim=args.embed_dim,
        out_dim=1,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Create tensorboard writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"tspdl_gnn_{timestamp}_{uuid.uuid4().hex[:8]}"
    log_dir = os.path.join(args.output_dir, "logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Save parameters
    params = vars(args)
    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

    # Train model
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.n_epochs):
        # Train
        train_loss, train_path_loss, train_violation_loss = train(
            model, train_loader, criterion, optimizer, device, args.draft_violation_weight
        )

        # Validate
        val_loss, val_path_loss, val_violation_loss = validate(
            model, val_loader, criterion, device, args.draft_violation_weight
        )

        # Log
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/train_path", train_path_loss, epoch)
        writer.add_scalar("Loss/val_path", val_path_loss, epoch)
        writer.add_scalar("Loss/train_violation", train_violation_loss, epoch)
        writer.add_scalar("Loss/val_violation", val_violation_loss, epoch)

        print(f"Epoch {epoch+1}/{args.n_epochs} - "
              f"Train Loss: {train_loss:.4f} (Path: {train_path_loss:.4f}, Violation: {train_violation_loss:.4f}), "
              f"Val Loss: {val_loss:.4f} (Path: {val_path_loss:.4f}, Violation: {val_violation_loss:.4f})")

        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            os.path.join(args.output_dir, f"checkpoint_{epoch+1}.pt")
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss,
                os.path.join(args.output_dir, "best_model.pt")
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, args.n_epochs, train_loss, val_loss,
        os.path.join(args.output_dir, "final_model.pt")
    )

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}")
