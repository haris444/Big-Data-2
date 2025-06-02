"""
Training and Evaluation Engine for Deep Learning Models

This module provides core training and evaluation functions for PyTorch deep learning models,
specifically designed for classification tasks. The engine handles the complete training loop
including forward/backward passes, loss computation, metric tracking, and logging.

FUNCTIONS:

train_one_epoch:
   Trains the model for one epoch using the provided dataloader, criterion, and optimizer.
   Sets model to training mode, iterates through batches performing forward/backward passes,
   updates parameters, tracks loss and metrics. Returns average epoch loss and computed metrics.
   
   Parameters: model, dataloader, criterion, optimizer, device, metrics_computer, epoch_num
   Returns: tuple[float, dict] - (average_epoch_loss, epoch_metrics_dict)

evaluate_model:
   Evaluates the model on the provided dataloader using the criterion without gradient computation.
   Sets model to evaluation mode, processes all batches, collects predictions and targets,
   computes loss and metrics. Returns loss, metrics, and complete prediction/target tensors.
   
   Parameters: model, dataloader, criterion, device, metrics_computer, epoch_num  
   Returns: tuple[float, dict, torch.Tensor, torch.Tensor] - (avg_loss, metrics_dict, predictions_tensor, targets_tensor)

DATALOADER EXPECTATIONS:
Both functions expect dataloaders that yield (input_data, labels) tuples where input_data
is compatible with the model's forward method and labels are 1D tensors of class indices.

METRIC HANDLING:
Uses torchmetrics.MetricCollection for standardized metric computation. Automatically resets
metrics at epoch start, updates batch-by-batch, and computes final aggregated metrics.
"""


import torch
import torch.nn
from torchmetrics import MetricCollection



def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics_computer: MetricCollection,
    epoch_num: int  # For logging purposes
) -> tuple[float, dict]:
    """
    Trains the model for one epoch using the provided dataloader, criterion, and optimizer.
    Calculates loss and performance metrics.
    
    Returns:
        tuple[float, dict]: (average_epoch_loss, epoch_metrics_dict)
    """
    # Set model to training mode
    model.train()
    
    # Initialize total loss
    total_loss = 0.0
    
    # Reset metrics computer at start of epoch
    metrics_computer.reset()
    
    # Iterate through dataloader
    for batch_idx, (pixel_data_batch, labels_batch) in enumerate(dataloader):
        # Move data to device
        pixel_data_batch = pixel_data_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_data_batch)
        
        # Calculate loss
        loss = criterion(outputs, labels_batch)
        
        # Backward pass
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Add current batch loss to total
        total_loss += loss.item()
        
        # Update metrics computer
        metrics_computer.update(outputs, labels_batch)
    
    # Calculate average loss for the epoch
    avg_epoch_loss = total_loss / len(dataloader)
    
    # Compute epoch metrics
    epoch_metrics = metrics_computer.compute()
    
    # Print log statement
    print(f"Epoch {epoch_num} [Train]: Avg. Loss: {avg_epoch_loss:.4f}")
    for metric_name, metric_value in epoch_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return avg_epoch_loss, epoch_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    metrics_computer: MetricCollection,
    epoch_num: int  # For logging purposes, use -1 or specific string for final eval
) -> tuple[float, dict, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model on the provided dataloader using the criterion.
    Calculates loss, performance metrics, and collects all predictions and targets.
    
    Returns:
        tuple[float, dict, torch.Tensor, torch.Tensor]: (avg_loss, metrics_dict, predictions_tensor, targets_tensor)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize total loss
    total_loss = 0.0
    
    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []
    
    # Reset metrics computer
    metrics_computer.reset()
    
    # Disable gradient calculations
    with torch.no_grad():
        # Iterate through dataloader
        for batch_idx, (pixel_data_batch, labels_batch) in enumerate(dataloader):
            # Move data to device
            pixel_data_batch = pixel_data_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(pixel_data_batch)
            
            # Calculate loss
            loss = criterion(outputs, labels_batch)
            
            # Add current batch loss to total
            total_loss += loss.item()
            
            # Get predictions (class indices from logits)
            preds = torch.argmax(outputs, dim=1)
            
            # Append batch predictions and labels to lists
            all_predictions.append(preds.cpu())
            all_targets.append(labels_batch.cpu())
            
            # Update metrics computer
            metrics_computer.update(outputs, labels_batch)
    
    # Calculate average loss for the dataset
    avg_loss = total_loss / len(dataloader)
    
    # Compute overall metrics for the dataset
    metrics = metrics_computer.compute()
    
    # Concatenate all predictions and targets into single tensors
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    # Print log statement
    print(f"Epoch {epoch_num} [Val/Test]: Avg. Loss: {avg_loss:.4f}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    return avg_loss, metrics, all_predictions_tensor, all_targets_tensor