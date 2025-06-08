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
        epoch_num: int
) -> tuple[float, dict]:

    model.train()

    total_loss = 0.0

    metrics_computer.reset()

    for batch_idx, (pixel_data_batch, labels_batch) in enumerate(dataloader):

        pixel_data_batch = pixel_data_batch.to(device)
        labels_batch = labels_batch.to(device)


        optimizer.zero_grad()

        outputs = model(pixel_data_batch)

        loss = criterion(outputs, labels_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        metrics_computer.update(outputs, labels_batch)

    avg_epoch_loss = total_loss / len(dataloader)

    epoch_metrics = metrics_computer.compute()

    print(f"Epoch {epoch_num} Train: Avg. Loss: {avg_epoch_loss:.4f}")
    for metric_name, metric_value in epoch_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return avg_epoch_loss, epoch_metrics


def evaluate_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        metrics_computer: MetricCollection,
        epoch_num: int
) -> tuple[float, dict, torch.Tensor, torch.Tensor]:


    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_targets = []

    metrics_computer.reset()

    with torch.no_grad():

        for batch_idx, (pixel_data_batch, mask_batch, labels_batch) in enumerate(dataloader):

            pixel_data_batch = pixel_data_batch.to(device)
            mask_batch = mask_batch.to(device)
            labels_batch = labels_batch.to(device)


            outputs = model(pixel_data_batch, mask=mask_batch)


            loss = criterion(outputs, labels_batch)


            total_loss += loss.item()


            preds = torch.argmax(outputs, dim=1)


            all_predictions.append(preds.cpu())
            all_targets.append(labels_batch.cpu())


            metrics_computer.update(outputs, labels_batch)


    avg_loss = total_loss / len(dataloader)


    metrics = metrics_computer.compute()


    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)


    print(f"Epoch {epoch_num} Val: Avg. Loss: {avg_loss:.4f}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    return avg_loss, metrics, all_predictions_tensor, all_targets_tensor