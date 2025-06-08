import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold
import torchmetrics
import json


import paradotea.data_utils as data_utils
import paradotea.model   as model
import paradotea.engine as engine



#constants

DATA_ROOT = "timematch_data/denmark/32VNH/2017"
META_DIR = os.path.join(DATA_ROOT, "meta")
PARCEL_DATA_DIR = os.path.join(DATA_ROOT, "data")
DATES_FILE = os.path.join(META_DIR, "dates.json")
LABELS_FILE = os.path.join(META_DIR, "labels.json")

#experiment parameters
S_NUM_SAMPLED_PIXELS = 96
NUM_CHANNELS = 10
NUM_TIME_STEPS = 52
MAX_PIXEL_VALUE = 10000

#training Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 3

#crosvalidation setup
NUM_FOLDS = 5
RANDOM_SEED = 42

#architecture
pse_config = {
    'mlp1_dims': [NUM_CHANNELS, 32, 64],
    'mlp2_dims': [128, 128]
}

transformer_config = {
    'd_model': 128,
    'n_heads': 16,
    'num_layers': 2,
    'dropout_rate': 0.1
}

classifier_config = {
    'mlp_dims': [128, 64, -1],
    'dropout_rate': 0.1
}

#to train on gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



#dates
with open(DATES_FILE, 'r') as f:
    date_data = json.load(f)
date_strings = [str(date) for date in date_data]
day_of_year_array = data_utils.get_day_of_year_array(date_strings)

#parcel ids
parcel_ids_on_disk = [f.replace('.zarr', '') for f in os.listdir(PARCEL_DATA_DIR) if f.endswith('.zarr')]
print(f"Found {len(parcel_ids_on_disk)} parcels")

#load and count labels
labels_df_analyzed = data_utils.load_and_analyze_labels(LABELS_FILE, parcel_ids_on_disk)


#filter < 200 and reindex
filtered_labels_df, label_mapping, NUM_CLASSES = data_utils.filter_and_reindex_labels(
    labels_df_analyzed, min_samples_per_category=200
)
print(f"Classes after filter: {NUM_CLASSES}")

#configuration based on numclasses
classifier_config['mlp_dims'][-1] = NUM_CLASSES
transformer_config['d_model'] = pse_config['mlp2_dims'][-1]
classifier_config['mlp_dims'][0] = transformer_config['d_model']

all_parcel_info = list(zip(filtered_labels_df['parcel_id'].values, filtered_labels_df['new_reindexed_label'].values))
print(f"Total pairs: {len(all_parcel_info)}")


#k fold

kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
print(f"Starting 5 fold cross-validation...")

fold_results = []


for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(all_parcel_info)):
    print(f"Fold {fold_idx + 1}/{NUM_FOLDS} ")

    current_train_info = [all_parcel_info[i] for i in train_indices]
    current_val_info = [all_parcel_info[i] for i in val_indices]


    print("Calculating normalization statistics...")
    rescaler = data_utils.RescalePixels(MAX_PIXEL_VALUE)
    norm_stats = data_utils.calculate_normalization_stats(
        parcel_info_list=current_train_info,
        data_root_path=PARCEL_DATA_DIR,
        num_channels=NUM_CHANNELS,
        rescale_transform=rescaler
    )

    train_dataset = data_utils.ParcelDataset(
        parcel_info_list=current_train_info,
        data_root_path=PARCEL_DATA_DIR,
        S_num_sampled_pixels=S_NUM_SAMPLED_PIXELS,
        is_train=True,
        num_channels=NUM_CHANNELS,
        num_time_steps=NUM_TIME_STEPS,
        rescale_transform=rescaler,
        normalization_stats=norm_stats
    )

    val_dataset = data_utils.ParcelDataset(
        parcel_info_list=current_val_info,
        data_root_path=PARCEL_DATA_DIR,
        S_num_sampled_pixels=S_NUM_SAMPLED_PIXELS,
        is_train=False,
        num_channels=NUM_CHANNELS,
        num_time_steps=NUM_TIME_STEPS,
        rescale_transform=rescaler,
        normalization_stats=norm_stats
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8, #padding adds a lot of ram ussage
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=data_utils.collate_fn_padd
    )


    current_model = model.ParcelTimeSeriesClassifier(
        pse_config=pse_config,
        transformer_config=transformer_config,
        classifier_config=classifier_config,
        day_of_year_sequence=day_of_year_array
    ).to(device)

    current_criterion = nn.CrossEntropyLoss()
    current_optimizer = torch.optim.Adam(current_model.parameters(), lr=LEARNING_RATE)

    #metrics
    print("Setting up validation metrics...")
    val_metrics_computer = torchmetrics.MetricCollection({
        'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
        'precision_micro': torchmetrics.Precision(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
        'recall_micro': torchmetrics.Recall(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
        'f1_micro': torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='micro'),
        'precision_weighted': torchmetrics.Precision(task='multiclass', num_classes=NUM_CLASSES, average='weighted'),
        'recall_weighted': torchmetrics.Recall(task='multiclass', num_classes=NUM_CLASSES, average='weighted'),
        'f1_weighted': torchmetrics.F1Score(task='multiclass', num_classes=NUM_CLASSES, average='weighted')
    }).to(device)

    print("training...")
    best_val_f1_weighted = -1.0
    best_model_state_dict = None

    for epoch in range(NUM_EPOCHS):
        print(f"\nFold {fold_idx + 1}/{NUM_FOLDS} - Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_metrics_results = engine.train_one_epoch(
            current_model,
            train_loader,
            current_criterion,
            current_optimizer,
            device,
            val_metrics_computer.clone(),
            epoch + 1
        )

        val_loss, val_metrics_results, all_val_preds, all_val_targets = engine.evaluate_model(
            current_model,
            val_loader,
            current_criterion,
            device,
            val_metrics_computer,
            epoch + 1
        )

        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")
        print(f"Val f1 (weighted): {val_metrics_results['f1_weighted']:.4f}")
        print(f"Val Accuracy: {val_metrics_results['accuracy']:.4f}")

        if val_metrics_results['f1_weighted'] > best_val_f1_weighted:
            best_val_f1_weighted = val_metrics_results['f1_weighted']
            best_model_state_dict = current_model.state_dict().copy()
            print(f"New best validation f1 weighted: {best_val_f1_weighted:.4f}")

    print(f"\nTraining completed for Fold {fold_idx + 1}")
    print(f"Best validation f1 weighted: {best_val_f1_weighted:.4f}")

    if best_model_state_dict is not None:
        current_model.load_state_dict(best_model_state_dict)
        torch.save(current_model.state_dict(), f'final_model_fold_{fold_idx}.pth')

    #final eval (same as best)
    print("Final eval:")
    final_val_loss, final_val_metrics, final_all_val_preds, final_all_val_targets = engine.evaluate_model(
        current_model,
        val_loader,
        current_criterion,
        device,
        val_metrics_computer,
        epoch_num=-1
    )


    fold_result = {
        'fold_idx': fold_idx,
        'best_val_f1_weighted': best_val_f1_weighted,
        **final_val_metrics
    }
    fold_results.append(fold_result)

    # confmat
    final_preds_cpu = final_all_val_preds.cpu().numpy()
    final_targets_cpu = final_all_val_targets.cpu().numpy()

    cm = confusion_matrix(final_targets_cpu, final_preds_cpu)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1}\nF1 weighted: {final_val_metrics["f1_weighted"]:.4f}')
    plt.xlabel('Pred Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.show()

    print(f"\nFold {fold_idx + 1} Final Results:")
    print(f"  Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"  F1 micro: {final_val_metrics['f1_micro']:.4f}")
    print(f"  F1 weighted: {final_val_metrics['f1_weighted']:.4f}")
    print(f"  Precision weighted: {final_val_metrics['precision_weighted']:.4f}")
    print(f"  Recall weighted: {final_val_metrics['recall_weighted']:.4f}")

    print(f"Fold {fold_idx + 1} completed!")


print("All folds completed!")
print("------------------------------------")
print("Summary")



#agregate metrics mean and std
metric_keys = [
    'accuracy',
    'f1_micro', 'f1_weighted',
    'precision_micro', 'precision_weighted',
    'recall_micro', 'recall_weighted'
]

aggregated_results = {}

for metric_key in metric_keys:
    metric_values = []
    for fold_result in fold_results:
        value = fold_result[metric_key]
        if hasattr(value, 'item'):
            metric_values.append(value.item())
        else:
            metric_values.append(float(value))

    mean_value = np.mean(metric_values)
    std_value = np.std(metric_values)

    aggregated_results[metric_key] = {
        'mean': mean_value,
        'std': std_value,
        'values': metric_values
    }


# Primary metrics
print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Mean ± Std'}")
print("-" * 60)

for metric_key in metric_keys:
    mean_val = aggregated_results[metric_key]['mean']
    std_val = aggregated_results[metric_key]['std']
    metric_name = metric_key.replace('_', ' ').title()
    print(f"{metric_name:<20} {mean_val:<10.4f} {std_val:<10.4f} {mean_val:.4f} +- {std_val:.4f}")

print("-" * 60)

print(f"\n Fold by fold:")
print("-" * 60)
print(f"{'Fold':<6} {'Accuracy':<10} {'F1-Weighted':<12} {'F1-Micro':<10} {'Precision-W':<12} {'Recall-W':<10}")
print("-" * 60)
for i, fold_result in enumerate(fold_results):
    acc = fold_result['accuracy'].item() if hasattr(fold_result['accuracy'], 'item') else fold_result['accuracy']
    f1_w = fold_result['f1_weighted'].item() if hasattr(fold_result['f1_weighted'], 'item') else fold_result['f1_weighted']
    f1_m = fold_result['f1_micro'].item() if hasattr(fold_result['f1_micro'], 'item') else fold_result['f1_micro']
    prec_w = fold_result['precision_weighted'].item() if hasattr(fold_result['precision_weighted'], 'item') else fold_result['precision_weighted']
    rec_w = fold_result['recall_weighted'].item() if hasattr(fold_result['recall_weighted'], 'item') else fold_result['recall_weighted']

    print(f"{i+1:<6} {acc:<10.4f} {f1_w:<12.4f} {f1_m:<10.4f} {prec_w:<12.4f} {rec_w:<10.4f}")

print("-" * 60)


results_summary = {
    'experiment_config': {
        'num_folds': NUM_FOLDS,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'num_sampled_pixels': S_NUM_SAMPLED_PIXELS,
        'num_classes': NUM_CLASSES
    },
    'aggregated_metrics': aggregated_results,
    'fold_results': fold_results
}