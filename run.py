import os
import time
import sys
import json
import datetime
from traceback import print_exception

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, ConstantLR, SequentialLR
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
from torch import autocast, GradScaler

from torchinfo import summary

from utils.dataset import TumorMRIDataset, split_dataset_by_class, split_dataset_into_loaders

from model import get_umamba, get_umamba_classification, transform_deep_supervision

from nnunetv2.training.loss.bypass_bn import enable_running_stats, disable_running_stats
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss


def train_model_with_timing(train_loader, model, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    timer_order = [
        'reset',
        'data_load',
        'data_transfer',
        'zero_grad',
        'forward',
        'loss',
        'backward',
    ]
    timer_points = [time.perf_counter()]
    for images, labels in train_loader:
        timer_points.append(time.perf_counter())
        images, labels = images.to(device), labels.to(device)
        timer_points.append(time.perf_counter())      
        
        optimizer.zero_grad()
        timer_points.append(time.perf_counter())

        # Use autocast for mixed precision
        # with autocast(device.type):
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        outputs = model(images)
        timer_points.append(time.perf_counter())
        loss = criterion(outputs, labels)
        timer_points.append(time.perf_counter())

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            # Scale the loss before backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        timer_points.append(time.perf_counter())

        running_loss += loss.item()
        timer_points.append(time.perf_counter())
    # Now step the scheduler after optimizer step
    scheduler.step()
    return running_loss / len(train_loader), timer_points

# Training function
def train_model(train_loader, model, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        
        data = nn.functional.interpolate(data, size=(20,256,224), mode='trilinear', align_corners=False)
        target = nn.functional.interpolate(target, size=(20,256,224), mode='nearest')
        
        data = data.to(device, non_blocking=True)
        target = transform_deep_supervision(target)
        target = [i.to(device, non_blocking=True) for i in target]
        
    
        # first forward-backward step
        enable_running_stats(model)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.pre_step(zero_grad=True)

        # second step
        disable_running_stats(model)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(zero_grad=True)
        
        running_loss += loss.item()
    enable_running_stats(model)
    # Now step the scheduler after optimizer step
    scheduler.step()
    return running_loss / len(train_loader)
            
def train_model_classify(train_loader, model, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # Use autocast for mixed precision
        # with autocast(device.type):
        #     outputs = model(images)
        #     loss = criterion(outputs, labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            # Scale the loss before backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.item()
    # Now step the scheduler after optimizer step
    scheduler.step()
    return running_loss / len(train_loader)


# Test and Evaluation function
def test_model(test_loader, model, criterion, device, evaluate=False):
    label_manager_has_regions = False
    evals = {'loss': [], 'tp_hard': [], 'fp_hard': [], 'fn_hard': []}
    with torch.no_grad():
        model.eval()    
        for data, target in test_loader:
        
            data = nn.functional.interpolate(data, size=(20,256,224), mode='trilinear', align_corners=False)
            target = nn.functional.interpolate(target, size=(20,256,224), mode='nearest')
            
            data = data.to(device, non_blocking=True)
            target = transform_deep_supervision(target)
            target = [i.to(device, non_blocking=True) for i in target]
            
            output = model(data)
            loss = criterion(output, target)
            
            # For enable_deep_supervision: we only need the output with the highest output resolution (if DS enabled)
            output = output[0]
            target = target[0]

            # the following is needed for online evaluation. Fake dice (green line)
            axes = [0] + list(range(2, output.ndim))

            if label_manager_has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            else:
                # no need for softmax
                output_seg = output.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            # if self.label_manager.has_ignore_label:
            #     if not label_manager_has_regions:
            #         mask = (target != self.label_manager.ignore_label).float()
            #         # CAREFUL that you don't rely on target after this line!
            #         target[target == self.label_manager.ignore_label] = 0
            #     else:
            #         mask = 1 - target[:, -1:]
            #         # CAREFUL that you don't rely on target after this line!
            #         target = target[:, :-1]
            # else:
            #     mask = None

            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)

            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            
            if not label_manager_has_regions:
                # if we train with regions all segmentation heads predict some kind of foreground. In conventional
                # (softmax training) there needs tobe one output for the background. We are not interested in the
                # background Dice
                # [1:] in order to remove background
                tp_hard = tp_hard[1:]
                fp_hard = fp_hard[1:]
                fn_hard = fn_hard[1:]
            
            evals['loss'].append(loss.detach().cpu().numpy())
            evals['tp_hard'].append(tp_hard)
            evals['fp_hard'].append(fp_hard)
            evals['fn_hard'].append(fn_hard)
    
    loss_here = np.mean(evals['loss']).item()
    tp = np.sum(evals['tp_hard'], 0)
    fp = np.sum(evals['fp_hard'], 0)
    fn = np.sum(evals['fn_hard'], 0)
    
    global_dc_per_class = [i.item() for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
    mean_fg_dice = np.nanmean(global_dc_per_class).item()
    evaluation_results = {
        'mean_fg_dice': mean_fg_dice,
        'dice_per_class_or_region': global_dc_per_class,
        'val_losses': loss_here,
    }
    return loss_here, evaluation_results
    
def test_model_classify(test_loader, model, criterion, device, evaluate=False):
    model.eval()
    true_labels, pred_labels = [], []
    running_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    results = [pred_labels, running_loss / len(test_loader)]
    if evaluate:
        acc = accuracy_score(true_labels, pred_labels)
        cm = confusion_matrix(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        auc = roc_auc_score(true_labels, pred_labels)
        auc_pr = average_precision_score(true_labels, pred_labels)
        results.append((acc, cm.tolist(), f1, auc, auc_pr))
    return results


# Cross-validation function
def cross_validate(train_loader, model_class, criterion, optimizer_class, scheduler_class, device, num_epochs=20, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_loader.dataset)):
        print(f"Fold {fold+1}/{k_folds}")

        train_subset = Subset(train_loader.dataset, train_idx)
        val_subset = Subset(train_loader.dataset, val_idx)

        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Instantiate model and optimizer
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters())
        scheduler = scheduler_class(optimizer)

        # Initialize GradScaler for mixed precision training
        # scaler = GradScaler()
        scaler = None

        # Train and evaluate on each fold
        for epoch in range(num_epochs):
            train_loss = train_model(train_loader_fold, model, criterion, optimizer, scheduler, device, scaler)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

        _1, _2, (val_acc, _, val_f1, val_auc, val_auc_pr) = test_model(val_loader_fold, model, criterion, device, evaluate=True)
        fold_results.append((val_acc, val_f1, val_auc, val_auc_pr))

    # Return average metrics across folds
    avg_acc = np.mean([r[0] for r in fold_results])
    avg_f1 = np.mean([r[1] for r in fold_results])
    avg_auc = np.mean([r[2] for r in fold_results])
    avg_auc_pr = np.mean([r[3] for r in fold_results])

    return avg_acc, avg_f1, avg_auc, avg_auc_pr


# Results saving function
def save_results_to_file(file_path, epoch_count, evals_dict, extra_info:str=None):
    with open(file_path+'.log', 'a') as f:
        f.write(f"Epochs: {epoch_count}\n")
        for key, value in evals_dict.items():
            f.write(f"{key}: {value}\n")
        if extra_info:
            f.write(extra_info+'\n')
        f.write('\n')

def save_loss_plots_to_file(file_path, train_losses, test_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(file_path+'.png')
    plt.clf()
    
    # Save losses to json file
    with open(file_path+'.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'test_losses': test_losses}, f)

def save_lr_plot_to_file(file_path, lr_values):
    plt.figure()
    plt.plot(lr_values)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(file_path+'.png')
    plt.clf()

def save_model_to_file(file_path, model):
    torch.save(model.state_dict(), file_path+'.pth')
    print('Model saved to file')

def getTime():
    return datetime.datetime.now().strftime("%y%m%d-%Hh%Mm")

startTime = getTime()

# Paths and configurations
root_dir = os.getenv('DATASET_PATH', './MICCAI_BraTS_2019_Data_Training/')
# pretrained_path = './pretrain_weight/checkpoint_UU-Mamba.pth'
pretrained_path = './pretrain_weight/checkpoint_UUMamba_BraTS.pth'

batch_size = 4
data_limit = 100

initial_lr = 1e-2
weight_decay = 3e-5
num_epochs = 250

modalities = ['t1ce', 't1', 't2', 'flair']
# modalities = ['t1ce']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
print('Parameters:', batch_size, initial_lr, num_epochs, data_limit, weight_decay, modalities, device)

# Dataset and DataLoader
# dataset = TumorMRIDataset(root_dir, modalities=modalities, limit=data_limit, segmentation=True)
dataset = TumorMRIDataset(root_dir, modalities=modalities, limit=data_limit, segmentation=False)
nChannels, *imgSize = dataset[0][0].shape

train_loader, test_loader = split_dataset_into_loaders(dataset, train_ratio=0.8, batch_size=batch_size, random_seed=42)

# sys.exit()

# Model, Loss, Optimizer, and Scheduler
# model_params = {
#     'img_size': imgSize, # (155, 240, 240)
#     # 'patch_size': (5, 8, 8),
#     'patch_size': (5, 4, 4),
#     'in_chans': nChannels,
#     'num_classes': 2,
#     # 'depths': [2, 2, 6, 2],
#     # 'dims': [96, 192, 384, 768],
#     'depths': [4, 4, 4, 4],
#     'dims': [96, 192, 288, 384],
#     'dropout': 0.0,
#     'debug': False,
# }
# print('Model Params:', model_params)
# model_class = lambda: VisionMamba3D(**model_params)
# criterion = nn.CrossEntropyLoss()
# optimizer_class = lambda params: AdamW(params, lr=initial_lr, weight_decay=weight_decay)
# # scheduler_class = lambda opt: StepLR(opt, step_size=2, gamma=0.5)
# scheduler_class = lambda opt: SequentialLR(opt, schedulers=[
#     ConstantLR(opt, factor=0.8, total_iters=num_epochs//5),
#     StepLR(opt, step_size=num_epochs//10, gamma=0.5)
# ], milestones=[num_epochs//5])

# # Perform cross-validation
# # cv_acc, cv_f1, cv_auc, cv_auc_pr = cross_validate(train_loader, model_class, criterion, optimizer_class, scheduler_class, device, num_epochs=num_epochs, k_folds=5)

# # Train on the full training set and evaluate on the test set
# model = model_class().to(device)
# optimizer = optimizer_class(model.parameters())
# scheduler = scheduler_class(optimizer)

# model, criterion, optimizer, scheduler = get_umamba(initial_lr, weight_decay, num_epochs, device, pretrained_path=pretrained_path)
model, criterion, optimizer, scheduler = get_umamba_classification(initial_lr, weight_decay, num_epochs, device, pretrained_path=pretrained_path)


# print('Model Summary')
# summary(model, input_size=(batch_size, nChannels, *imgSize), depth=3)
# sys.exit()

# Initialize GradScaler for full training
# scaler = GradScaler()
scaler = None

# Create results directory if it doesn't exist
os.makedirs(f'results', exist_ok=True)
run_id = os.getenv('RUN_ID', startTime)
results_prefix = f'results/{run_id}'
print(f'Training with RUN_ID: {run_id} at {startTime}')

# Train model on the entire training set
# with torch.autograd.detect_anomaly(): # For debugging NaNs
train_losses, test_losses = [], []
lr_values = []
try:
    for epoch in range(num_epochs):
        # Visualize learning rate schedule later
        lr_values.append(optimizer.param_groups[0]['lr'])
        
        # train_loss = train_model(train_loader, model, criterion, optimizer, scheduler, device, scaler)
        # test_loss, evals = test_model(test_loader, model, criterion, device, evaluate=True)
        train_loss = train_model_classify(train_loader, model, criterion, optimizer, scheduler, device, scaler)
        test_loss, evals = test_model_classify(test_loader, model, criterion, device, evaluate=True)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print('Test Evaluation:\n', *(f'{k}: {v}\n' for k, v in evals.items()))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Overwrite results to file after each epoch
        # Save plot of learning rate values
        save_lr_plot_to_file(f'{results_prefix}-lr', lr_values)
        # Save plots of training and test losses to disk 
        save_loss_plots_to_file(f'{results_prefix}-losses', train_losses, test_losses)
        # Save test results to file
        save_results_to_file(f'{results_prefix}-results', epoch+1, evals)
        print('Training Epoch results saved to file')

except Exception as e:
    print('Error occured:', e)
    print_exception(e)
finally:
    if (train_losses):
        print('Training lasted from', startTime, 'to', getTime())
        # Save model to disk
        save_model_to_file(f'{results_prefix}-model', model)
    else:
        print('No training was performed.')

