import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models.vgg import VGG_A
from models.vgg import VGG_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 0
batch_size = 128

# add our package dir to path 
module_path = os.getcwd()
home_path = module_path
loss_path = os.path.join(home_path, 'reports', 'loss')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(loss_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers)
for X,y in train_loader:
    print(f"Batch shape: {X.shape}, Label shape: {y.shape}")
    break

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    ## --------------------
    # Add code as needed
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            _, pred = torch.max(output.data, 1)
            total += y.size(0)
            correct += (pred==y).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"runs/experiment_{timestamp}")
    model.to(device)
    learning_curve = [] 
    train_accuracy_curve = [] 
    val_accuracy_curve = [] 
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        running_loss = 0
        correct = 0
        total = 0

        # loss_list = []  # use this to record the loss value of each step
        # grad = []  # use this to record the loss gradient of each step
        # learning_curve[epoch] = 0  # maintain this to plot the training curve

        for batch_idx, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, prediction = output.max(1)
            total += y.size(0)
            correct += (prediction==y).sum().item()

            writer.add_scalar('Training batch loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # 梯度检查
        total_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                total_grad += grad_mean
        grad = total_grad/len(list(model.parameters()))
        print(f"\nEpoch {epoch} - 平均梯度: {total_grad/len(list(model.parameters())):.6f}")
        grads.append(grad)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        learning_curve.append(epoch_loss)
        train_accuracy_curve.append(epoch_acc)

        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve.append(val_acc)

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

        writer.add_scalar('Training epoch loss', epoch_loss, epoch)
        writer.add_scalar('Training accuracy', epoch_acc, epoch)
        writer.add_scalar('Validation accuracy', val_acc, epoch)

        print(f'Epoch {epoch+1}/{epochs_n}:')
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}%')
        print(f'Val Acc: {val_acc*100:.2f}%')

        display.clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        ax1.plot(learning_curve, label="Training Loss")
        ax1.set_title("Training Loss Curve")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2.plot(train_accuracy_curve, label='Training Accuracy')
        ax2.plot(val_accuracy_curve, label='Validation Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # plt.savefig(os.path.join(figures_path, f'training_curves_epoch_{epoch}.png'))
        plt.close()
    
    writer.close()
    return learning_curve, val_accuracy_curve, grads


# Train your model
# feel free to modify
epo = 20
loss_save_path = loss_path
grad_save_path = loss_path

learning_rate = [0.1, 0.01, 0.001, 0.0001]

# VGG_A
for lr in learning_rate:
    print("-"*60)
    print(f"learning rate: {lr}")
    print("model: VGG_A")

    set_random_seeds(seed_value=2020, device=device)
    model = VGG_A()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    best_model_path = os.path.join(models_path, f'vgg_a_best_{lr}.pth')

    loss, val_acc , grads= train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, best_model_path=best_model_path)
    np.savetxt(os.path.join(loss_save_path, f'loss_{lr}.txt'), loss, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, f'val_acc_{lr}.txt'), val_acc, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, f'grads_{lr}.txt'), grads, fmt='%s', delimiter=' ')

    # VGG_BatchNorm
    print(f"learning rate: {lr}")
    print("model: VGG_A_BN")

    set_random_seeds(seed_value=2020, device=device)
    model_bn = VGG_BatchNorm()
    optimizer_bn = torch.optim.SGD(model_bn.parameters(), lr=lr)
    best_model_path_bn = os.path.join(models_path, f'vgg_a_bn_best_{lr}.pth')

    loss_bn, val_acc_bn, grads_bn = train(model_bn, optimizer_bn, criterion, train_loader, val_loader,
                           epochs_n=epo, best_model_path=best_model_path_bn)

    np.savetxt(os.path.join(loss_save_path, f'loss_bn_{lr}.txt'), loss_bn, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, f'val_acc_bn_{lr}.txt'), val_acc_bn, fmt='%s', delimiter=' ')
    np.savetxt(os.path.join(loss_save_path, f'grads_bn_{lr}.txt'), grads_bn, fmt='%s', delimiter=' ')


# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
# min_curve = []
# max_curve = []
# for step in range(len(loss)):  # 假设所有loss列表长度相同
#     # 收集当前step所有模型的loss值
#     current_losses = [loss[step], loss_bn[step]]  # 可以添加更多模型
    
#     min_curve.append(min(current_losses))
#     max_curve.append(max(current_losses))
# # 需要增加plot的代码

# # Use this function to plot the final loss landscape,
# # fill the area between the two curves can use plt.fill_between()
# def plot_loss_landscape(loss1, loss2, label1='VGG_A', label2='VGG_A_BN'):
#     plt.figure(figsize=(10, 5))
#     plt.plot(loss1, label=label1)
#     plt.plot(loss2, label=label2)
#     plt.title('Loss Landscape Comparison')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))
#     plt.close()

# plot_loss_landscape(loss, loss_bn)

print("Training completed. Use 'tensorboard --logdir=runs' to view training metrics.")