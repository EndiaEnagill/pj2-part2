import matplotlib.pyplot as plt
import os
import json

# 假设你已经有了四个数列：VGG_A 的 maxloss 和 minloss，以及 VGG_A_BN 的 maxloss 和 minloss
# 这些数列的长度应该相同
loss_vgg_a_max = [] 
loss_vgg_a_min = []  
loss_vgg_a_bn_max = []
loss_vgg_a_bn_min = []

with open("loss_data/vgg_a_0.1.json", 'r') as f:
    loss_a1 = json.load(f)

with open("loss_data/vgg_a_0.01.json", 'r') as f:
    loss_a2 = json.load(f)

with open("loss_data/vgg_a_bn_0.1.json", 'r') as f:
    loss_bn1 = json.load(f)

with open("loss_data/vgg_a_bn_0.01.json", 'r') as f:
    loss_bn2 = json.load(f)

for i in range(len(loss_a1)):
    l1 = float(loss_a1[i][-1])
    l2 = float(loss_a2[i][-1])
    loss_vgg_a_max.append(max(l1,l2))
    loss_vgg_a_min.append(min(l1,l2))


for i in range(len(loss_bn1)):
    l1 = float(loss_bn1[i][-1])
    l2 = float(loss_bn2[i][-1])
    loss_vgg_a_bn_max.append(max(l1,l2))
    loss_vgg_a_bn_min.append(min(l1,l2))


def plot_loss_landscape(loss1_max, loss1_min, loss2_max, loss2_min, label1='VGG_A', label2='VGG_A_BN', figures_path='./figures'):
    plt.figure(figsize=(10, 5))
    
    plt.plot(loss1_max, label=f'{label1} Max Loss', color='lightblue')
    plt.plot(loss1_min, label=f'{label1} Min Loss', color='lightblue', linestyle='--')
    plt.fill_between(range(len(loss1_max)), loss1_min, loss1_max, color='lightblue', alpha=0.2)

    plt.plot(loss2_max, label=f'{label2} Max Loss', color='pink')
    plt.plot(loss2_min, label=f'{label2} Min Loss', color='pink', linestyle='--')
    plt.fill_between(range(len(loss2_max)), loss2_min, loss2_max, color='pink', alpha=0.2)
    
    plt.title('Loss Landscape Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_landscape.png')
    plt.close()

# 调用函数绘制损失景观
plot_loss_landscape(loss_vgg_a_max, loss_vgg_a_min, loss_vgg_a_bn_max, loss_vgg_a_bn_min)