import torch
from torchvision import transforms
from net import LeNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

from sklearn.metrics import confusion_matrix

model_path = '/Users/siwen/PycharmProjects/包络/Runs/exp17/train/best_model.pth'
model_dir = os.path.dirname(model_path)  # 提取目录路径
model_file = os.path.basename(model_path)  # 提取文件名

# 提取所有数字
exp_num = ''.join(filter(str.isdigit, model_path))
print(exp_num)

save_path = 'Runs/exp' + exp_num

if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = save_path + '/test'
if not os.path.exists(save_path):
    os.mkdir(save_path)

import time #时间


start = time.time()
# C 1500预测900
ROOT_test =  r"/Users/siwen/Documents/ML/800"
   # r"/Users/siwen/Documents/ML/E"
    # r"/Users/siwen/Documents/ML/800"
    # r'/Users/siwen/Downloads/C/1500'
# '/Users/siwen/PycharmProjects/包络/B_6/600'
# "/Users/siwen/Documents/B_4/B3_4/train"
# r'/Users/siwen/Documents/ML/D4'

'''3、损失和优化'''
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵

model = LeNet()
'''这里用了个异常处理结构'''
try:
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file), map_location=torch.device('cpu')))
except:
    '''如果模型是在GPU上训练的，需要将模型加载到CPU上'''
    model.load_state_dict(torch.load(os.path.join(model_dir, model_file), map_location=torch.device('cpu')))

transform = transforms.Compose([
    transforms.Resize((128,128)),#统一大小
    transforms.ToTensor(),#转换数据
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])#归一化
])

test_dataset = ImageFolder(ROOT_test,transform=transform)
test_dataloader = DataLoader(test_dataset)

#取文件路径
all_files = []
file_list = os.walk(ROOT_test)  # 获取当前路径下的所有文件和目录
for dirpath, dirnames, filenames in file_list:  # 从file_list中获得三个元素
    for file in filenames:
        all_files.append(os.path.join(dirpath, file))  # 用os.path.join链接文件名和路径，跟新进all_files列表里



classes = ['H', 'OF','IF','BF']  # 更新为 ['H', 'OF', 'BF']'IF',


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

log_file = open(save_path+'/log.txt', 'w')

def test(dataloader, model, loss_fn):
    loss_sum, current, n = 0.0, 0.0, 0
    model.eval()

    y_true = []
    y_pred = []
    predictions = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input, target = data

            output = model(input)
            loss = loss_fn(output, target)
            _, prad = torch.max(output, dim=1)
            predictions.append(prad)
            cur_acc = torch.sum(target == prad) / output.shape[0]

            loss_sum += loss.item()
            current += cur_acc.item()
            n = n + 1

            val_loss = loss_sum / n
            val_acc = current / n

            y_true += target.tolist()
            y_pred += prad.tolist()

            print("epoch:{0} This state maybe: {1}; The correct need to be: {2}".format(n, classes[prad[0]],
                                                                                        classes[target[0]]))

        end3 = time.time()
        log_file.write("Validation Accuracy: {:.2f}%\n".format(val_acc * 100))
        print("预测结束 运行时间：{:.3f}分钟".format((end3 - start) / 60))
        print("test_loss:{0},test_acc：{1}%".format(val_loss, val_acc * 100))

        labels = ['H', 'OF','IF','BF']  # ['H', 'OF', 'BF', 'IF']

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=12)  # 调整x轴标签字体大小
        ax.set_yticklabels(labels, fontsize=12)  # 调整y轴标签字体大小
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, "{:d}".format(cm[i, j]),
                               fontsize=12,
                               ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")

        plt.savefig(save_path + '/confusion_matrix.png')

        print(cm)


        from sklearn.metrics import classification_report

        report = classification_report(y_true, y_pred, target_names=classes, digits=4)

        print(report)

        with open(save_path+'/classification_report.txt', 'w') as f:
            f.write(report



        return val_loss, val_acc, y_true, y_pred

import matplotlib.pyplot as plt
import numpy as np

def simple_visualization(y_true, y_pred):
    # 获取唯一类别
    # 将列表转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 获取唯一类别
    unique_classes = np.unique(y_true)

    plt.figure(figsize=(8, 6))
    for label in unique_classes:
        # 获取对应类别的索引
        indices = np.where(y_true == label)

        # 绘制散点图
        plt.scatter(y_true[indices], y_pred[indices], label=f'Class {label}', alpha=0.7)

    plt.title('True vs. Predicted Labels')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.legend(loc='upper left')
    plt.plot([0, max(y_true)], [0, max(y_pred)], 'r--', lw=2)  # 绘制对角线
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Using your test function
a,b, y_true, y_pred = test(test_dataloader, model, criterion)
simple_visualization(y_true, y_pred)


log_file.close()