'''训练模型'''
import torch
from net import LeNet# 导入写好的网络模型
from torch.optim import lr_scheduler  # 优化器
import os#文件处理

# 处理数据集的库
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import time #时间
import matplotlib.pyplot as plt

start = time.time()

# 解决中文现实问题
# plt.rcParams["font.sans-serif"] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

'''1、数据处理'''
# 数据集的路径
ROOT_train = r"/Users/siwen/PycharmProjects/包络/B_6 2/800"
# r'/Users/siwen/Downloads/E/3120'
# r"/Users/siwen/PycharmProjects/包络/B_6 2/1000"
#   r'/Users/siwen/Downloads/C/900'
ROOT_val = r"/Users/siwen/PycharmProjects/包络/B_6 val/800"
    # r'/Users/siwen/Downloads/E/3120的副本'
    # r"/Users/siwen/PycharmProjects/包络/B_6 val/1000"
    # r'/Users/siwen/Downloads/C的副本/1500'
    # r"/Users/siwen/PycharmProjects/包络/B_6 val/1000"

# 做数据归一化， 让图像的数据归一化到[0，1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 把图像做数据处理
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(128),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     normalize
# ])

train_transforms = transforms.Compose([
    transforms.Resize((128,128)),  # 把所有图像统一定义一个大小，论文里是224*224
    transforms.RandomVerticalFlip(),  # 随机垂直旋转，让数据更多做数据增强
    transforms.ToTensor(),  # 把图片转换为张量数据
    normalize  # 归一化
])


val_transforms = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    normalize
])

# # 数据最终处理
train_dataset = ImageFolder(ROOT_train, transform=train_transforms,)
val_dataset = ImageFolder(ROOT_val, transform=val_transforms)

# 把数据分批次bacth,shuffle=True 打乱数据集
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# val_dataset = ImageFolder(ROOT_test, transform=val_transforms)

# 导入GPU寻训练,N卡需要安装cuda，我苹果电脑没有
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''2、导入模型'''
model = LeNet().to(device)

'''3、损失和优化'''
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # model.parameters()把模型参数传入

# # 学习率每隔10轮变为原来的0.5
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

'''4、训练'''


def train(dataloader, model, loss_fn, optimizer):
    loss_sum, current, n = 0.0, 0.0, 0
    for batch_idx, data in enumerate(dataloader):
        input, target = data#.to(device)
        optimizer.zero_grad()  # 梯度清零

        output = model(input)  # 训练
        loss = loss_fn(output, target)  # 算损失
        _, prad = torch.max(output, dim=1)  # 去最高值
        cur_acc = torch.sum(target == prad) / output.shape[0]  # 算精确率

        loss.backward()  # 反向传播
        optimizer.step()  # 优化梯度

        n = n + 1  # 总轮次
        loss_sum += loss.item()  # 每批次总的损失值
        current += cur_acc.item()  # 每批次总的精确率

    train_loss = loss_sum / n  # 计算每批次平均损失值
    train_acc = current / n  # 计算每批次平均精确度
    print(train_acc)
    print("train_loss:{0}%,train_acc：{1}%".format(train_loss, train_acc * 100))
    end1 = time.time()
    print("训练结束 运行时间：{:.3f}分钟".format((end1 - start) / 60))
    return train_loss, train_acc


# 定义验证函数
def val(dataloader, model, loss_fn):
    loss_sum, current, n = 0.0, 0.0, 0
    model.eval()#将模型转化为验证模式
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input, target = data# .to(device)

            output = model(input)  # 训练
            loss = loss_fn(output, target)  # 算损失
            _, prad = torch.max(output, dim=1)  # 去最高值
            cur_acc = torch.sum(target == prad) / output.shape[0]  # 算精确率

            loss_sum += loss.item()  # 每批次总的损失值
            current += cur_acc.item()  # 每批次总的精确率
            n = n + 1  # 总轮次

        val_loss = loss_sum / n  # 计算每批次平均损失值
        val_acc = current / n  # 计算每批次平均精确度

        end3 = time.time()
        print("预测结束 运行时间：{:.3f}分钟".format((end3 - start) / 60))
        print("val_loss:{0}%,val_acc：{1}%".format(val_loss, val_acc * 100))
        return val_loss, val_acc




# 定义列表接收数据，画图需要用
t_loss = []
t_acc = []
v_loss = []
v_acc = []


path = 'Runs'
if not os.path.exists(path):
    os.mkdir(path)

all_exp = os.listdir(path)
all_exp.sort()
if all_exp is None:
    path = path + '/exp' + str(0) + '/train'
    os.makedirs(path)
else:
    path = path + '/exp' + str(len(all_exp)) + '/train'
    os.makedirs(path)

log_file = open(path+'/log.txt', 'w')

# 开始训练
min_acc = 0
epoch = 50

for i in range(epoch):

    print(f"第{i+1}轮，开始训练>>>>>")
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer)
    val_loss, val_acc = val(val_dataloader, model, criterion)

    # 存入列表
    t_loss.append(train_loss)
    t_acc.append(train_acc)
    v_loss.append(val_loss)
    v_acc.append(val_acc)

    # 保存最好的模型文件
    if val_acc > min_acc:
        print(val_acc)
        foldoad = "save_model"
        if not os.path.exists(foldoad):
            os.mkdir("save_model")
        min_acc = val_acc  # 把最新的精度更新进去
        print(f"save best model,第{i + 1}轮")
        torch.save(model.state_dict(), path + '/best_model.pth')  # 修改保存路径

    log_file.write(
        f"Epoch {i + 1}: Train Loss = {train_loss}, Train Acc = {train_acc}, Val Loss = {val_loss}, Val Acc = {val_acc}\n")

    if i == epoch - 1:
        torch.save(model.state_dict(), path + '/last_model.pth')

    print('done!')

# matplot_loss(t_loss, v_loss)
# matplot_acc(t_acc, v_acc)
end2 = time.time()
print("程序结束,程序运行时间：{:.3f}分钟".format((end2 - start) / 60))

log_file.close()

epoch_list = list(range(1,epoch+1))

'''定义画图函数'''

'''定义画图函数'''
def matplot_loss(train_loss,val_loss):
    plt.plot(epoch_list,train_loss,label='train_loss')
    plt.plot(epoch_list,val_loss, label='test_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss value between training set')
    plt.show()
    plt.savefig(path + '/loss.png')

def matplot_acc(train_acc,val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='test_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('accuracy between training set')
    plt.show()
    plt.savefig(path+'/acc.png')


matplot_loss(t_loss, v_loss)
matplot_acc(t_acc, v_acc)
