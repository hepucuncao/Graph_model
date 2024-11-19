import torch
import random
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from net import GCN
import matplotlib.pyplot as plt

# 选择数据集（Cora、CiteSeer、PubMed等）
dataset_name = 'Cora'
dataset = Planetoid(root=f'Cora', name='Cora', transform=NormalizeFeatures())

# 模型参数
input_dim = dataset.num_node_features
hidden_dim = 128
output_dim = dataset.num_classes

# 加载数据
data = dataset[0]

# 创建模型
model = GCN(input_dim, hidden_dim, output_dim)

# 加载保存的模型参数
model.load_state_dict(torch.load("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/Graph/save_model/best_model.pth"))

# 模型推理时切换到评估模式
model.eval()

# 模型推理
def inference():
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
        out = model(data)
        pred = out.argmax(dim=1)  # 预测类别
    return pred

# 进行推理测试
def test(num_samples=20):
    pred = inference()

    # 获取训练集实际标签和预测标签
    train_mask = data.train_mask
    actual_labels = data.y[train_mask]  #实际标签
    predicted_labels = pred[train_mask]  #预测标签

    # 随机选择 num_samples 个样本的索引
    sample_indices = random.sample(range(train_mask.sum().item()), num_samples)

    # 打印实际标签与预测标签
    for sample_index in sample_indices:
        idx = train_mask.nonzero(as_tuple=True)[0][sample_index]  # 获取训练集中样本的实际索引
        actual = actual_labels[idx].item()
        predicted = predicted_labels[idx].item()
        print(f'Actual: {actual}, Predicted: {predicted}, Match: {actual == predicted}')

        # 可视化节点特征
        feature = data.x[idx].numpy().reshape(1, -1)  # 将特征重塑为1x特征数
        plt.imshow(feature, cmap='gray', aspect='auto')
        plt.title(f'Actual: {actual}, Predicted: {predicted}')
        plt.axis('off')
        plt.show()


# 进行推理测试
train_acc = test(num_samples=20)




'''
import torch
from net import LeNet5
from torch.autograd import Variable
from torchvision import datasets,transforms
from torchvision.transforms import ToPILImage

#数据集中的数据是向量格式，要输入到神经网络中要将数据转化为tensor格式
data_transform=transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_transform,download=True) #下载手写数字数据集
train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)
#batch_size:一组数据有多少个批次
# shuffle：是否打乱

#加载测试数据集
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_transform,download=True) #下载训练集
test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

#如果有显卡，可以转到GPU
device='cuda' if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
model=LeNet5().to(device)

#把模型加载进来
model.load_state_dict(torch.load("C:/Users/元气少女郭德纲/PycharmProjects/pythonProject1/DeepLearning/Graph/save_model/best_model.pth"))
#写绝对路径 win系统要求改为反斜杠

#获取结果
classes=[
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

#把tensor转化为图片，方便可视化
show=ToPILImage()

#进入验证
for i in range(20): #取前20张图片
    X,y=test_dataset[i][0],test_dataset[i][1]
    show(X).show()
    #把张量扩展为四维
    X=Variable(torch.unsqueeze(X, dim=0).float(),requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted,actual=classes[torch.argmax(pred[0])],classes[y]
        print(f'predicted:"{predicted}",actual:"{actual}"')
'''