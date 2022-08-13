import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import os
import torch.optim as optim
from Rsnet import rsnet34,  rsnet18
from torchvision.models import resnet  # pytorch也提供了ResNet源码，并且给出了预训练模型参数，
import xlwt

# 表格用来存放正确率
workbook = xlwt.Workbook(encoding='utf-8')
booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
rowdata2_loss = []  # 存储每次迭代的正确率
rowdata2_acc = []
run_epoch = 30
epochs = list(range(run_epoch))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set2/flower_data/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)  # 总的mini_batch数

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=39)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

net = rsnet18()
# load pretrain weights
# model_weight_path = "./resnet34-pre.pth"
# 导入预训练参数，当前预训练参数来源于pytorch
# missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

# for param in net.parameters():
#     param.requires_grad = False
# change fc layer structure
inchannel = net.fc.in_features
net.fc = nn.Linear(inchannel, 39)  # 目前所使用的数据库花只有五类
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

best_acc = 0.0
save_path = './rsnet18.pth'
t1 = time.perf_counter()
for epoch in range(run_epoch):

    # train
    net.train()  # 不能忘记
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
    print()

    # validate
    net.eval()  # 不能忘记
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))  # eval model only have last output layer
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))
        loss = running_loss / step
        rowdata2_loss.append(loss)  # 使用 append() 添加元素
        rowdata2_acc.append(val_accurate)

print('Finished Training')
print()
print(time.perf_counter()-t1)
for i in range(len(rowdata2_loss)):
    row = i
    col = 0
    booksheet.write(row, col, rowdata2_loss[i])
workbook.save('rsnet18_plantVillage2_loss_5.28.xls')

for i in range(len(rowdata2_acc)):
    row = i
    col = 0
    booksheet.write(row, col, rowdata2_acc[i])
workbook.save('rsnet18_plantVillage2_acc_5.28.xls')

plt.figure()
plt.plot(epochs, rowdata2_loss, 'b', label='Validation loss')  # 进行实现
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, rowdata2_acc, 'b', label='Validation acc')  # 进行实现
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Validation acc')
plt.legend()
plt.show()
