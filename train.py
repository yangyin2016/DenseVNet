import os
from time import time

import torch
from model.vnet import get_net
from data_loader.data_loader import MyDataset
from torch.utils.data import DataLoader
from loss.avg_dice_loss import AvgDiceLoss

# 超参数
on_server = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
torch.backends.cudnn.benchmark = True
Epoch = 3000
leaing_rate = 1e-4

batch_size = 1 if on_server is False else 4
num_workers = 1 if on_server is False else 2
pin_memory = False if on_server is False else True

# 模型
net = get_net(training=True)
net = torch.nn.DataParallel(net).cuda() if on_server else net.cuda()

# 数据
train_ds = MyDataset('D:/Projects/OrgansSegment/Data/data_preprocess/train_info.csv')
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 损失函数
loss_func = AvgDiceLoss()

# 优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

# 学习率衰减
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [900])

# 训练
start = time()
for epoch in range(Epoch):
    lr_decay.step()
    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()

        outputs_stage1, outputs_stage2 = net(ct)
        loss = loss_func(outputs_stage1, outputs_stage2, seg)

        mean_loss.append(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2 == 0:
            print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    # 每十个个epoch保存一次模型参数
    # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
    if epoch % 10 is 0:
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss.item(), mean_loss))
