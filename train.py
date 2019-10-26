import os
from time import time
import numpy as np
import torch
from model.cas_vnet import get_net
from data_loader.data_loader import MyDataset
from torch.utils.data import DataLoader
from loss.avg_dice_loss import AvgDiceLoss
from loss.wgt_dice_loss import WgtDiceLoss
from loss.focal_dice_loss import FocalDiceLoss
from utils import accuracy
from val import dataset_accuracy
from tensorboardX import SummaryWriter

# 超参数
organs_name = ['spleen', 'left kidney', 'gallbladder', 'esophagus',
               'liver', 'stomach', 'pancreas', 'duodenum']

on_server = True
resume_training = False
module_dir = './module/net45-0.712-0.682.pth'

os.environ['CUDA_VISIBLE_DEVICES'] = '0' if on_server is False else '1,2,3'
torch.backends.cudnn.benchmark = True
Epoch = 1000
leaing_rate = 1e-4

batch_size = 3 if on_server else 1
num_workers = 2 if on_server else 1
pin_memory = True if on_server else False

# 模型
net = get_net(training=True)
net = torch.nn.DataParallel(net).cuda() if on_server else net.cuda()
if resume_training:
    print('----------resume training-----------')
    net.load_state_dict(torch.load(module_dir))
    net.train()


# 数据
train_ds = MyDataset('csv_files/train_info.csv')
train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

# 损失函数
# loss_func = AvgDiceLoss()
loss_func = FocalDiceLoss()
# loss_func = DynWgtDiceLoss()

# 优化器
opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=0.0005)

# 学习率衰减
# lr_decay = torch.optim.lr_scheduler.StepLR(opt, 500, gamma=0.8)
lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', factor=0.5, patience=10)

# 训练
writer = SummaryWriter()
start = time()
for epoch in range(Epoch):
    mean_loss = []
    mean_acc = []
    orgs_accs = []
    epoch_start = time()
    for step, (ct, seg) in enumerate(train_dl):
        ct = ct.cuda()

        # switch model to training mode, clear gradient accumulators
        net.train()
        opt.zero_grad()

        # forward + backward + optimize
        outputs_stage1, outputs_stage2 = net(ct)
        loss = loss_func(outputs_stage1, outputs_stage2, seg)
        org_acc, acc = accuracy(outputs_stage2.cpu().detach().numpy(), seg.numpy())

        orgs_accs.append(org_acc)
        mean_loss.append(loss)
        mean_acc.append(acc)

        loss.backward()
        opt.step()

        if step % 4 == 0:
            s = 'epoch:{}, step:{}, loss:{:.3f}, accuracy:{:.3f}, time:{:.3f} min'.format(epoch, step, loss.item(), acc, (time() - start) / 60)
            os.system('echo %s' % s)

    mean_loss = sum(mean_loss) / len(mean_loss)
    mean_acc = sum(mean_acc) / len(mean_acc)

    lr_decay.step(mean_acc)  # 如果10个epoch内train acc不上升，则lr = lr * 0.5

    writer.add_scalar('train/loss', mean_loss, epoch)
    writer.add_scalar('train/accuracy', mean_acc, epoch)
    # writer.add_scalar('lr', lr_decay.get_lr(), epoch)  # ReduceLROnPlateau没有get_lr()方法
    writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

    orgs_accs = np.array(orgs_accs)
    org_mean_dice = [np.mean(np.array(list(set(orgs_accs[:, i]).difference(['None'])), dtype=np.float16))
                     for i in range(len(organs_name))]
    writer.add_scalars('orgs_dice', {name:org_mean_dice[i] for i, name in enumerate(organs_name)}, epoch)

    print(' '.join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, org_mean_dice)]))
    print('--- epoch:%d, mean loss:%.3f, mean accuracy:%.3f, epoch time:%.3f min\n'
          % (epoch, mean_loss, mean_acc, (time() - epoch_start) / 60))

    # 每十个个epoch保存一次模型参数
    # 网络模型的命名方式为：epoch轮数+train acc+val acc
    if epoch % 5 is 0:
        # 验证集accuracy
        val_org_acc, val_mean_acc = dataset_accuracy(net, 'csv_files/val_info.csv')
        print('------------------------')
        print('epoch:%d - train loss:%.3f, train accuracy:%.3f, validation accuracy:%.3f, time:%.3f min' %
              (epoch, mean_loss, mean_acc, val_mean_acc, (time() - start) / 60))
        print(' '.join(["%s:%.3f" % (i, j) for i, j in zip(organs_name, val_org_acc)]))
        writer.add_scalar('validation/accuracy', val_mean_acc, epoch)
        print('------------------------\n')

        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, mean_acc, val_mean_acc))


writer.close()
