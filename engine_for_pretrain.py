# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import time
from util import DataUpdater


def train_one_epoch_for_mae_ssl(train_loader, net, optimizer,
                                num_ite_per_epoch, lr_schedule_values,
                                wd_schedule_values, logger,
                                folder_num, i, mask_ratio):
    batch_time = DataUpdater()
    losses = DataUpdater()
    tm = time.time()
    net.train()
    device = next(net.parameters()).device
    for j, data in enumerate(train_loader):
        images, _, _ = data
        batch_size = images.size(0)
        images = images.to(device, non_blocking=True)
        loss, labels, _ = net(images, mask_ratio=mask_ratio)

        # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
        global_step = i * num_ite_per_epoch + j
        for _, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[global_step]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[global_step]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        logger.update(step=global_step, batch_loss=losses.avg)

        print('Folder: {}\t'
              'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {loss.val:.4f} (Avg:{loss.avg:.4f})\t'
              'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
            folder_num, i + 1, j * len(labels), len(train_loader.dataset), 100. * j / len(train_loader),
            loss=losses, batch_time=batch_time))
    logger.update(step=i, epoch_loss=losses.avg)
    print('the {} epoch training time {batch_time.sum:.3f}\n'.format(i, batch_time=batch_time))
