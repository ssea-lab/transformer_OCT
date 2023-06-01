# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import time
import torch
import torch.nn.functional as F
from util import DataUpdater, Matrics


def train_one_epoch_for_mae_finetune(train_loader, net, criterion, optimizer,
                                     model_ema, mixup_fn, num_ite_per_epoch,
                                     lr_schedule_values, wd_schedule_values,
                                     logger, folder_num, i
                                     ):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    net.train()
    device = next(net.parameters()).device
    for j, data in enumerate(train_loader):
        images, labels, names = data
        batch_size = len(labels)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        outputs = net(images)  # batch_size, num_class
        loss = criterion(outputs, labels)

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
        # modelEMA 使用指数移动平滑更新参数
        if model_ema:
            model_ema.update(net)

        losses.update(loss.item(), batch_size)
        if mixup_fn is not None:
            top1.update(0, batch_size)  # 当存在mixup_fn的时候，为了方便，我们将accurancy置为0
        else:
            top1.update(metrics.accuracy(output=outputs, target=labels, topk=(1,))[0].item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        logger.update(step=global_step, batch_loss=losses.avg)
        logger.update(step=global_step, batch_top1=top1.avg)

        print('Folder: {}\t'
              'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {loss.val:.4f} (Avg:{loss.avg:.4f})\t'
              'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
              'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
            folder_num, i + 1, j * len(labels), len(train_loader.dataset), 100. * j / len(train_loader),
            loss=losses, top1=top1, batch_time=batch_time))
    logger.update(step=i, epoch_loss=losses.avg)
    logger.update(step=i, epoch_top1=top1.avg)
    print('the {} epoch training time {batch_time.sum:.3f}\n'.format(i, batch_time=batch_time))


def train_one_epoch_for_mae_finetune_ensemble_post_cross_attention(train_loader, net, criterion, optimizer,
                                                                   model_ema, mixup_fn, num_ite_per_epoch,
                                                                   lr_schedule_values, wd_schedule_values,
                                                                   logger, folder_num, i
                                                                   ):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    metrics = Matrics()
    tm = time.time()
    net.train()
    device = next(net.parameters()).device
    print_freq = 10
    for j, data in enumerate(train_loader):
        images, labels, names = data
        batch_size = len(labels)
        for k in range(len(images)):
            images[k] = images[k].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_list = []

        # if mixup_fn is not None:
        #     for k in range(len(images)):
        #         label_list.append(labels[:])
        #         images[k], label_list[k] = mixup_fn(images[k], label_list[k])

        outputs = net(images)  # batch_size, num_class
        # if mixup_fn is not None:
        #     # 将label_list中张量进行平均计算损失
        #     average_label = torch.mean(torch.stack(label_list, dim=0), dim=0)
        #     loss = criterion(outputs, average_label)
        #
        # else:
        loss = criterion(outputs[0], labels)
        loss = loss * F.sigmoid(net.loss_weight[0])
        for out_idx in range(1, len(outputs)):
            output = outputs[out_idx]
            loss += criterion(output, labels) * F.sigmoid(net.loss_weight[out_idx])
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
        # modelEMA 使用指数移动平滑更新参数
        if model_ema:
            model_ema.update(net)

        losses.update(loss.item(), batch_size)
        outputs = F.softmax(torch.stack(outputs, dim=0), dim=-1)  # [3, batch_size, 5]
        with torch.no_grad():
            branch_weight = F.sigmoid(net.loss_weight.data)[:, None, None]  # [3, 1, 1]
            outputs = outputs * branch_weight

        outputs = outputs.transpose(0, 1).mean(dim=1)  # [ batch_size, 5]
        outputs = F.softmax(outputs, -1)
        top1.update(metrics.accuracy(output=outputs, target=labels, topk=(1,))[0].item(), batch_size)
        batch_time.update(time.time() - tm)
        tm = time.time()
        logger.update(step=global_step, batch_loss=losses.avg)
        logger.update(step=global_step, batch_top1=top1.avg)

        print('Folder: {}\t'
              'Train Epoch: {} [{}/{} ({:.0f}%)]\t'
              'Loss: {loss.val:.4f} (Avg:{loss.avg:.4f})\t'
              'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
              'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
            folder_num, i + 1, j * len(labels), len(train_loader.dataset), 100. * j / len(train_loader),
            loss=losses, top1=top1, batch_time=batch_time))
    logger.update(step=i, epoch_loss=losses.avg)
    logger.update(step=i, epoch_top1=top1.avg)
    print('the {} epoch training time {batch_time.sum:.3f}\n'.format(i, batch_time=batch_time))
