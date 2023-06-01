import os
import time
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from data_reader import OCTDataSet
from timm.models import VGG, ResNet
from timm.models.layers import trunc_normal_
from pos_embed import interpolate_pos_embed
from util import DataUpdater, Matrics, save_checkpoint, save_checkpoint_ssl, get_full_model_name, cosine_scheduler, \
    TensorboardLogger, MultiCropWrapper, DINOLoss
from torch.optim import AdamW
from engine_for_pretrain import train_one_epoch_for_mae_ssl
from engine_for_finetune import train_one_epoch_for_mae_finetune, train_one_epoch_for_mae_finetune_ensemble_post_cross_attention
import model_pretrain
import model_finetune
import torch.backends.cudnn as cudnn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2' \
                                     ''


def worker(args, folder_num=0):
    # model configuration
    model_name = args.model_name
    num_class = args.num_class
    # input configuration, image size settings
    image_size = args.image_size
    patch_size = args.patch_size
    # model parameters initialization pattern
    pre_train = args.pre_train

    # MAE self supervised training configuration
    save_ckpt_freq = args.save_ckpt_freq
    mask_ratio = args.mask_ratio
    drop_path = args.drop_path
    norm_pix_loss = args.norm_pix_loss

    # MAE finetuning configuration
    use_model_ema = args.use_model_ema
    model_ema_decay = args.model_ema_decay
    model_ema_force_cpu = args.model_ema_force_cpu
    color_jitter = args.color_jitter
    aa = args.aa
    smoothing = args.smoothing
    train_interpolation = args.train_interpolation
    # * Random Erase params
    reprob = args.reprob
    remode = args.remode
    recount = args.recount
    resplit = args.resplit
    # * Mixup params
    mixup = args.mixup
    cutmix = args.cutmix
    cutmix_minmax = args.cutmix_minmax
    mixup_prob = args.mixup_prob
    mixup_switch_prob = args.mixup_switch_prob
    mixup_mode = args.mixup_mode
    # * Finetuning params
    finetune_checkpoint = args.finetune_checkpoint
    init_scale = args.init_scale
    # ensemble model params
    image_size_list = args.image_size_list
    ensemble = args.ensemble
    cross_attention = args.cross_attention
    post_merge_type = args.post_merge_type

    # coresponded data augmentation type for various train pattern
    augmentation = args.augmentation

    # training configuration
    # Trainer settings
    epochs = args.epochs
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_mode = args.train_mode
    gpu = args.gpu

    # optimizer configuration
    lr = args.lr
    weight_decay = args.weight_decay
    clip_grad = args.clip_grad
    weight_decay_end = args.weight_decay_end
    layer_decay = args.layer_decay
    warmup_lr = args.warmup_lr
    min_lr = args.min_lr
    warmup_epochs = args.warmup_epochs
    warmup_steps = args.warmup_steps

    # resume configuration
    resume = args.resume
    resume_file_name = args.resume_file_name
    resume_file = ""

    # fine-tune
    pre_ssl_train_path = args.pre_ssl_train_path

    val_set = None
    if train_mode == 'self_supervised':
        assert augmentation in ['mae_ssl_train'], 'ssl augmentation type must be dino or mae'
        train_set = OCTDataSet(args, folder_num=folder_num, mode='ssl', augmentation=augmentation)
    elif train_mode == 'fine_tune':
        assert augmentation in ['mae_fine_tune_train'], \
            'ssl fine tune augmentation type must be dino or mae'
        train_set = OCTDataSet(args, folder_num=folder_num, mode='train', augmentation=augmentation)
        val_set = OCTDataSet(args, folder_num=folder_num, mode='val', augmentation='val_or_test')
    else:
        print('not support train mode')
        sys.exit(-1)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    if val_set is not None:
        val_loader = DataLoader(dataset=val_set,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=False)
    else:
        val_loader = None
    net = None

    if train_mode == 'self_supervised':
        # 需要根据DINO和MAE具体使用模型
        if augmentation == 'mae_ssl_train':
            net = model_pretrain.__dict__[model_name](norm_pix_loss=norm_pix_loss)
            kind = model_name.split('_')[2]  # base or large
            pre_train_weight = ''
            if kind == 'base':
                pre_train_weight = 'mae_imagenet_weight/mae_pretrain_vit_base_full.pth'
            elif kind == 'large':
                pre_train_weight = 'mae_imagenet_weight/mae_pretrain_vit_large_full.pth'
            else:
                print('not support mae pretrain weight!')
                sys.exit(-1)
            mae_checkpoint = torch.load(pre_train_weight, map_location='cpu')
            msg = net.load_state_dict(mae_checkpoint['model'], strict=False)
            print(msg)

        else:
            print('not support self supervised train model')
            sys.exit(-1)

    elif train_mode == 'fine_tune':
        # 需要根据不同的预训练方式定义不同的模型，并加载对应的权重
        if augmentation == 'mae_fine_tune_train':
            # 创建微调模型
            if not ensemble:
                net = model_finetune.__dict__[model_name](
                    img_size=image_size,
                    num_classes=num_class,
                    drop_path_rate=drop_path,
                    global_pool=False
                )
            else:
                if cross_attention:
                    net = model_finetune.__dict__[model_name](
                        image_size_list=image_size_list,
                        num_classes=num_class,
                        drop_path_rate=drop_path,
                        global_pool=False
                    )

            # 加载mae自监督预训练好的模型的权重
            mae_ssl_checkpoint = torch.load(pre_ssl_train_path, map_location='cpu')
            print("Load mae self supervised training checkpoint from %s" % pre_ssl_train_path)
            checkpoint_model_state_dict = mae_ssl_checkpoint['state_dict']
            if not ensemble:
                net_state_dict = net.state_dict()
                for k in ['head.weight', 'head.bias']:
                    if k in checkpoint_model_state_dict and checkpoint_model_state_dict[k].shape != net_state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model_state_dict[k]
                # interpolate position embedding
                interpolate_pos_embed(net, checkpoint_model_state_dict)
                print('cha zhi')
                # load pre-trained model
                msg = net.load_state_dict(checkpoint_model_state_dict, strict=False)
                print(msg)
                trunc_normal_(net.head.weight, std=2e-5)
            else:
                transformer_list = [net.vision_transformer1, net.vision_transformer2,
                                    net.vision_transformer3]
                for net_vision_transformer in transformer_list:
                    net_state_dict = net_vision_transformer.state_dict()

                    for k in ['head.weight', 'head.bias']:
                        if k in checkpoint_model_state_dict and checkpoint_model_state_dict[k].shape != net_state_dict[
                            k].shape:
                            print(f"Removing key {k} from pretrained checkpoint")
                            del checkpoint_model_state_dict[k]
                    # interpolate position embedding
                    interpolate_pos_embed(net_vision_transformer, checkpoint_model_state_dict)
                    print('cha zhi')
                    # load pre-trained model
                    msg = net_vision_transformer.load_state_dict(checkpoint_model_state_dict, strict=False)
                    print(msg)

        else:
            print('not support fine tune model')
            sys.exit(-1)

    else:
        print('not support train mode')
        sys.exit(-1)

    optimizer = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    # configuration for MAE finetune
    # mixup function for MAE finetune
    mixup_fn = None
    mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=smoothing, num_classes=num_class)

    loss = None
    if train_mode == 'fine_tune':
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            loss = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            loss = torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(loss))
    else:
        # 这个地方需要根据不同的自监督方式定义对应的损失函数
        if train_mode == 'self_supervised':
            if augmentation == 'mae_ssl_train':
                loss = nn.MSELoss()

    device = torch.device('cuda:' + str(gpu))
    if net is not None:
        net = net.to(device)
    # print('==' * 50)
    loss = loss.to(device)

    best_acc = 0.0
    # configuration for MAE finetune
    # model_ema for MAE finetune
    model_ema = None
    if use_model_ema:
        model_ema = ModelEmaV2(net, model_ema_decay)
        print("Using EMA with decay = %.8f" % model_ema_decay)

    param_pattern = 'imagenet_pretrain' if pre_train else 'random_initial'

    if not ensemble:
        dir_postfix = os.path.join(train_mode, model_name, 'patch' + str(patch_size) + '_' + str(image_size),
                                   param_pattern, 'fold_' + str(folder_num))
    else:
        img_size_str = '_'.join(list(map(str, image_size_list)))
        dir_postfix = os.path.join(train_mode, 'mae', model_name,
                                   'patch' + str(patch_size) + '_' + img_size_str,
                                   'fold_' + str(folder_num))

    if resume:
        resume_dir = os.path.join('checkpoint', dir_postfix)
        resume_file = os.path.join(resume_dir, resume_file_name)

        # resume from checkpoint
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file_name))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if model_ema:
                model_ema.module.load_state_dict(checkpoint['ema_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    # 需要对其中的一些设置进行动态地修改
    # learning rate根据batch_size修改
    lr = lr * batch_size / 256
    # using the cosine learning rate scheduler
    num_ite_per_epoch = len(train_set) // batch_size
    lr_schedule_values = cosine_scheduler(lr, min_lr, epochs, num_ite_per_epoch, warmup_epochs)
    print('-' * 100)
    print(epochs)
    print('-' * 100)
    print(num_ite_per_epoch)
    print('-' * 100)

    # using the cosine weight decay scheduler
    if weight_decay_end is None:
        weight_decay_end = weight_decay
    wd_schedule_values = cosine_scheduler(weight_decay, weight_decay_end, epochs, num_ite_per_epoch)

    # momentum parameter is increased to 1. during training with a cosine schedule
    # using the TensorboardLogger(SummaryWriter) to log the metric during training process
    log_dir = os.path.join('log', dir_postfix)
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorboardLogger(log_dir=log_dir)

    checkpoint_dir = os.path.join('checkpoint', dir_postfix)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_settings = {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'net': net,
        'optimizer': optimizer,
        'num_ite_per_epoch': num_ite_per_epoch,
        'lr_schedule_values': lr_schedule_values,
        'wd_schedule_values': wd_schedule_values,
        'logger': logger,
        'train_mode': train_mode,
        'model_name': model_name,
        'loss': loss,
        'best_acc': best_acc,
        'folder_num': folder_num,
        'image_size': image_size,
        'patch_size': patch_size,
        'param_pattern': param_pattern,
        'checkpoint_dir': checkpoint_dir,
        'save_ckpt_freq': save_ckpt_freq,
        'mask_ratio': mask_ratio,
        'drop_path': drop_path,
        'norm_pix_loss': norm_pix_loss,
        'augmentation': augmentation,
        'mixup_fn': mixup_fn,
        'model_ema': model_ema,
        'ensemble': ensemble,
        'cross_attention': cross_attention,
        'post_merge_type': post_merge_type
    }
    if train_mode == 'self_supervised':
        self_supervised_train(start_epoch, epochs, **model_settings)
    else:
        mae_fine_tune(start_epoch, epochs, **model_settings)


def self_supervised_train(start_epoch, epochs, **model_settings):
    augmentation = model_settings['augmentation']
    assert augmentation in ['mae_ssl_train'], 'ssl augmentation type must be dino or mae'
    if augmentation == 'mae_ssl_train':
        mae_self_supervised(start_epoch, epochs, **model_settings)





def mae_self_supervised(start_epoch, epochs, **model_settings):
    """
    this function is used to do mae self supervised training
    :return:
    """

    train_loader = model_settings['train_loader']
    net = model_settings['net']
    optimizer = model_settings['optimizer']
    num_ite_per_epoch = model_settings['num_ite_per_epoch']
    lr_schedule_values = model_settings['lr_schedule_values']
    wd_schedule_values = model_settings['wd_schedule_values']
    logger = model_settings['logger']
    model_name = model_settings['model_name']
    loss = model_settings['loss']
    folder_num = model_settings['folder_num']
    checkpoint_dir = model_settings['checkpoint_dir']
    save_ckpt_freq = model_settings['save_ckpt_freq']
    mask_ratio = model_settings['mask_ratio']

    for i in range(start_epoch, epochs):
        train_one_epoch_for_mae_ssl(train_loader=train_loader, net=net, optimizer=optimizer,
                                    num_ite_per_epoch=num_ite_per_epoch, lr_schedule_values=lr_schedule_values,
                                    wd_schedule_values=wd_schedule_values, logger=logger,
                                    folder_num=folder_num, i=i, mask_ratio=mask_ratio,
                                    )
        # 测试

        if i % save_ckpt_freq == 0 or (i + 1) == epochs:
            save_checkpoint({
                'epoch': i + 1,
                'arch': model_name,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_dir=checkpoint_dir, is_best=False)



def mae_fine_tune(start_epoch, epochs, **model_settings):
    train_loader = model_settings['train_loader']
    val_loader = model_settings['val_loader']
    net = model_settings['net']
    optimizer = model_settings['optimizer']
    num_ite_per_epoch = model_settings['num_ite_per_epoch']
    lr_schedule_values = model_settings['lr_schedule_values']
    wd_schedule_values = model_settings['wd_schedule_values']
    logger = model_settings['logger']
    train_mode = model_settings['train_mode']
    model_name = model_settings['model_name']
    loss = model_settings['loss']
    best_acc = model_settings['best_acc']
    folder_num = model_settings['folder_num']
    checkpoint_dir = model_settings['checkpoint_dir']
    mixup_fn = model_settings['mixup_fn']
    model_ema = model_settings['model_ema']
    ensemble = model_settings['ensemble']
    cross_attention = model_settings['cross_attention']
    post_merge_type = model_settings['post_merge_type']

    for i in range(start_epoch, epochs):
        if not ensemble:
            train_one_epoch_for_mae_finetune(train_loader=train_loader, net=net, optimizer=optimizer,
                                             model_ema=model_ema, mixup_fn=mixup_fn,
                                             num_ite_per_epoch=num_ite_per_epoch,
                                             lr_schedule_values=lr_schedule_values,
                                             wd_schedule_values=wd_schedule_values,
                                             logger=logger, folder_num=folder_num, i=i, criterion=loss)
            # 测试
            val_acc = valid(val_loader=val_loader, net=net, criterion=loss,
                            folder_num=folder_num, logger=logger, i=i)
        else:

            train_one_epoch_for_mae_finetune_ensemble_post_cross_attention(train_loader=train_loader, net=net,
                                                                           optimizer=optimizer,
                                                                           model_ema=model_ema, mixup_fn=mixup_fn,
                                                                           num_ite_per_epoch=num_ite_per_epoch,
                                                                           lr_schedule_values=lr_schedule_values,
                                                                           wd_schedule_values=wd_schedule_values,
                                                                           logger=logger, folder_num=folder_num,
                                                                           i=i,
                                                                           criterion=loss)
            # 测试
            val_acc = valid_mae_ensemble_post_cross_attention(val_loader=val_loader, net=net, criterion=loss,
                                                              folder_num=folder_num, logger=logger, i=i)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best or (i + 1) % 10 == 0:
            save_checkpoint({
                'epoch': i + 1,
                'arch': model_name,
                'state_dict': net.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'model_ema_state_dict': model_ema.module.state_dict() if model_ema else None
            }, checkpoint_dir=checkpoint_dir, is_best=is_best)


def train_one_epoch(train_loader, net, optimizer, criterion,
                    num_ite_per_epoch, lr_schedule_values,
                    wd_schedule_values, logger,
                    folder_num, i):
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
        outputs = net(images)  # batch_size, num_class
        if isinstance(outputs, tuple):  # deit will return a tuple
            outputs = outputs[0]
        loss = criterion(outputs, labels)
        global_step = i * num_ite_per_epoch + j
        if not (isinstance(net, VGG) or isinstance(net, ResNet)):
            # 根据cosine scheduler 修改optimizer的learning rate 和 weight decay
            for _, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[global_step]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), batch_size)
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


def valid(val_loader, net, criterion, folder_num, logger, i):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    net.eval()
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Matrics()
    with torch.no_grad():
        tm = time.time()
        for j, data in enumerate(val_loader):
            images, labels, _ = data
            batch_size = len(labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            if isinstance(outputs, tuple):  # deit will return a tuple
                outputs = outputs[0]
            loss = criterion(outputs, labels)

            acc1 = metrics.accuracy(output=outputs, target=labels, topk=(1,))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - tm)
            tm = time.time()

            print('Folder: {}\t'
                  'Validation Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {loss.val:.4f}(Avg:{loss.avg:.4f})\t'
                  'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
                  'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
                folder_num, i + 1, j * batch_size, len(val_loader.dataset), 100. * j / len(val_loader),
                loss=losses, top1=top1, batch_time=batch_time))

        logger.update(step=i, valid_epoch_loss=losses.avg)
        logger.update(step=i, valid_epoch_top1=top1.avg)
        print(' * AccTop1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def valid_mae_ensemble_post_cross_attention(val_loader, net, criterion, folder_num, logger, i):
    batch_time = DataUpdater()
    losses = DataUpdater()
    top1 = DataUpdater()
    net.eval()
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss().to(device)
    metrics = Matrics()
    with torch.no_grad():
        tm = time.time()
        for j, data in enumerate(val_loader):
            images, labels, _ = data
            batch_size = len(labels)
            for k in range(len(images)):
                images[k] = images[k].to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = net(images)
            loss = criterion(outputs[0], labels)
            for output in outputs[1:]:
                loss += criterion(output, labels)
            outputs = F.softmax(torch.stack(outputs, dim=0), dim=-1)  # [3, batch_size, 5]
            branch_weight = F.sigmoid(net.loss_weight.data)[:, None, None]  # [3, 1, 1]
            outputs = outputs * branch_weight

            outputs = outputs.transpose(0, 1).mean(dim=1)  # [ batch_size, 5]
            outputs = F.softmax(outputs, -1)

            acc1 = metrics.accuracy(output=outputs, target=labels, topk=(1,))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - tm)
            tm = time.time()

            print('Folder: {}\t'
                  'Validation Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Loss: {loss.val:.4f}(Avg:{loss.avg:.4f})\t'
                  'AccTop1: {top1.val:.3f} (Avg:{top1.avg:.3f})\t'
                  'Time: {batch_time.val:.3f} (Avg:{batch_time.avg:.3f})'.format(
                folder_num, i + 1, j * batch_size, len(val_loader.dataset), 100. * j / len(val_loader),
                loss=losses, top1=top1, batch_time=batch_time))

        logger.update(step=i, valid_epoch_loss=losses.avg)
        logger.update(step=i, valid_epoch_top1=top1.avg)
        print(' * AccTop1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    arg_setting = argparse.ArgumentParser()
    arg_setting.add_argument('--model_name', type=str, default='Vit-T', help='the model we used.')
    arg_setting.add_argument('--num_class', default=5, help="class num", type=str)

    # input configuration, image size settings
    arg_setting.add_argument('--image_size', type=int, default=224, help='the image size input to the model')
    arg_setting.add_argument('--patch_size', type=int, default=16, help='the patch size for a divided patch')

    # model parameters initialization pattern
    arg_setting.add_argument('--pre_train', action='store_true', help="weight initialized by weight pretrained from "
                                                                      "imageNet")
    arg_setting.add_argument('--pre_ssl_train', action='store_true',
                             help="load the pretrained model learned by MAE")

    # MAE self supervised training configuration
    arg_setting.add_argument('--save_ckpt_freq', default=20, type=int)
    arg_setting.add_argument('--mask_ratio', default=0.75, type=float,
                             help='ratio of the visual tokens/patches need be masked')
    arg_setting.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                             help='Drop path rate (default: 0.1)')
    arg_setting.add_argument('--norm_pix_loss', default=True, type=bool,
                             help='normalized the target patch pixels')

    # MAE finetuning configuration
    arg_setting.add_argument('--use_model_ema', action='store_true', default=False)
    arg_setting.add_argument('--model_ema_decay', type=float, default=0.999, help='')
    arg_setting.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    arg_setting.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                             help='Color jitter factor (default: 0.4)')
    arg_setting.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                             help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    arg_setting.add_argument('--smoothing', type=float, default=0.0,
                             help='Label smoothing (default: 0.1)')
    arg_setting.add_argument('--train_interpolation', type=str, default='bicubic',
                             help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # * Random Erase params
    arg_setting.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                             help='Random erase prob (default: 0.25)')
    arg_setting.add_argument('--remode', type=str, default='pixel',
                             help='Random erase mode (default: "pixel")')
    arg_setting.add_argument('--recount', type=int, default=1,
                             help='Random erase count (default: 1)')
    arg_setting.add_argument('--resplit', action='store_true', default=False,
                             help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    arg_setting.add_argument('--mixup', type=float, default=0.0,
                             help='mixup alpha, mixup enabled if > 0. default=0.8')
    arg_setting.add_argument('--cutmix', type=float, default=0.0,
                             help='cutmix alpha, cutmix enabled if > 0. default=1.0')
    arg_setting.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                             help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    arg_setting.add_argument('--mixup_prob', type=float, default=1.0,
                             help='Probability of performing mixup or cutmix when either/both is enabled')
    arg_setting.add_argument('--mixup_switch_prob', type=float, default=0.5,
                             help='Probability of switching to cutmix when both mixup and cutmix enabled')
    arg_setting.add_argument('--mixup_mode', type=str, default='batch',
                             help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # emsemble model configuration
    arg_setting.add_argument('--ensemble', action='store_true', help="if need to ensemble the mae finetune model")
    arg_setting.add_argument('--cross_attention', action='store_true', help="if need to use the cross attention")
    arg_setting.add_argument('--post_merge_type', type=str, default='mean', choices=['mean', 'max'],
                             help="merge type for ensembling the mae finetune model")
    arg_setting.add_argument('--image_size_list', type=int, nargs='+', default=[224, 326, 512],
                             help='the multi-scale image size to form the image pyramid')

    # * Finetuning params
    arg_setting.add_argument('--finetune_checkpoint', default='', help='finetune from checkpoint')
    arg_setting.add_argument('--init_scale', default=0.001, type=float)

    # coresponded data augmentation type for various train pattern
    arg_setting.add_argument('--augmentation', type=str, default='mae_ssl_train',
                             choices=['mae_ssl_train', 'mae_fine_tune_train']
                             , help="augmentation type for different training pattern")

    # training configuration
    # Trainer settings

    arg_setting.add_argument('--epochs', type=int, default=60, help="training epoch")
    arg_setting.add_argument('--start_epoch', type=int, default=0, help="start epoch")
    arg_setting.add_argument('--batch_size', type=int, default=32, help="training batch size")
    arg_setting.add_argument('--num_workers', type=int, default=1, help="data loader thread")
    # train_mode configuration
    arg_setting.add_argument('--train_mode', type=str, default="self_supervised",
                             choices=['self_supervised', 'fine_tune'],
                             help="training mode",
                             )
    arg_setting.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
    arg_setting.add_argument('--patient_epochs', type=int, default=8, help='the early stopping epochs.')
    arg_setting.add_argument('--print_every', type=int, default=100,
                             help='the number of iterations printing the training loss.')
    arg_setting.add_argument('--checkpoint', type=str, default='checkpoint',
                             help='the directory to save the model weights.')
    arg_setting.add_argument('--seed', default=0, type=int)
    arg_setting.add_argument('--log_dir', default='log', type=str, help='summary writer log directory')

    # optimizer
    arg_setting.add_argument('--optimizer', default='AdamW', type=str, help='optimizer')
    arg_setting.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    arg_setting.add_argument('--weight_decay', default=0.02, type=float, help="weight_decay")
    arg_setting.add_argument('--clip_grad', type=float, default=None, help='clip gradient norm')
    arg_setting.add_argument('--weight_decay_end', type=float, default=0.005, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    arg_setting.add_argument('--layer_decay', type=float, default=0.75)
    arg_setting.add_argument('--warmup_lr', type=float, default=1e-6,
                             help='warmup learning rate (default: 1e-6)')
    arg_setting.add_argument('--min_lr', type=float, default=1e-6,
                             help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    arg_setting.add_argument('--warmup_epochs', type=int, default=12,
                             help='epochs to warmup LR, if scheduler supports')
    arg_setting.add_argument('--warmup_steps', type=int, default=-1,
                             help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # resume
    arg_setting.add_argument('--resume', action='store_true', help="if need to resume from the latest checkpoint")
    arg_setting.add_argument('--resume_file_name', default='checkpoint.pth', help="path to checkpoint", type=str)

    # fine-tune
    arg_setting.add_argument('--pre_ssl_train_path', default='checkpoint.pth', help="file path for checkpoint",
                             type=str)
    args = arg_setting.parse_args()

    # set seed for reproduce
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # folders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    folders = [0]
    for folder_num in folders:
        worker(args, folder_num=folder_num)
