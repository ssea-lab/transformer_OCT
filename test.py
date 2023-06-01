import os
import sys
import argparse
import torch
import timm
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import model_finetune
import vision_transformer as vits
import torch.backends.cudnn as cudnn
import yaml
import time
from torch.utils.data import DataLoader
from data_reader import OCTDataSet, TiffDataSet
from util import get_full_model_name
from torchvision.models.resnet import resnet50, resnet101
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score


def worker(args):
    test_kind = args.test_kind
    if test_kind in ['cross_validation', 'patch_test']:
        folders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for folder_num in folders:
            worker_patch(args, folder_num=folder_num)

    elif test_kind in ['tiff_refine3_test', 'tiff_xiangya_test', 'tiff_huaxi_test', 'tiff_whdx_test',
                       'tiff_huaxi_modified_test']:
        folders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for folder_num in folders:
            worker_image_tiff(args, folder_num=folder_num)
    elif test_kind in ['patch_test_expert', 'tiff_xiangya_expert']:
        worker_investigator(test_kind)
    else:
        print('not support test kind !')
        sys.exit(-1)


def worker_patch(args, folder_num=0):
    # model configuration
    model_name = args.model_name
    test_kind = args.test_kind
    test_mode = args.test_mode
    pre_train = args.pre_train
    # coresponded data augmentation type for various train pattern
    augmentation = args.augmentation

    num_class = args.num_class
    # input configuration, image size settings
    image_size = args.image_size
    patch_size = args.patch_size
    batch_size = args.batch_size

    # MAE self supervised training configuration
    drop_path = args.drop_path
    # ensemble model params
    image_size_list = args.image_size_list
    ensemble = args.ensemble
    cross_attention = args.cross_attention
    post_merge_type = args.post_merge_type

    use_model_ema = args.use_model_ema
    num_workers = args.num_workers
    gpu = args.gpu
    checkpoint = args.checkpoint
    mode = ''
    if test_kind == 'cross_validation':
        mode = 'val'
    elif test_kind == 'patch_test':
        mode = 'test'
    else:
        print('not support test kind!')
        sys.exit(-1)
    result_dir = args.result_dir

    test_set = OCTDataSet(args, folder_num=folder_num, mode=mode, augmentation='val_or_test')
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)
    net = None
    if test_mode == 'fine_tune':
        # 根据不同的预训练方式，加载不同的被微调过的模型
        # 需要根据不同的预训练方式定义不同的模型，并加载对应的权重
        if augmentation == 'mae_fine_tune_train':
            net = model_finetune.__dict__[model_name](
                image_size_list=image_size_list,
                num_classes=num_class,
                drop_path_rate=drop_path,
                global_pool=False
            )
        else:
            print('not support fine tune model')
            sys.exit(-1)

    else:
        print('not support train mode')
        sys.exit(-1)

    model_weight_path = ''
    dir_postfix = ''
    dir_total_postfix = ''
    # 接下来需要根据模型的不同训练模态确定需要加载的模型权重的目录
    if test_mode == 'supervised':
        param_pattern = 'imagenet_pretrain' if pre_train else 'random_initial'
        if model_name in ['Swin-T', 'Swin-S', 'Swin-B']:
            dir_postfix = os.path.join(test_mode, model_name, str(image_size), param_pattern,
                                       'fold_' + str(folder_num))
        else:
            dir_postfix = os.path.join(test_mode, model_name, 'patch' + str(patch_size) + '_' + str(image_size),
                                       param_pattern, 'fold_' + str(folder_num))

    elif test_mode == 'fine_tune':
        # 这个地方的dir_postfix要去根据下面的路径进行调整
        if not ensemble:
            kind = augmentation.split('_')[0]
            dir_postfix = os.path.join('fine_tune', kind, model_name,
                                       'patch' + str(patch_size) + '_' + str(image_size), 'fold_' + str(folder_num))
        else:
            img_size_str = '_'.join(list(map(str, image_size_list)))
            dir_postfix = os.path.join('fine_tune', 'mae', model_name,
                                       'patch' + str(patch_size) + '_' + img_size_str,
                                       'fold_' + str(folder_num))

    model_weight_path = os.path.join(checkpoint, dir_postfix, 'model_best.pth')

    # 得到模型权重的路径后，加载训练好的模型权重，以便于后面的模型的预测
    model_checkpoint = torch.load(model_weight_path, map_location='cpu')
    print('loading checkpoint from {}'.format(model_weight_path))
    model_state = model_checkpoint['state_dict']
    if use_model_ema:
        model_state = model_checkpoint['ema_state_dict']
    net.load_state_dict(model_state)

    # 模型加载完训练好的权重后，将模型移动到gpu上
    device = torch.device('cuda:' + str(gpu))
    if net is not None:
        net = net.to(device)
    # 根据我们测试的模型种类和和测试的类别确定我们保存结果的文件路径
    result_path_folder = os.path.join(result_dir, test_kind, dir_postfix)
    if not os.path.exists(result_path_folder):
        os.makedirs(result_path_folder)
    result_path = os.path.join(result_path_folder, 'val_result.txt')
    if not ensemble:
        valid(test_loader, net, result_path)
    else:
        valid_mae_ensemble_post_cross_attention(test_loader, net, result_path)

    dir_total_postfix = os.path.join(os.path.split(dir_postfix)[0], 'total')
    total_result_folder = os.path.join(result_dir, test_kind, dir_total_postfix)
    if not os.path.exists(total_result_folder):
        os.makedirs(total_result_folder)
    total_result_path = os.path.join(total_result_folder, 'total_val_result.txt')
    get_metrics_from_file(result_path, total_result_path)
    # print(total_result_path)


def valid(val_loader, net, result_path):
    net.eval()
    device = next(net.parameters()).device
    fout = open(result_path, 'w', encoding='utf8')
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            images, labels, img_paths = data
            dim = images.ndim
            if dim == 5:  # 如果tensor为5维，则取第一个元素[60, 3, H, W]
                images = images[0]

            tm = time.time()
            images = images.to(device, non_blocking=True)
            outputs = net(images)
            if isinstance(outputs, tuple):  # deit will return a tuple
                outputs = outputs[0]
            outputs = F.softmax(outputs, dim=1)
            # 接下来我们要把模型预测的结果和真实的标签写到结果文件中
            outputs = outputs.cpu().numpy()
            labels = labels.numpy()
            num_patchs = outputs.shape[0]
            if dim == 5:  # dim==5 证明是tiff的测试，否则的话是patch的测试
                output_labels = outputs.argmax(axis=1)  # [60]
                img_path = img_paths[0]
                zero_count = (output_labels == 0).sum()
                one_count = (output_labels == 1).sum()
                two_count = (output_labels == 2).sum()
                three_count = (output_labels == 3).sum()
                four_count = (output_labels == 4).sum()
                # 接下来计算每个patch的恶性概率
                outputs = outputs[: 10*(num_patchs // 10)]
                high_risk_prob = (outputs[:, -1] + outputs[:, -2]).reshape(10, num_patchs // 10)
                output_prob = high_risk_prob.reshape(-1)
                max_prob = high_risk_prob.max()
                position = high_risk_prob.argmax()
                h_index = int(np.floor(position // high_risk_prob.shape[1]))
                v_index = int(position % high_risk_prob.shape[1])
                final_predict_label = 0
                # 接下来根据十字验证法找到最大风险patch的上下左右，然后根据投票机制得到最终的预测类别
                l_h_index = h_index
                l_v_index = v_index - 1
                r_h_index = h_index
                r_v_index = (v_index + 1) % high_risk_prob.shape[1]
                u_h_index = h_index - 1
                u_v_index = v_index
                low_h_index = (h_index + 1) % high_risk_prob.shape[0]
                low_v_index = v_index
                risk_count = 0
                prob_threshold = 0.5
                risk_count += (high_risk_prob[h_index, v_index] > prob_threshold).sum()
                risk_count += (high_risk_prob[l_h_index, l_v_index] > prob_threshold).sum()
                risk_count += (high_risk_prob[r_h_index, r_v_index] > prob_threshold).sum()
                risk_count += (high_risk_prob[u_h_index, u_v_index] > prob_threshold).sum()
                risk_count += (high_risk_prob[low_h_index, low_v_index] > prob_threshold).sum()
                risk_count_threshhold = 1
                if risk_count > risk_count_threshhold:
                    final_predict_label = 1
                output_labels = ' '.join(list(map(str, output_labels.reshape(-1).tolist())))
                output_prob = ' '.join(list(map(str, output_prob.reshape(-1).tolist())))
                line_list = [img_path, zero_count, one_count, two_count, three_count, four_count, output_labels,
                             output_prob, max_prob, '(' + str(h_index) + ',' + str(v_index) + ')', final_predict_label]
                line = '\t'.join(list(map(str, line_list))) + '\n'
                fout.write(line)
            else:
                for i in range(len(img_paths)):
                    img_path = img_paths[i]
                    output = '\t'.join(map(str, outputs[i]))
                    label = str(labels[i])
                    fout.write(img_path + '\t' + output + '\t' + label + '\n')
    fout.close()


def valid_mae_ensemble_post_cross_attention(val_loader, net, result_path):
    net.eval()
    device = next(net.parameters()).device
    fout = open(result_path, 'w', encoding='utf8')
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            images, labels, img_paths = data
            dim = images[0].ndim
            if dim == 5:  # 如果tensor为5维，则取第一个元素[60, 3, H, W]
                for i in range(len(images)):
                    images[i] = images[i][0]

            for i in range(len(images)):
                images[i] = images[i].to(device, non_blocking=True)
            tm = time.time()
            # 由于切完之后patch数目太多，所以我们需要分开输入
            split_num = images[0].size(0) // 2
            images_1 = [image[:split_num, :, :, :] for image in images]
            images_2 = [image[split_num:, :, :, :] for image in images]
            output1 = net(images_1)
            output2 = net(images_2)
            output1 = F.softmax(torch.stack(output1, dim=0), dim=-1)  # [3, batch_size, 5]
            output2 = F.softmax(torch.stack(output2, dim=0), dim=-1)  # [3, batch_size, 5]
            output1 = output1.transpose(0, 1).mean(dim=1)  # [ batch_size, 5]
            output2 = output2.transpose(0, 1).mean(dim=1)  # [ batch_size, 5]
            outputs = torch.cat((output1, output2), dim=0)
            outputs = F.softmax(outputs, -1)
            # 接下来我们要把模型预测的结果和真实的标签写到结果文件中
            outputs = outputs.cpu().numpy()
            num_patchs = outputs.shape[0]
            labels = labels.numpy()
            if dim == 5:  # dim==5 证明是tiff的测试，否则的话是patch的测试
                output_labels = outputs.argmax(axis=1)  # [60]
                img_path = img_paths[0]
                zero_count = (output_labels == 0).sum()
                one_count = (output_labels == 1).sum()
                two_count = (output_labels == 2).sum()
                three_count = (output_labels == 3).sum()
                four_count = (output_labels == 4).sum()
                # 接下来计算每个patch的恶性概率
                outputs\
                    = outputs[: 10*(num_patchs // 10)]
                high_risk_prob = (outputs[:, -1] + outputs[:, -2]).reshape(10, num_patchs // 10)
                output_prob = high_risk_prob.reshape(-1)
                max_prob = high_risk_prob.max()
                position = high_risk_prob.argmax()
                h_index = int(np.floor(position // high_risk_prob.shape[1]))
                v_index = int(position % high_risk_prob.shape[1])
                final_predict_label = 0
                # 接下来根据十字验证法找到最大风险patch的上下左右，然后根据投票机制得到最终的预测类别
                l_h_index = h_index
                l_v_index = v_index - 1
                r_h_index = h_index
                r_v_index = (v_index + 1) % high_risk_prob.shape[1]
                u_h_index = h_index - 1
                u_v_index = v_index
                low_h_index = (h_index + 1) % high_risk_prob.shape[0]
                low_v_index = v_index
                risk_count = 0
                # thresh_hold = 0.5, 0.45, 0.4
                thresh_hold = 0.35
                risk_count += (high_risk_prob[h_index, v_index] > thresh_hold).sum()
                risk_count += (high_risk_prob[l_h_index, l_v_index] > thresh_hold).sum()
                risk_count += (high_risk_prob[r_h_index, r_v_index] > thresh_hold).sum()
                risk_count += (high_risk_prob[u_h_index, u_v_index] > thresh_hold).sum()
                risk_count += (high_risk_prob[low_h_index, low_v_index] > thresh_hold).sum()
                risk_count_threshhold = 2
                if risk_count > risk_count_threshhold:
                    final_predict_label = 1
                output_labels = ' '.join(list(map(str, output_labels.reshape(-1).tolist())))
                output_prob = ' '.join(list(map(str, output_prob.reshape(-1).tolist())))
                line_list = [img_path, zero_count, one_count, two_count, three_count, four_count, output_labels,
                             output_prob, max_prob, '(' + str(h_index) + ',' + str(v_index) + ')', final_predict_label]
                line = '\t'.join(list(map(str, line_list))) + '\n'
                fout.write(line)

            else:
                for i in range(len(img_paths)):
                    img_path = img_paths[i]
                    output = '\t'.join(map(str, outputs[i]))
                    label = str(labels[i])
                    fout.write(img_path + '\t' + output + '\t' + label + '\n')
            print('该批度数据推理所用时间：')
            print(time.time() - tm)
    fout.close()


def get_metrics_from_file(val_result_file, total_result_file):
    # 所要计算的指标包括五分类准确率(five class accuracy)，二分类准确率(binary class accuracy),
    # 特异性(sensitivity),灵敏度(specificity), PPV, NPV, AUC
    line_count = 0
    if os.path.exists(total_result_file):
        with open(total_result_file, 'r', encoding='utf8') as f:
            line_count = len(f.readlines())

    total_file_out = open(total_result_file, 'a', encoding='utf8')
    five_class_probabilities = []
    binary_class_probabilities = []
    five_class_labels = []
    binary_class_labels = []
    with open(val_result_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            img_path = line_list[0]
            five_class_pro = list(map(float, line_list[1: 6]))
            label = int(line_list[-1])
            five_class_probabilities.append(five_class_pro)
            five_class_labels.append(label)
            binary_class_pro = [five_class_pro[0] + five_class_pro[1] + five_class_pro[2],
                                five_class_pro[3] + five_class_pro[4]]
            binary_label = 0 if label <= 2 else 1
            binary_class_probabilities.append(binary_class_pro)
            binary_class_labels.append(binary_label)
    five_class_probabilities = np.array(five_class_probabilities)
    five_class_labels = np.array(five_class_labels)
    binary_class_probabilities = np.array(binary_class_probabilities)
    binary_class_labels = np.array(binary_class_labels)

    # 利用sklearn的接口计算得到五分类的准确率和二分类的准确率
    five_accracy = accuracy_score(y_true=five_class_labels, y_pred=np.argmax(five_class_probabilities, axis=1))
    # 利用sklearn的接口计算得到混淆矩阵
    five_confusion_matrix = confusion_matrix(y_true=five_class_labels,
                                             y_pred=np.argmax(five_class_probabilities, axis=1))

    # 接下来利用二分类的混淆矩阵计算各种指标，包括灵敏度，特异性，PPV, NPV
    # metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    metrics = cal_bin_metrics(y_true=binary_class_labels, y_pred=binary_class_probabilities)
    binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix = metrics[0], metrics[1], metrics[2], \
                                                                                metrics[3], metrics[4], metrics[5]

    # 接下来计算二分类的auc的值
    auc = roc_auc_score(y_true=binary_class_labels, y_score=binary_class_probabilities[:, 1])
    header = ['Five_Accuracy', 'Binary_Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'AUC',
              'Binary_Confusion_Matrix', 'Five_Confusion_Matrix']
    header = '\t'.join(header) + '\n'
    five_confusion_matrix = five_confusion_matrix.reshape(-1)
    binary_confusion_matrix = binary_confusion_matrix.reshape(-1)
    metrics = [five_accracy, binary_accracy, sensitivity, specifity, ppv, npv, auc,
               binary_confusion_matrix, five_confusion_matrix]
    metrics = '\t'.join(list(map(str, metrics))) + '\n'
    if line_count == 0:
        total_file_out.write(header)
    total_file_out.write(metrics)
    total_file_out.close()


def worker_investigator(test_kind):
    file_name = test_kind + '_label.txt'
    label_file_name = '_'.join(test_kind.split('_')[:-1]) + '.txt'
    expert_label_path = os.path.join('dataset', file_name)
    true_label_path = os.path.join('dataset', label_file_name)
    # 首先读取label文件，获取图片路径到真实标签的映射，即构造一个词典
    true_label_dict = {}
    with open(true_label_path, 'r', encoding='utf8') as fin:
        for line in fin:
            img_path = line.split('\t')[0]
            label = int(line.split('\t')[1])
            true_label_dict[img_path] = label

    investigator1_pres = []
    investigator2_pres = []
    investigator3_pres = []
    investigator4_pres = []
    true_labels = []
    with open(expert_label_path, 'r', encoding='utf8') as fin:
        lines = fin.readlines()
        for l_num, line in enumerate(lines):
            if l_num == 0:
                continue
            line_list = line.strip('\n').split('\t')
            if line_list[0] in true_label_dict:
                investigator1_pres.append([int(line_list[1])])
                investigator2_pres.append([int(line_list[2])])
                investigator3_pres.append([int(line_list[3])])
                investigator4_pres.append([int(line_list[4])])
                true_labels.append(true_label_dict[line_list[0]])

    investigator1_pres = np.array(investigator1_pres)
    investigator2_pres = np.array(investigator2_pres)
    investigator3_pres = np.array(investigator3_pres)
    investigator4_pres = np.array(investigator4_pres)
    true_labels = np.array(true_labels)
    # 由于patch test的专家标注是分为5类进行标注的，而tiff test的专家标注是分为两类标注的，所以分情况讨论
    if test_kind == 'patch_test_expert':
        # 首先计算一下五分类的准确率和五分类的混淆矩阵
        five_accracy_1 = accuracy_score(y_true=true_labels, y_pred=investigator1_pres)
        five_accracy_2 = accuracy_score(y_true=true_labels, y_pred=investigator2_pres)
        five_accracy_3 = accuracy_score(y_true=true_labels, y_pred=investigator3_pres)
        five_accracy_4 = accuracy_score(y_true=true_labels, y_pred=investigator4_pres)

        five_confusion_matrix_1 = confusion_matrix(y_true=true_labels, y_pred=investigator1_pres).reshape(-1)
        five_confusion_matrix_2 = confusion_matrix(y_true=true_labels, y_pred=investigator2_pres).reshape(-1)
        five_confusion_matrix_3 = confusion_matrix(y_true=true_labels, y_pred=investigator3_pres).reshape(-1)
        five_confusion_matrix_4 = confusion_matrix(y_true=true_labels, y_pred=investigator4_pres).reshape(-1)

        # 讲五分类的标签转化为二分类的标签
        investigator1_pres = np.where(investigator1_pres <= 2, 0, 1)
        investigator2_pres = np.where(investigator2_pres <= 2, 0, 1)
        investigator3_pres = np.where(investigator3_pres <= 2, 0, 1)
        investigator4_pres = np.where(investigator4_pres <= 2, 0, 1)
        true_labels = np.where(true_labels <= 2, 0, 1)

    # 接下来分别计算4个investigators的指标，即五分类准确率，二分类准确率，灵敏度，特异性，PPV, NPV
    # metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    inves_1_metrics = cal_bin_metrics(y_true=true_labels, y_pred=investigator1_pres)
    inves_2_metrics = cal_bin_metrics(y_true=true_labels, y_pred=investigator2_pres)
    inves_3_metrics = cal_bin_metrics(y_true=true_labels, y_pred=investigator3_pres)
    inves_4_metrics = cal_bin_metrics(y_true=true_labels, y_pred=investigator4_pres)

    # 根据是5分类还是2分类确定header和要输出的metrics
    header = ''
    metrics = []
    result_path = ''
    if test_kind == 'patch_test_expert':
        header = ['Five_Accuracy', 'Binary_Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
                  'Binary_Confusion_Matrix', 'Five_Confusion_Matrix']
        inves_1_metrics.insert(0, five_accracy_1)
        inves_1_metrics.insert(-1, five_confusion_matrix_1)
        inves_2_metrics.insert(0, five_accracy_2)
        inves_2_metrics.insert(-1, five_confusion_matrix_2)
        inves_3_metrics.insert(0, five_accracy_3)
        inves_3_metrics.insert(-1, five_confusion_matrix_3)
        inves_4_metrics.insert(0, five_accracy_4)
        inves_4_metrics.insert(-1, five_confusion_matrix_4)
        result_path = os.path.join('result', 'patch_test', 'investigator', 'val_result.txt')

    else:
        header = ['Binary_Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
                  'Binary_Confusion_Matrix']
        result_path = os.path.join('result', 'tiff_xiangya_test', 'investigator', 'val_result.txt')

    header = '\t'.join(header) + '\n'
    inves_1_metrics = '\t'.join(list(map(str, inves_1_metrics))) + '\n'
    inves_2_metrics = '\t'.join(list(map(str, inves_2_metrics))) + '\n'
    inves_3_metrics = '\t'.join(list(map(str, inves_3_metrics))) + '\n'
    inves_4_metrics = '\t'.join(list(map(str, inves_4_metrics))) + '\n'

    # 将header和4个investigator的指标写到文件中
    with open(result_path, 'w', encoding='utf8') as fout:
        fout.write(header)
        fout.write(inves_1_metrics)
        fout.write(inves_2_metrics)
        fout.write(inves_3_metrics)
        fout.write(inves_4_metrics)


def worker_image_tiff(args, folder_num=0):
    # model configuration
    model_name = args.model_name
    test_kind = args.test_kind
    test_mode = args.test_mode
    pre_train = args.pre_train
    # coresponded data augmentation type for various train pattern
    augmentation = args.augmentation

    num_class = args.num_class
    # input configuration, image size settings
    image_size = args.image_size
    patch_size = args.patch_size
    batch_size = args.batch_size

    # the tiff file we need to test
    tiff_file = args.tiff_file

    # MAE self supervised training configuration
    drop_path = args.drop_path
    # ensemble model params
    image_size_list = args.image_size_list
    ensemble = args.ensemble
    cross_attention = args.cross_attention
    post_merge_type = args.post_merge_type

    use_model_ema = args.use_model_ema
    num_workers = args.num_workers
    gpu = args.gpu
    checkpoint = args.checkpoint
    result_dir = args.result_dir

    test_set = TiffDataSet(args, tiff_file=tiff_file)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             pin_memory=True,
                             shuffle=False)
    net = None
    if test_mode == 'supervised':
        net = timm.create_model(get_full_model_name(model_name, image_size), pretrained=pre_train)
        net.reset_classifier(num_classes=num_class)
    elif test_mode == 'fine_tune':
        # 根据不同的预训练方式，加载不同的被微调过的模型
        # 需要根据不同的预训练方式定义不同的模型，并加载对应的权重
        if augmentation == 'dino_fine_tune_train':
            # 创建微调模型
            if model_name in vits.__dict__:
                net = vits.__dict__[model_name](patch_size=patch_size, num_classes=num_class)
            elif model_name in ['Swin-T', 'Swin-S', 'Swin-B']:
                net = timm.create_model(get_full_model_name(model_name, image_size), pretrained=pre_train)
                net.reset_classifier(num_classes=num_class)

        elif augmentation == 'mae_fine_tune_train':
            # 创建微调模型
            if not ensemble:
                net = model_finetune.__dict__[model_name](
                    img_size=image_size,
                    num_classes=num_class,
                    drop_path_rate=drop_path,
                    global_pool=False
                )
            else:
                net = model_finetune.__dict__[model_name](
                    image_size_list=image_size_list,
                    num_classes=num_class,
                    drop_path_rate=drop_path,
                    global_pool=False
                )
        else:
            print('not support fine tune model')
            sys.exit(-1)

    else:
        print('not support train mode')
        sys.exit(-1)

    model_weight_path = ''
    dir_postfix = ''
    dir_total_postfix = ''
    # 接下来需要根据模型的不同训练模态确定需要加载的模型权重的目录
    if test_mode == 'supervised':
        param_pattern = 'imagenet_pretrain' if pre_train else 'random_initial'
        if model_name in ['Swin-T', 'Swin-S', 'Swin-B']:
            dir_postfix = os.path.join(test_mode, model_name, str(image_size), param_pattern,
                                       'fold_' + str(folder_num))
        else:
            dir_postfix = os.path.join(test_mode, model_name, 'patch' + str(patch_size) + '_' + str(image_size),
                                       param_pattern, 'fold_' + str(folder_num))

    elif test_mode == 'fine_tune':
        # 这个地方的dir_postfix要去根据下面的路径进行调整
        if not ensemble:
            kind = augmentation.split('_')[0]
            dir_postfix = os.path.join('fine_tune', kind, model_name,
                                       'patch' + str(patch_size) + '_' + str(image_size), 'fold_' + str(folder_num))
        else:
            img_size_str = '_'.join(list(map(str, image_size_list)))
            dir_postfix = os.path.join('fine_tune', 'mae', model_name,
                                       'patch' + str(patch_size) + '_' + img_size_str,
                                       'fold_' + str(folder_num))

    model_weight_path = os.path.join(checkpoint, dir_postfix, 'model_best.pth')

    # 得到模型权重的路径后，加载训练好的模型权重，以便于后面的模型的预测
    model_checkpoint = torch.load(model_weight_path, map_location='cpu')
    print('loading checkpoint from {}'.format(model_weight_path))
    model_state = model_checkpoint['state_dict']
    if use_model_ema:
        model_state = model_checkpoint['ema_state_dict']
    net.load_state_dict(model_state)

    # 模型加载完训练好的权重后，将模型移动到gpu上
    device = torch.device('cuda:' + str(gpu))
    if net is not None:
        net = net.to(device)
    # 根据我们测试的模型种类和和测试的类别确定我们保存结果的文件路径
    result_folder = os.path.join(result_dir, test_kind, dir_postfix)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    print(result_folder)
    result_path = os.path.join(result_folder, 'val_result.txt')
    if not ensemble:
        valid(test_loader, net, result_path)
    else:
        valid_mae_ensemble_post_cross_attention(test_loader, net, result_path)

    dir_total_postfix = os.path.join(os.path.split(dir_postfix)[0], 'total')
    total_result_folder = os.path.join(result_dir, test_kind, dir_total_postfix)
    if not os.path.exists(total_result_folder):
        os.makedirs(total_result_folder)
    total_result_path = os.path.join(total_result_folder, 'total_val_result.txt')
    get_tiff_metrics_from_file(result_path, total_result_path, test_kind)


def get_tiff_metrics_from_file(val_result_file, total_result_file, test_kind):
    # 首先从标签文件中获得tiff真实的标签
    label_file_name = '_'.join(test_kind.split('_')[:-1]) + '.txt'
    true_label_path = os.path.join('dataset', label_file_name)
    true_label_dict = {}
    with open(true_label_path, 'r', encoding='utf8') as fin:
        for line in fin:
            image_path = line.strip('\n').split('\t')[0]
            image_label = int(line.strip('\n').split('\t')[1])
            image_label = 1 if image_label > 2 else 0
            true_label_dict[image_path] = image_label

    # 读取预测数据，从数据结果中获取模型预测tiff的标签
    # 所要计算的指标包括二分类准确率(binary class accuracy),
    # 灵敏度(sensitivity),特异性(specificity), PPV, NPV, AUC
    pred_label_dict = {}
    line_count = 0
    if os.path.exists(total_result_file):
        with open(total_result_file, 'r', encoding='utf8') as f:
            line_count = len(f.readlines())

    total_file_out = open(total_result_file, 'a', encoding='utf8')
    with open(val_result_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            img_path = line_list[0]
            print('-'*50)
            label = int(line_list[-1])
            print(label)
            print('-'*50)
            pred_label_dict[img_path] = label

    true_label_list = []
    pred_label_list = []
    print('+'*50)
    print(true_label_dict)
    print(pred_label_dict)
    print('+'*50)
    for img, label in true_label_dict.items():
        if img in pred_label_dict:
            true_label_list.append(label)
            pred_label_list.append(pred_label_dict[img])
    true_label_list = np.array(true_label_list)
    pred_label_list = np.array(pred_label_list)

    # 接下来利用二分类的混淆矩阵计算各种指标，包括灵敏度，特异性，PPV, NPV
    # metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    metrics = cal_bin_metrics(y_true=true_label_list, y_pred=pred_label_list)
    binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix = metrics[0], metrics[1], metrics[2], \
                                                                                metrics[3], metrics[4], metrics[5]

    header = ['Binary_Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV',
              'Binary_Confusion_Matrix']
    header = '\t'.join(header) + '\n'
    binary_confusion_matrix = binary_confusion_matrix.reshape(-1)
    metrics = [binary_accracy, sensitivity, specifity, ppv, npv,
               binary_confusion_matrix]
    metrics = '\t'.join(list(map(str, metrics))) + '\n'
    if line_count == 0:
        total_file_out.write(header)
    total_file_out.write(metrics)
    total_file_out.close()


def cal_bin_metrics(y_true, y_pred):
    """
    the function is used to calculate the corresponding metrics for binary classification,
    including 二分类准确率，灵敏度，特异性，PPV, NPV, AUC
    :param y_true:
    :param y_pred:
    :return:
    """
    # 首先判断一下y_true, y_pred的维度，再进行相应的操作
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    binary_accracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    # 利用sklearn的接口计算得到混淆矩阵
    binary_confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # print(y_true)
    # print(y_pred)
    # print(binary_confusion_matrix)
    # 接下来利用二分类的混淆矩阵计算各种指标，包括灵敏度，特异性，PPV, NPV
    tp = binary_confusion_matrix[1][1]
    fn = binary_confusion_matrix[1][0]
    fp = binary_confusion_matrix[0][1]
    tn = binary_confusion_matrix[0][0]
    sensitivity = tp / (tp + fn)
    specifity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    binary_confusion_matrix = binary_confusion_matrix.reshape(-1)
    metrics = [binary_accracy, sensitivity, specifity, ppv, npv, binary_confusion_matrix]
    return metrics


if __name__ == '__main__':
    arg_setting = argparse.ArgumentParser('the parameters for testing the model performance')
    arg_setting.add_argument('--model_name', type=str, default='Vit-T',
                             help='the model we used.')
    # test_mode configuration
    arg_setting.add_argument('--test_mode', type=str, default="fine_tune",
                             choices=['fine_tune'], help="test mode"
                             )
    # model parameters initialization pattern
    arg_setting.add_argument('--pre_train', action='store_true', help="weight initialized by weight pretrained from "
                                                                      "imageNet")
    # coresponded data augmentation type for various train pattern
    arg_setting.add_argument('--augmentation', type=str, default='mae_fine_tune_train',
                             choices=['val_or_test', 'mae_fine_tune_train'],
                             help="augmentation type for different training pattern")
    arg_setting.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                             help='Drop path rate (default: 0.1)')

    # emsemble mae model configuration
    arg_setting.add_argument('--ensemble', action='store_true', help="if need to ensemble the mae finetune model")
    arg_setting.add_argument('--cross_attention', action='store_true', help="if need to use the cross attention")
    arg_setting.add_argument('--post_merge_type', type=str, default='mean', choices=['mean', 'max'],
                             help="merge type for ensembling the mae finetune model")
    arg_setting.add_argument('--image_size_list', type=int, nargs='+', default=[224, 326, 512],
                             help='the multi-scale image size to form the image pyramid')

    arg_setting.add_argument('--num_class', default=5, help="class num", type=str)
    arg_setting.add_argument('--test_kind', default='cross_validation',
                             choices=['cross_validation', 'patch_test', 'tiff_refine3_test', 'tiff_xiangya_test',
                                      'tiff_huaxi_test', 'patch_test_expert', 'tiff_xiangya_expert', 'tiff_whdx_test',
                                      'tiff_huaxi_modified_test'],
                             help="the test kind", type=str)

    arg_setting.add_argument('--tiff_file', type=str, default='dataset/tiff_xiangya.txt',
                             choices=['dataset/tiff_refine3.txt', 'dataset/tiff_xiangya.txt',
                                      'dataset/tiff_huaxi.txt', 'dataset/tiff_whdx.txt',
                                      'dataset/tiff_huaxi_modified.txt'],
                             help='the tiff file we need to test')

    # input configuration, image size settings
    arg_setting.add_argument('--image_size', type=int, default=224, help='the image size input to the model')
    arg_setting.add_argument('--patch_size', type=int, default=16, help='the patch size for a divided patch')

    # MAE finetuning configuration
    arg_setting.add_argument('--use_model_ema', action='store_true', default=False)

    arg_setting.add_argument('--batch_size', type=int, default=32, help="training batch size")
    arg_setting.add_argument('--num_workers', type=int, default=1, help="data loader thread")

    arg_setting.add_argument('--gpu', type=int, default=0, help='the gpu number will be used')
    arg_setting.add_argument('--checkpoint', type=str, default='checkpoint',
                             help='the directory to save the model weights.')
    arg_setting.add_argument('--seed', default=0, type=int)
    arg_setting.add_argument('--result_dir', default='result', type=str, help='the directory of the model testing '
                                                                              'performance')
    args = arg_setting.parse_args()

    # set seed for reproduce
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    worker(args)
