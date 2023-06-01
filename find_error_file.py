import os
import numpy as np
from find_threshold import get_pred_label


def find_err_file(prob_threshold, risk_count_threshold, model_predict_file, label_file_name):
    """
     this function is used to try different thresholds to get results.
     :param prob_threshold:
     :param risk_count_threshold:
     :param model_predict_file:
     :return:

     """
    # 首先从标签文件中获得tiff真实的标签
    true_label_path = os.path.join('dataset', label_file_name)
    true_label_dict = {}
    with open(true_label_path, 'r', encoding='utf8') as fin:
        for line in fin:
            image_path = line.strip('\n').split('\t')[0]
            image_label = int(line.strip('\n').split('\t')[1])
            # image_label = 1 if image_label > 2 else 0
            true_label_dict[image_path] = image_label

    pred_label_dict = {}
    with open(model_predict_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            image_path = line_list[0]
            image_pred_prob = line_list[-4]
            num_patches = len(image_pred_prob.split())
            pre_prob_matrix = np.array(list(map(float, image_pred_prob.split()))).reshape(10,
                                                                                          num_patches // 10)
            predict_label = get_pred_label(pre_prob_matrix, prob_threshold, risk_count_threshold)
            pred_label_dict[image_path] = predict_label

    error_file = []
    for img_path, label in true_label_dict.items():
        pre_label = pred_label_dict[img_path]
        if pre_label != label:
            error_file.append(img_path)
    with open('error_file05_1.txt', 'w', encoding='utf8') as fout:
        for img_pa in error_file:
            fout.write(img_pa + '\n')
    print(error_file)


if __name__ == '__main__':
    prob_threshold = 0.5
    risk_count_threshold = 1
    model_predict_file = 'result/tiff_huaxi_test/fine_tune/mae/vit_base_patch16/patch16_326/fold_3/val_result.txt'
    label_file_name = 'tiff_huaxi.txt'
    find_err_file(prob_threshold, risk_count_threshold, model_predict_file, label_file_name)
