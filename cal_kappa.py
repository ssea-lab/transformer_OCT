import numpy as np


def get_kappa_coefficient(conf_matrix):
    """
    利用预测的混淆矩阵计算kappa系数
        N  P
    N
    P
    :param conf_matrix:
    :return:
    """
    total = conf_matrix.sum()
    OA = (conf_matrix[0][0] + conf_matrix[1][1]) / total
    AC = (conf_matrix[0][1] + conf_matrix[1][1]) / total * (conf_matrix[1][0] + conf_matrix[1][1]) / total \
         + (conf_matrix[0][0] + conf_matrix[1][0]) / total * (conf_matrix[0][0] + conf_matrix[0][1]) / total
    K = (OA - AC) / (1 - AC)
    return K


def get_conf_matrix(inves1_label_dict, inves2_label_dict):
    conf_matrix = np.array([[0, 0],
                            [0, 0]])
    for img_path, label1 in inves1_label_dict.items():
        label2 = inves2_label_dict[img_path]
        conf_matrix[label1][label2] += 1

    return conf_matrix


def cal_inves_kappa(file):
    inves1_label_dict = {}
    inves2_label_dict = {}
    inves3_label_dict = {}
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip().split('\t')
            img_name = line_list[0]
            inv1_label = int(line_list[1])
            inv2_label = int(line_list[2])
            inv3_label = int(line_list[3])
            inves1_label_dict[img_name] = inv1_label
            inves2_label_dict[img_name] = inv2_label
            inves3_label_dict[img_name] = inv3_label

    dict_list = [(inves1_label_dict, inves2_label_dict), (inves1_label_dict, inves3_label_dict),
                 (inves2_label_dict, inves3_label_dict)]
    for inv1_dict, inv2_dict in dict_list:
        conf_matrix = get_conf_matrix(inv1_dict, inv2_dict)
        print(conf_matrix)
        K = get_kappa_coefficient(conf_matrix)
        print(K)


if __name__ == '__main__':
    pass
