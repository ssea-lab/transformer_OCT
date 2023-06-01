import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_binary_error_classes():
    with open('val_result.txt', 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            img_path = line_list[0]
            label = 0 if int(line_list[-1]) <= 2 else 1
            five_probs = list(map(float, line_list[1: -1]))
            binary_probs = [five_probs[0] + five_probs[1] + five_probs[2], five_probs[3] + five_probs[4]]
            pre_label = np.argmax(binary_probs)
            # 当模型预测错误时，需要分情况讨论，并分别创建对用的文件夹
            if pre_label != label:
                # negative -> positive
                if label == 0 and pre_label == 1:
                    create_dir('neg_pos')
                    with open('neg_pos/file.txt', 'a', encoding='utf8') as fout:
                        line = '\t'.join([img_path, str(np.argmax(five_probs))]) + '\n'
                        fout.write(line)

                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('neg_pos', img_name)
                    cv2.imwrite(img_name, img)

                # positive -> negative
                elif label == 1 and pre_label == 0:
                    create_dir('pos_neg')
                    with open('pos_neg/file.txt', 'a', encoding='utf8') as fout:
                        line = '\t'.join([img_path, str(np.argmax(five_probs))]) + '\n'
                        fout.write(line)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('pos_neg', img_name)
                    cv2.imwrite(img_name, img)

                else:
                    pass


def get_error_classes():
    with open('val_result.txt', 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            img_path = line_list[0]
            label = int(line_list[-1])
            five_probs = list(map(float, line_list[1: -1]))
            pre_label = np.argmax(five_probs)
            # 当模型预测错误时，需要分情况讨论，并分别创建对用的文件夹
            if pre_label != label:
                # MI->CY
                if label == 0 and pre_label == 1:
                    create_dir('MI_CY')
                    with open('MI_CY/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[0]), str(five_probs[1]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('MI_CY', img_name)
                    cv2.imwrite(img_name, img)

                # CY->MI
                elif label == 1 and pre_label == 0:
                    create_dir('CY_MI')
                    with open('CY_MI/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[1]), str(five_probs[0]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('CY_MI', img_name)
                    cv2.imwrite(img_name, img)

                # MI->EP
                elif label == 0 and pre_label == 2:
                    create_dir('MI_EP')
                    with open('MI_EP/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[0]), str(five_probs[2]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('MI_EP', img_name)
                    cv2.imwrite(img_name, img)

                # EP->MI
                elif label == 2 and pre_label == 0:
                    create_dir('EP_MI')
                    with open('EP_MI/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[2]), str(five_probs[0]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('EP_MI', img_name)
                    cv2.imwrite(img_name, img)

                # EP->HSIL
                elif label == 2 and pre_label == 3:
                    create_dir('EP_HSIL')
                    with open('EP_HSIL/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[2]), str(five_probs[3]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('EP_HSIL', img_name)
                    cv2.imwrite(img_name, img)

                # HSIL->MI
                elif label == 3 and pre_label == 2:
                    create_dir('HSIL_MI')
                    with open('HSIL_MI/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[3]), str(five_probs[2]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('HSIL_EP', img_name)
                    cv2.imwrite(img_name, img)

                # HSIL->EP
                elif label == 3 and pre_label == 2:
                    create_dir('HSIL_EP')
                    with open('HSIL_EP/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[3]), str(five_probs[2]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('HSIL_EP', img_name)
                    cv2.imwrite(img_name, img)

                # HSIL->CC
                elif label == 3 and pre_label == 4:
                    create_dir('HSIL_CC')
                    with open('HSIL_CC/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[3]), str(five_probs[4]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('HSIL_CC', img_name)
                    cv2.imwrite(img_name, img)

                # CC->HSIL
                elif label == 4 and pre_label == 3:
                    create_dir('CC_HSIL')
                    with open('CC_HSIL/file.txt', 'a', encoding='utf8') as fout:
                        in_top2 = label in np.argsort(np.array(five_probs))[-2:]
                        li = '\t'.join([img_path, str(five_probs[4]), str(five_probs[3]), str(in_top2)])
                        li += '\n'
                        fout.write(li)
                    img = cv2.imread(img_path, 1)
                    img_name = img_path.split('/')[-1]
                    img_name = os.path.join('CC_HSIL', img_name)
                    cv2.imwrite(img_name, img)

                else:
                    pass


def draw_box_plot():
    labels = ['MI_CY', 'CY_MI', 'MI_EP', 'EP_MI',
              'EP_HSIL', 'HSIL_EP', 'HSIL_CC', 'CC_HSIL']
    MI_and_CY_data = []
    MI_and_EP_data = []
    EP_and_HSIL_data = []
    HSIL_and_CC_data = []
    map_dict = {'MI_CY': MI_and_CY_data,
                'CY_MI': MI_and_CY_data,
                'MI_EP': MI_and_EP_data,
                'EP_MI': MI_and_EP_data,
                'EP_HSIL': EP_and_HSIL_data,
                'HSIL_EP': EP_and_HSIL_data,
                'HSIL_CC': HSIL_and_CC_data,
                'CC_HSIL': HSIL_and_CC_data,
                }
    label_names = 'MI&CY', 'MI&EP', 'EP&HSIL', 'HSIL&CC'

    for l in labels:
        with open(os.path.join(l, 'file.txt')) as fin:
            for line in fin:
                line_list = line.strip('\n').split('\t')
                prob_diff = abs(float(line_list[1]) - float(line_list[2]))
                map_dict[l].append(prob_diff)

    plt.grid(True)  # 显示网格
    box = plt.boxplot([MI_and_CY_data, MI_and_EP_data, EP_and_HSIL_data, HSIL_and_CC_data],
                      medianprops={'color': 'orange', 'linewidth': '1.0'},
                      widths=0.4,
                      # meanline=True,
                      showmeans=True,
                      # patch_artist=True,
                      boxprops={'color': 'steelblue', 'linewidth': '0.8'},
                      # meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.0'},
                      flierprops={"marker": "o", "markersize": 8, 'markeredgecolor': 'coral',
                                  'markerfacecolor': 'coral'},
                      # capprops={'color': 'black', 'linewidth': '1.0'},
                      whiskerprops={'color': 'steelblue', 'linewidth': '0.7'},
                      labels=label_names)
    # colors = ['pink', 'lightblue', 'lightgreen', 'yellow']
    # for patch, color in zip(box['boxes'], colors):
    #     patch.set_facecolor(color)

    plt.yticks(np.arange(0.0, 0.4, 0.05))
    plt.ylabel('probability  difference')
    plt.savefig('box_line.png', dpi=1200, bbox_inches='tight')
    plt.show()


def draw_pie():
    # labels = ['Neg->Pos', 'Pos->Neg']
    labels = ['Neg->Pos(EP)', 'Neg->Pos(MI)', 'Pos->Neg(HSIL)']
    # data = [34, 56]
    data = [ 25, 9, 56]
    colors = ['orange', 'orange', 'brown']
    explode = (0.0, 0.0, 0.0)
    plt.figure(figsize=(10, 8))
    patches, l_text, p_text = plt.pie(data, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.2f%%', labeldistance=1.1)
    for t in p_text:
        t.set_size(20)

    for t in l_text:
        t.set_size(20)

    plt.axis('equal')

    plt.savefig('pie.png', dpi=1200,  bbox_inches='tight')
    plt.show()


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


def cal_inves_kappa():
    inves1_label_dict = {}
    inves2_label_dict = {}
    inves3_label_dict = {}
    inves4_label_dict = {}
    model_label_dict = {}
    with open('tiff_xiangya_expert_label.txt', 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip().split('\t')
            img_name = line_list[0]
            # inv1_label = 0 if int(line_list[1]) < 3 else 1
            # inv2_label = 0 if int(line_list[2]) < 3 else 1
            # inv3_label = 0 if int(line_list[3]) < 3 else 1
            # inv4_label = 0 if int(line_list[4]) < 3 else 1
            inv1_label = int(line_list[1])
            inv2_label = int(line_list[2])
            inv3_label = int(line_list[3])
            inv4_label = int(line_list[4])
            inves1_label_dict[img_name] = inv1_label
            inves2_label_dict[img_name] = inv2_label
            inves3_label_dict[img_name] = inv3_label
            inves4_label_dict[img_name] = inv4_label
    # with open('val_result.txt', 'r', encoding='utf8') as fi:
    #     for line in fi:
    #         line_list = line.strip().split('\t')
    #         img_name = line_list[0]
    #         pre_label = np.argmax(list(map(float, line_list[1: -1])))
    #         pre_label = 0 if pre_label < 3 else 1
    #         model_label_dict[img_name] = pre_label
    with open('tiff_xiangya_predict_result.txt', 'r', encoding='utf8') as fi:
        for line in fi:
            line_list = line.strip().split('\t')
            img_name = line_list[0]
            pre_label = int(line_list[-1])
            model_label_dict[img_name] = pre_label

    dict_list = [(inves1_label_dict, inves2_label_dict), (inves1_label_dict, inves3_label_dict),
                 (inves1_label_dict, inves4_label_dict), (inves1_label_dict, model_label_dict),
                 (inves2_label_dict, inves3_label_dict), (inves2_label_dict, inves4_label_dict),
                 (inves2_label_dict, model_label_dict), (inves3_label_dict, inves4_label_dict),
                 (inves3_label_dict, model_label_dict), (inves4_label_dict, model_label_dict)]
    for inv1_dict, inv2_dict in dict_list:
        conf_matrix = get_conf_matrix(inv1_dict, inv2_dict)
        print(conf_matrix)
        K = get_kappa_coefficient(conf_matrix)
        print(K)


def test_channel():
    # img = cv2.imread('CC_HSIL/4_B1903579-2_circle_5.0x5.0_C01_S0003_0_0_0.png')
    img = cv2.imread('CC_HSIL/M0003_2019_P0000162_circle_2.0x2.0_C12_S012_2.png')
    print(img[:, :, 0])
    print(img[:, :, 1])
    print(img[:, :, 2])
    print(img[:, :, 0].shape)
    print((img[:, :, 0] == img[:, :, 1]))
    print((img[:, :, 0] == img[:, :, 2]))
    print((img[:, :, 0] == img[:, :, 1]).sum())
    print((img[:, :, 0] == img[:, :, 2]).sum())
    # with open('three_channel_2.txt', 'w', encoding='utf8') as fout:
    #     for i in img[:, :, 2]:
    #         fout.write(str(i) + '\n')


def get_chi2_test():
    def combine_arrays(arr1, arr2):
        arr1 = np.array([arr1[0][0] + arr1[1][1], arr1[0][1] + arr1[1][0]])
        arr2 = np.array([arr2[0][0] + arr2[1][1], arr2[0][1] + arr2[1][0]])
        arr = np.stack([arr1, arr2], axis=0)
        return arr

    investigator1_patch_conf_matrix = np.array([[1847, 2],
                                                [35, 1394]])
    investigator2_patch_conf_matrix = np.array([[1844, 5],
                                                [258, 1171]])
    investigator3_patch_conf_matrix = np.array([[1767, 82],
                                                [118, 1311]])
    investigator4_patch_conf_matrix = np.array([[1810, 39],
                                                [55, 1374]])
    model_patch_conf_matrix = np.array([[1815, 34],
                                        [56, 1373]])
    conf_matrix_list = [investigator1_patch_conf_matrix, investigator2_patch_conf_matrix,
                        investigator3_patch_conf_matrix,
                        investigator4_patch_conf_matrix, model_patch_conf_matrix]

    for i in range(len(conf_matrix_list)):
        for j in range(i + 1, len(conf_matrix_list)):
            print(i, j)
            table = combine_arrays(conf_matrix_list[i], conf_matrix_list[j])
            chi2, p, dof, expected = chi2_contingency(table)
            print(f"p-value:            {p}")


def write_to_csv():
    header = ['path', 'investigator1', 'investigator2', 'investigator3', 'investigator4', 'model',
              '0_sum', '1_sum']

    inves1_label_dict = {}
    inves2_label_dict = {}
    inves3_label_dict = {}
    inves4_label_dict = {}
    model_label_dict = {}
    with open('tiff_xiangya_expert_label.txt', 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip().split('\t')
            img_name = line_list[0]
            # inv1_label = 0 if int(line_list[1]) < 3 else 1
            # inv2_label = 0 if int(line_list[2]) < 3 else 1
            # inv3_label = 0 if int(line_list[3]) < 3 else 1
            # inv4_label = 0 if int(line_list[4]) < 3 else 1
            inv1_label = int(line_list[1])
            inv2_label = int(line_list[2])
            inv3_label = int(line_list[3])
            inv4_label = int(line_list[4])
            inves1_label_dict[img_name] = inv1_label
            inves2_label_dict[img_name] = inv2_label
            inves3_label_dict[img_name] = inv3_label
            inves4_label_dict[img_name] = inv4_label
    # with open('val_result.txt', 'r', encoding='utf8') as fi:
    #     for line in fi:
    #         line_list = line.strip().split('\t')
    #         img_name = line_list[0]
    #         pre_label = np.argmax(list(map(float, line_list[1: -1])))
    #         pre_label = 0 if pre_label < 3 else 1
    #         model_label_dict[img_name] = pre_label
    with open('tiff_xiangya_predict_result.txt', 'r', encoding='utf8') as fi:
        for line in fi:
            line_list = line.strip().split('\t')
            img_name = line_list[0]
            pre_label = int(line_list[-1])
            model_label_dict[img_name] = pre_label
    data = []
    for img_name, label in inves1_label_dict.items():
        i1_label = label
        i2_label = inves2_label_dict[img_name]
        i3_label = inves3_label_dict[img_name]
        i4_label = inves4_label_dict[img_name]
        model_label = model_label_dict[img_name]
        one_num = sum([i1_label, i2_label, i3_label, i4_label, model_label])
        zero_num = 5 - one_num
        data.append([img_name, i1_label, i2_label, i3_label, i4_label, model_label, zero_num, one_num])
    with open('tiff_xiangya_kappa.csv', 'w', encoding='utf8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        writer.writerows(data)


if __name__ == '__main__':
    # get_error_classes()
    get_binary_error_classes()
    # draw_box_plot()

    # draw_pie()
    # cal_inves_kappa()
    # test_channel()
    # get_chi2_test()
    # write_to_csv()
