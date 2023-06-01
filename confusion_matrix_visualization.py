import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import numpy as np
import os
from matplotlib import ticker
from sklearn import metrics

mode = ['val', 'fine-tune_val']
model_name = ['lbp_resnet101', 'resnet101']
classes_name = ['MI', 'CY', 'EP', 'HSIL', 'CC']
bin_classes_name = ['N', 'P']
title_names = ['model', 'investigator1', 'investigator2', 'investigator3', 'investigator4']


def show_confMat(confusion_mat, label_name):
    """
    this function is used to visualize the confusion matrix
    :param confusion_mat: the confusion matrix
    :param label_name: 'a'
    :return:
    """
    # 归一化
    fig, ax = plt.subplots()
    plt.title(label_name, y=-0.2, fontweight='black', fontdict={'fontsize': 14})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # 根据label_name判断class_name
    class_name = classes_name if label_name <= 'e' else bin_classes_name
    confusion_mat_N = np.zeros((confusion_mat.shape[0], confusion_mat.shape[1]))
    for i in range(len(class_name)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    color = 'Blues' if label_name in ['a', 'f'] else 'Reds'
    cmap = plt.cm.get_cmap(color)  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    # 只有label name 为 'a', 'e', 'f', 'j'时才进行colorbar的绘制
    if label_name in ['a', 'e', 'f', 'j']:
        cb1 = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
        cb1.locator = tick_locator
        # cb1.set_ticks([0.0, 0.2, 0.4])
        cb1.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        cb1.ax.tick_params(labelsize=14)
        cb1.update_ticks()

    # 设置文字
    xlocations = np.array(range(len(class_name)))
    plt.xticks(xlocations, class_name, fontsize=14)
    plt.yticks(xlocations, class_name, fontsize=14)
    if label_name >= 'f':
        plt.xlabel('Prediction', fontsize=14, labelpad=0)
    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            if confusion_mat_N[i][j] > 0.2:
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='white', fontsize=16)
            else:
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='black', fontsize=16)
    # 保存
    # plt.show()
    plt.savefig(os.path.join('confusion_matrix_image', label_name + '.eps'), bbox_inches='tight', dpi=1200)
    plt.close()


# def generate_confusion_matrix(cofusion_matrix_list, name_list, disease_name_list, title_list):
#     # 首先根据混淆矩阵的个数确定子图的个数及排列
#     total_plots = len(cofusion_matrix_list)
#     fig, axes = plt.subplots(2, total_plots // 2, figsize=(15, 8))
#     print(type(axes))
#     print(type(axes[0][0]))
#     # 调节子图间的宽度,以留出放colorbar的空间.
#     plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)
#     # fig.subplots_adjust(wspace=1000)
#     # fig.subplots_adjust(hspace=200)
#
#     # 定义cmap用于进行上色
#     cmap1 = plt.cm.get_cmap('Blues')
#     cmap2 = plt.cm.get_cmap('Reds')
#     # 对混淆矩阵进行正则化，将数值缩放到0-1之间
#     cofusion_matrix_list_Norm = []
#     for confusion_mat in cofusion_matrix_list:
#         confusion_mat_N = np.zeros((confusion_mat.shape[0], confusion_mat.shape[1]))
#         for i in range(confusion_mat.shape[0]):
#             confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()
#         cofusion_matrix_list_Norm.append(confusion_mat_N)
#     # 利用axes对每个正则化后的混淆矩阵进行展示
#     for i in range(len(cofusion_matrix_list_Norm)):
#         h_idx = i // 5
#         v_idx = i % 5
#         cmap = cmap1 if v_idx == 0 else cmap2
#         print(i)
#         print(h_idx)
#         print(v_idx)
#         sub_polt = axes[h_idx][v_idx].imshow(cofusion_matrix_list_Norm[i], cmap=cmap)
#
#         # # 将每个子图的坐标刻度设为不可见
#         axes[h_idx][v_idx].spines['top'].set_visible(False)
#         axes[h_idx][v_idx].spines['right'].set_visible(False)
#         axes[h_idx][v_idx].spines['bottom'].set_visible(False)
#         axes[h_idx][v_idx].spines['left'].set_visible(False)
#         # 设置文字
#         xlocations = np.array(range(len(disease_name_list)))
#         print(disease_name_list)
#         print(xlocations)
#         print(type(axes[h_idx][v_idx]))
#
#         axes[h_idx][v_idx].set_xticks(ticks=xlocations)
#         axes[h_idx][v_idx].set_xticklabels(labels=disease_name_list)
#         axes[h_idx][v_idx].set_yticks(ticks=xlocations)
#         axes[h_idx][v_idx].set_yticklabels(labels=disease_name_list)
#         if i >= 5:
#             axes[h_idx][v_idx].set_xlabel('Prediction')
#         else:
#             axes[h_idx][v_idx].set_title(title_list[i])
#
#         # 打印数字
#         confusion_mat_Norm = cofusion_matrix_list_Norm[i]
#         confusion_mat = cofusion_matrix_list[i]
#         for j in range(confusion_mat_Norm.shape[0]):
#             for k in range(confusion_mat_Norm.shape[1]):
#                 if confusion_mat_Norm[j][k] > 0.2:
#                     axes[h_idx][v_idx].text(x=k, y=j, s=int(confusion_mat[j, k]), va='center', ha='center', color='white', fontsize=10)
#                 else:
#                     axes[h_idx][v_idx].text(x=k, y=j, s=int(confusion_mat[j, k]), va='center', ha='center', color='black', fontsize=10)
#
#         def add_right_cax(ax, pad, width):
#             """
#             在一个ax右边追加与之等高的cax.
#             pad是cax与ax的间距,width是cax的宽度.
#             """
#             axpos = ax.get_position()
#             caxpos = mtransforms.Bbox.from_extents(
#                 axpos.x1 + pad,
#                 axpos.y0,
#                 axpos.x1 + pad + width,
#                 axpos.y1
#             )
#             cax = ax.figure.add_axes(caxpos)
#             return cax
#
#         if i in [0, 4, 5, 9]:
#             cax = add_right_cax(axes[h_idx][v_idx], pad=0.00002, width=0.02)
#             color_bar = fig.colorbar(sub_polt, cax=cax)
#             tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数
#             color_bar.locator = tick_locator
#             color_bar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
#             color_bar.update_ticks()
#     plt.tight_layout(pad=1.5)
#     plt.show()


if __name__ == '__main__':

    model_5_conf_matrix = np.array([[670, 30, 9, 17, 0],
                                    [46, 753, 1, 0, 0],
                                    [13, 4, 276, 30, 0],
                                    [17, 0, 25, 1228, 10],
                                    [0, 0, 0, 15, 134]])
    investigator1_5_conf_matrix = np.array([[721, 4, 0, 1, 0],
                                            [1, 799, 0, 0, 0],
                                            [4, 0, 318, 1, 0],
                                            [19, 1, 4, 1164, 82],
                                            [1, 0, 0, 148, 0]])
    investigator2_5_conf_matrix = np.array([[712, 9, 2, 3, 0],
                                            [11, 789, 0, 0, 0],
                                            [20, 1, 300, 2, 0],
                                            [234, 0, 20, 1026, 0],
                                            [4, 0, 0, 145, 0]])
    investigator3_5_conf_matrix = np.array([[677, 0, 0, 49, 0],
                                            [30, 768, 0, 2, 0],
                                            [13, 0, 279, 31, 0],
                                            [107, 0, 5, 1168, 0],
                                            [6, 0, 0, 143, 0]])
    investigator4_5_conf_matrix = np.array([[546, 154, 3, 23, 0],
                                            [0, 800, 0, 0, 0],
                                            [0, 0, 307, 16, 0],
                                            [14, 10, 31, 1215, 10],
                                            [0, 0, 0, 134, 15]])

    model_2_conf_matrix = np.array([[1815, 34],
                                    [56, 1373]])
    investigator1_2_conf_matrix = np.array([[1847, 2],
                                            [35, 1394]])
    investigator2_2_conf_matrix = np.array([[1844, 5],
                                            [258, 1171]])
    investigator3_2_conf_matrix = np.array([[1767, 82],
                                            [118, 1311]])
    investigator4_2_conf_matrix = np.array([[1810, 39],
                                            [55, 1374]])
    conf_matrix_list = [model_5_conf_matrix, investigator1_5_conf_matrix, investigator2_5_conf_matrix,
                        investigator3_5_conf_matrix, investigator4_5_conf_matrix,
                        model_2_conf_matrix, investigator1_2_conf_matrix, investigator2_2_conf_matrix,
                        investigator3_2_conf_matrix, investigator4_2_conf_matrix]
    name_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    for conf_matrix, label_name in zip(conf_matrix_list, name_list):
        show_confMat(confusion_mat=conf_matrix, label_name=label_name)
