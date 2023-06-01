from sklearn.metrics import f1_score
import numpy as np


def get_pred_label(high_risk_prob, prob_threshold, risk_count_threshold):
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
    thresh_hold = prob_threshold
    risk_count += (high_risk_prob[h_index, v_index] > thresh_hold).sum()
    risk_count += (high_risk_prob[l_h_index, l_v_index] > thresh_hold).sum()
    risk_count += (high_risk_prob[r_h_index, r_v_index] > thresh_hold).sum()
    risk_count += (high_risk_prob[u_h_index, u_v_index] > thresh_hold).sum()
    risk_count += (high_risk_prob[low_h_index, low_v_index] > thresh_hold).sum()
    risk_count_threshhold = risk_count_threshold
    if risk_count > risk_count_threshhold:
        final_predict_label = 1
    return final_predict_label


def cal_f1_score(val_file):
    pre_probs = []
    labels = []
    with open(val_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            pre_probs.append(list(map(float, line_list[1:-1])))
            labels.append(int(line_list[-1]))
    pre_probs = np.array(pre_probs)
    labels = np.array(labels)
    marco_F1 = f1_score(y_true=labels, y_pred=np.argmax(pre_probs, axis=1), average='macro')
    micro_F1 = f1_score(y_true=labels, y_pred=np.argmax(pre_probs, axis=1), average='micro')
    print(marco_F1)
    print(micro_F1)


def cal_expert_f1_score(expert_file, label_file):
    with open(expert_file, 'r', encoding='utf8') as fin:
        list_dict = [{}, {}, {}, {}]
        line_count = 0
        for line in fin:
            line_count += 1
            if line_count == 1:
                continue
            else:
                line_list = line.strip('\n').split('\t')
                # print(line_list[1:])
                img_path = line_list[0]
                label_list = list(map(int, line_list[1:]))
                for i in range(len(label_list)):
                    list_dict[i][img_path] = label_list[i]

        with open(label_file, 'r', encoding='utf8') as f:
            true_labels = []
            expert_predict_label = [[], [], [], []]
            for line in f:
                line_list = line.strip('\n').split('\t')
                image_path = line_list[0]
                label = int(line_list[1])
                label = 1 if label > 2 else 0
                for i in range(len(list_dict)):
                    if image_path in list_dict[i]:
                        expert_predict_label[i].append(list_dict[i][image_path])
                true_labels.append(label)
        # print(len(true_labels))
        true_labels = np.array(true_labels)
        for expert_label in expert_predict_label:
            # print(len(expert_label))
            expert_label = np.array(expert_label)
            marco_F1 = f1_score(y_true=true_labels, y_pred=expert_label, average='macro')
            micro_F1 = f1_score(y_true=true_labels, y_pred=expert_label, average='micro')
            print(marco_F1)
            print(micro_F1)


def cal_volume_f1_score(label_file, val_file, prob_thresh, risk_count_thresh):
    # 首先从标签文件中获得tiff真实的标签
    true_label_dict = {}
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            image_path = line.strip('\n').split('\t')[0]
            image_label = int(line.strip('\n').split('\t')[1])
            image_label = 1 if image_label > 2 else 0
            true_label_dict[image_path] = image_label

    pred_label_dict = {}
    with open(val_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            image_path = line_list[0]
            image_pred_prob = line_list[-4]
            num_patches = len(image_pred_prob.split())
            pre_prob_matrix = np.array(list(map(float, image_pred_prob.split()))).reshape(10,
                                                                                          num_patches // 10)
            predict_label = get_pred_label(pre_prob_matrix, prob_thresh, risk_count_thresh)
            pred_label_dict[image_path] = predict_label

            true_label_list = []
            pred_label_list = []
            for img, label in true_label_dict.items():
                if img in pred_label_dict:
                    true_label_list.append(label)
                    pred_label_list.append(pred_label_dict[img])
            true_label_list = np.array(true_label_list)
            pred_label_list = np.array(pred_label_list)
            marco_F1 = f1_score(y_true=true_label_list, y_pred=pred_label_list, average='macro')
            micro_F1 = f1_score(y_true=true_label_list, y_pred=pred_label_list, average='micro')
        print(marco_F1)
        print(micro_F1)


def statistic_info(file):
    with open(file, 'r', encoding='utf8') as fin:
        cls_count = {}
        for line in fin:
            line_list = line.strip('\n').split('\t')
            cls = line_list[1]
            if cls in cls_count:
                cls_count[cls] += 1
            else:
                cls_count[cls] = 1
        print(cls_count)


if __name__ == '__main__':
    # cal_f1_score('val_result.txt')
    # cal_expert_f1_score('dataset/patch_test_expert_label.txt', 'dataset/patch_test.txt')
    # cal_expert_f1_score('dataset/tiff_xiangya_expert_label.txt', 'dataset/tiff_xiangya.txt')
    # cal_volume_f1_score('dataset/tiff_refine3.txt', 'dataset/val_result.txt',
    #                     prob_thresh=0.8, risk_count_thresh=2)
    # cal_volume_f1_score('dataset/tiff_xiangya.txt', 'dataset/val_result.txt',
    #                     prob_thresh=0.73, risk_count_thresh=1)
    statistic_info('dataset/tiff_xiangya.txt')
