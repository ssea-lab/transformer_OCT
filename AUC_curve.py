import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Import some data to play with
data_size = ['val', '50%_', '25%_']
mode = ['supervised', 'fine-tune']
model_name = ['lbp_resnet101', 'resnet101', 'resnet50', 'vgg19']


def read_label_pred():
    predict_file = 'val_result.txt'
    label_file = 'data_folder/test_folder.txt'
    label_dict = {}
    pred_dict = {}
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            image_path = line.strip('\n').split('\t')[0]
            label = int(line.strip('\n').split('\t')[1])
            label = 1 if label > 2 else 0
            label_dict[image_path] = label
    with open(predict_file, 'r', encoding='utf8') as fin:
        for line in fin:
            line_list = line.strip('\n').split('\t')
            image_path = line_list[0]
            five_probs = list(map(float, line_list[1:6]))
            binary_probs = five_probs[-1] + five_probs[-2]
            pred_dict[image_path] = binary_probs
    label_list = []
    pred_list = []
    for image_path, label in label_dict.items():
        if image_path in pred_dict:
            label_list.append(label)
            pred_list.append(pred_dict[image_path])
    fpr, tpr, thresholds = metrics.roc_curve(y_true=np.array(label_list), y_score=np.array(pred_list))
    return fpr, tpr


plt.figure()
lw = 2
fpr, tpr = read_label_pred()
plt.figure(figsize=(3, 3))
plt.plot(fpr, tpr, color='darkorange',
         lw=1.5, label='ROC curve (area = %0.4f)' % 0.9923)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='red', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.xticks(fontsize=8)
plt.ylim([0.0, 1.01])
plt.yticks(fontsize=8)

in1 = np.array([[1847, 2],
                [35, 1394]])
in2 = np.array([[1844, 5],
                [258, 1171]])
in3 = np.array([[1767, 82],
                [118, 1311]])
in4 = np.array([[1810, 39],
                [55, 1374]])
ina = np.array([[1817, 32],
                [116, 1313]])
plt.plot([in1[0, 1] / (in1[0, 1] + in1[0, 0]), in1[0, 1] / (in1[0, 1] + in1[0, 0])],
         [in1[1, 1] / (in1[1, 1] + in1[1, 0]), in1[1, 1] / (in1[1, 1] + in1[1, 0])],
         marker="x", label='Investigator 1', color='blue', markersize=4)
plt.plot([in2[0, 1] / (in2[0, 1] + in2[0, 0]), in2[0, 1] / (in2[0, 1] + in2[0, 0])],
         [in2[1, 1] / (in2[1, 1] + in2[1, 0]), in2[1, 1] / (in2[1, 1] + in2[1, 0])],
         marker="x", label='Investigator 2', color='green', markersize=4)
plt.plot([in3[0, 1] / (in3[0, 1] + in3[0, 0]), in3[0, 1] / (in3[0, 1] + in3[0, 0])],
         [in3[1, 1] / (in3[1, 1] + in3[1, 0]), in3[1, 1] / (in3[1, 1] + in3[1, 0])],
         marker="x", label='Investigator 3', color='pink', markersize=4)
plt.plot([in4[0, 1] / (in4[0, 1] + in4[0, 0]), in4[0, 1] / (in4[0, 1] + in4[0, 0])],
         [in4[1, 1] / (in4[1, 1] + in4[1, 0]), in4[1, 1] / (in4[1, 1] + in4[1, 0])],
         marker="x", label='Investigator 4', color='brown', markersize=4)
plt.plot([ina[0, 1] / (ina[0, 1] + ina[0, 0]), ina[0, 1] / (ina[0, 1] + ina[0, 0])],
         [ina[1, 1] / (ina[1, 1] + ina[1, 0]), ina[1, 1] / (ina[1, 1] + ina[1, 0])],
         marker="o", label='Investigator.Avg', color='darkred', markersize=5)

plt.xlabel('False Positive Rate', fontsize=8)
plt.ylabel('True Positive Rate', fontsize=8)
plt.rcParams.update({'font.size': 6})
plt.legend(loc='best')
plt.savefig('ROC_Curve.pdf', bbox_inches='tight', dpi=900)
plt.show()
