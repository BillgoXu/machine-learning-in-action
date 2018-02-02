from math import log


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vector in data_set:
        current_label = feat_vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def create_data_set():
    data_set = [[1, 1, '是'],
                [1, 0, '否'],
                [0, 1, '否'],
                [0, 1, '否'],
                [1, 1, '是'],
                ]
    data_labels = ['无需浮上水面', '脚蹼']
    return data_set, data_labels


def split_data_set(data_set, axis, value):
    """
    :param data_set: 待划分数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    ret_data_set = []
    for feat_vector in data_set:
        if feat_vector[axis] == value:
            reduced_feat_vendor = feat_vector[:axis]
            reduced_feat_vendor.extend(feat_vector[axis + 1:])
            ret_data_set.append(reduced_feat_vendor)
    return ret_data_set


my_data, labels = create_data_set()
print(my_data)
print(split_data_set(my_data, 0, 1))

