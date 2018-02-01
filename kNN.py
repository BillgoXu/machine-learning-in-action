"""
k临近算法
使用欧式距离计算公式计算向量点距离
对未知类别属性的数据集中的每个点依次执行以下操作:
1.计算已知类别数据集中每个点与当前点的距离
2.按照距离递增一次排序
3.选取与当前点最小的k个点
4.确定前k个点所在类别的出现频率
5.返回前k个点出现频率最高的类别作为当前点的预测分类
"""
from operator import itemgetter
from os import listdir

from numpy import *


def classify0(in_x, data_set, labels, k):
    """
    :param in_x: 输入向量
    :param data_set: 训练样本集
    :param labels: 标签向量集 元素数目和dataSet行数相同
    :param k: 最近邻居的数目
    :return:
    """
    # 计算距离
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 选择距离最小的k个点
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1

    # 排序
    sorted_class_count = sorted(class_count.items(), key=itemgetter(1), reverse=True)  # c
    return sorted_class_count[0][0]


def file_to_matrix(filename, ):
    fp = open(filename)
    d = 3  # 维度
    array_o_lines = fp.readlines()
    number_of_lines = len(array_o_lines)
    returns_mat = zeros((number_of_lines, d))  # 根据行数创建0填充的矩阵
    class_label_vector = []
    index = 0
    for line in array_o_lines:
        line = line.strip()  # 截取掉回车字符
        list_from_line = line.split('\t')  # 将整行数据通过tab字符分割成一个元素列表
        try:
            returns_mat[index, :] = list_from_line[0:d]  # 将特征值储存到特征矩阵
        except ValueError as error:
            print(str(index + 1) + "行出现问题!" + str(error))
            print(line)
            print(list_from_line)
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return returns_mat, class_label_vector


def auto_norm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file_to_matrix('data_set/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 3)
        print("分类器得到的结果是:%d,正确结果是:%d" % (classifier_result, dating_labels[i]))
        if classifier_result != dating_labels[i]:
            error_count += 1
    print("错误率:%f" % (error_count / float(num_test_vecs)))


def classfiy_person():
    result_list = ['not at all', 'in small does', 'in large does']
    percent_tats = float(input('电子游戏占时间比:'))  # c
    ff_miles = float(input('飞行常客里程数:'))
    ice_cream = float(input('每周消费冰淇淋(升):'))
    dating_data_mat, dating_labels = file_to_matrix('data_set/datingTestSet2.txt')
    norm_mat_set, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classify_result = classify0(in_arr, norm_mat_set, dating_labels, 3)
    print("预测结果:" + str(result_list[classify_result]))


def img_to_ventor(filename):
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):  # 行
        line_str = fr.readline()
        for j in range(32):
            return_vect[0, 32 * i + j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    hw_labels = []
    training_dir = 'training_digits'
    test_dir = 'test_digits'
    k = 3
    training_file_list = listdir(training_dir)
    n_training = len(training_file_list)
    training_mat = zeros((n_training, 1024))
    for i in range(n_training):
        file_name_str = training_file_list[i]
        class_num_str = (file_name_str.split('.')[0]).split('_')[0]
        hw_labels.append(class_num_str)
        training_mat[i, :] = img_to_ventor(training_dir + '/' + file_name_str)
    test_file_list = listdir(test_dir)
    n_test = len(test_file_list)
    error_count = 0.0
    for i in range(n_test):
        file_name_str = test_file_list[i]
        class_num_str = (file_name_str.split('.')[0]).split('_')[0]
        ventor_under_test = img_to_ventor(test_dir + '/' + file_name_str)
        classifier_result = classify0(ventor_under_test, training_mat, hw_labels, k)
        # print('分类器返回结果是:%s,正确结果是:%s'% (classifier_result, class_num_str))
        if (classifier_result != class_num_str):
            error_count += 1.0
    print("分类器错误数目:%d, 错误率:%f"%(error_count, error_count/float(n_test)))

# classfiy_person()
handwriting_class_test()
print()
