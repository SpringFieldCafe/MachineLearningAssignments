# -*- coding: UTF-8 -*-
import random
class VecHelper(object):
    @staticmethod
    def dot_product(left_vec, right_vec):
        result = 0.0
        for i in range(len(left_vec)):
            result += left_vec[i] * right_vec[i]
        return result
    @staticmethod
    def add_vector(vec_a, vec_b):
        new_vec = []
        for i in range(len(vec_a)):
            new_vec.append(vec_a[i] + vec_b[i])
        return new_vec
    @staticmethod
    def multiply_number(vec, number):
        new_vec = []
        for value in vec:
            new_vec.append(value * number)
        return new_vec
class SimplePerceptron(object):
    def __init__(self, feature_count, active_func):
        self.active_func = active_func
        # 初始化权重和偏置，AND 数据集有两个输入特征，所以默认会产生两个权重
        self.weight_list,self.bias_value= [0.0] * feature_count,0.0
    def __str__(self):
        """
        方便打印模型训练后的参数
        """
        info = "weights\t: {}\n".format(self.weight_list)
        info += "bias\t\t: {:.6f}\n".format(self.bias_value)
        return info
    def get_raw_score(self, input_data):
        score = VecHelper.dot_product(input_data, self.weight_list)
        score += self.bias_value
        return score

    def predict(self, input_data):
        """
        根据输入样本进行预测

        先计算线性加权和，再送入阶跃函数。
        """
        score = self.get_raw_score(input_data)
        return self.active_func(score)

    def update_by_one_sample(self, input_data, real_label, learn_rate):
        """
        使用单个样本更新参数
        感知机更新规则：error = real_label - predict_label
        weight = weight + learn_rate * error * input_data
        bias  = bias  + learn_rate * error
        """
        predict_label = self.predict(input_data)
        error = real_label - predict_label
        weight_change = VecHelper.multiply_number(
            input_data,
            learn_rate * error
        )
        self.weight_list = VecHelper.add_vector(
            self.weight_list,
            weight_change
        )
        self.bias_value += learn_rate * error
        return abs(error)

    # ==========================================================
    # 一、原始代码训练方式：顺序 SGD / 在线更新
    # ==========================================================
    def fit_original_order(self, train_x, train_y, epoch_num, learn_rate):
        """
        原始感知机代码的训练方式判断：
        原代码的特点是：
        1. 按照数据原本顺序读取样本；
        2. 每读取一个样本，就立即更新一次权重和偏置；
        3. 每个 epoch 没有随机打乱数据。
        因此，原始代码可以判断为：
        顺序随机梯度下降 / 在线更新方式。
        但是它没有 random.shuffle，
        所以不是严格意义上的“随机打乱 SGD”。
        """
        for epoch in range(epoch_num):
            total_wrong = 0
            for i in range(len(train_x)):
                total_wrong += self.update_by_one_sample(
                    train_x[i],
                    train_y[i],
                    learn_rate
                )
            print(
                "[Original Order Update] epoch = {:02d}, error = {}".format(
                    epoch + 1,
                    total_wrong
                )
            )
    # ==========================================================
    # 二、批量梯度下降 BGD
    # ==========================================================
    def fit_batch_gradient(self, train_x, train_y, epoch_num, learn_rate):
        """
        批量梯度下降法 Batch Gradient Descent

        核心思想：
        1. 每个 epoch 中，先遍历所有训练样本；
        2. 遍历过程中只累计误差和参数变化量；
        3. 全部样本遍历完之后，再统一更新一次权重和偏置。

        这体现了“遍历完所有数据再更新权重”的特点。
        """
        sample_count = len(train_x)

        for epoch in range(epoch_num):

            # 用于累计所有样本产生的权重变化
            sum_weight_change = [0.0] * len(self.weight_list)

            # 用于累计所有样本产生的偏置变化
            sum_bias_change = 0.0

            # 记录本轮训练中的错误数量
            total_wrong = 0

            for i in range(sample_count):

                input_data = train_x[i]
                real_label = train_y[i]

                predict_label = self.predict(input_data)

                error = real_label - predict_label

                total_wrong += abs(error)

                # 注意：这里暂时不更新模型参数，只是累计变化量
                for j in range(len(self.weight_list)):
                    sum_weight_change[j] += error * input_data[j]

                sum_bias_change += error

            # 对累计变化量取平均，得到整个数据集的平均更新方向
            avg_weight_change = []

            for value in sum_weight_change:
                avg_weight_change.append(value / sample_count)

            avg_bias_change = sum_bias_change / sample_count

            # 遍历完整个训练集之后，统一更新一次参数
            for j in range(len(self.weight_list)):
                self.weight_list[j] += learn_rate * avg_weight_change[j]

            self.bias_value += learn_rate * avg_bias_change

            print(
                "[BGD] epoch = {:02d}, error = {}".format(
                    epoch + 1,
                    total_wrong
                )
            )

    # ==========================================================
    # 三、随机梯度下降 SGD
    # ==========================================================
    def fit_random_gradient(self, train_x, train_y, epoch_num, learn_rate):
        """
        随机梯度下降法 Stochastic Gradient Descent

        核心思想：
        1. 每个 epoch 开始时，先打乱训练数据顺序；
        2. 每次只取一个样本；
        3. 每处理一个样本，就立即更新一次权重和偏置。

        题目要求：
        SGD 的每个 epoch 需要打乱数据顺序。
        """
        sample_count = len(train_x)

        for epoch in range(epoch_num):

            # 生成样本下标列表
            index_list = []

            for i in range(sample_count):
                index_list.append(i)

            # 每个 epoch 都随机打乱样本下标
            random.shuffle(index_list)

            total_wrong = 0

            for index in index_list:

                input_data = train_x[index]
                real_label = train_y[index]

                total_wrong += self.update_by_one_sample(
                    input_data,
                    real_label,
                    learn_rate
                )

            print(
                "[SGD] epoch = {:02d}, error = {}".format(
                    epoch + 1,
                    total_wrong
                )
            )

    # ==========================================================
    # 四、小批量梯度下降 MBGD
    # ==========================================================
    def fit_mini_batch_gradient(
            self,
            train_x,
            train_y,
            epoch_num,
            learn_rate,
            batch_num):
        """
        小批量梯度下降法 Mini-Batch Gradient Descent

        核心思想：
        1. 每个 epoch 开始时，先打乱训练数据顺序；
        2. 每次取 batch_num 个样本作为一个小批量；
        3. 在一个小批量内部累计参数变化；
        4. 每处理完一个小批量，更新一次权重和偏置。

        题目要求：
        Mini-Batch 的每个 epoch 也需要打乱数据顺序。
        """
        if batch_num <= 0:
            raise ValueError("batch_num 必须大于 0")

        sample_count = len(train_x)

        for epoch in range(epoch_num):

            # 生成样本编号
            index_list = []

            for i in range(sample_count):
                index_list.append(i)

            # 每个 epoch 都要打乱数据顺序
            random.shuffle(index_list)

            total_wrong = 0

            # 按照 batch_num 切分小批量
            start = 0

            while start < sample_count:

                end = start + batch_num

                if end > sample_count:
                    end = sample_count

                # 当前小批量中的样本编号
                batch_index_list = index_list[start:end]

                current_batch_size = len(batch_index_list)

                # 当前小批量的权重变化累计值
                batch_weight_change = [0.0] * len(self.weight_list)

                # 当前小批量的偏置变化累计值
                batch_bias_change = 0.0

                for index in batch_index_list:

                    input_data = train_x[index]
                    real_label = train_y[index]

                    predict_label = self.predict(input_data)

                    error = real_label - predict_label

                    total_wrong += abs(error)

                    # 累计当前 batch 中每个样本带来的权重变化
                    for j in range(len(self.weight_list)):
                        batch_weight_change[j] += error * input_data[j]

                    # 累计当前 batch 的偏置变化
                    batch_bias_change += error

                # 对当前 batch 的累计变化取平均
                for j in range(len(self.weight_list)):
                    batch_weight_change[j] = (
                        batch_weight_change[j] / current_batch_size
                    )

                batch_bias_change = batch_bias_change / current_batch_size

                # 每处理完一个小批量，更新一次权重和偏置
                for j in range(len(self.weight_list)):
                    self.weight_list[j] += (
                        learn_rate * batch_weight_change[j]
                    )

                self.bias_value += learn_rate * batch_bias_change

                # 进入下一个小批量
                start = end

            print(
                "[MBGD] epoch = {:02d}, error = {}".format(
                    epoch + 1,
                    total_wrong
                )
            )


def step_function(value):
    """
    阶跃激活函数

    当线性加权和大于 0 时，输出 1；
    否则输出 0。
    """
    if value > 0:
        return 1
    else:
        return 0


def create_and_dataset():
    """
    构造 AND 逻辑数据集

    AND 逻辑规则：
    1 AND 1 = 1
    0 AND 0 = 0
    1 AND 0 = 0
    0 AND 1 = 0
    """
    data_x = [
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1]
    ]

    data_y = [
        1,
        0,
        0,
        0
    ]

    return data_x, data_y


def show_result(model):
    """
    输出模型参数和预测结果
    """
    print("\n训练后的模型参数：")
    print(model)

    print("AND 逻辑测试结果：")
    print("1 AND 1 = {}".format(model.predict([1, 1])))
    print("0 AND 0 = {}".format(model.predict([0, 0])))
    print("1 AND 0 = {}".format(model.predict([1, 0])))
    print("0 AND 1 = {}".format(model.predict([0, 1])))


def show_line(title):
    """
    打印分隔线，方便区分不同实验部分
    """
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


if __name__ == "__main__":

    # 固定随机种子，使 SGD 和 MBGD 的实验结果可复现
    random.seed(2026)

    # 获取 AND 逻辑训练数据
    x_train, y_train = create_and_dataset()

    # 设置训练轮数
    max_epoch = 10

    # 设置学习率
    alpha = 0.1

    # 小批量梯度下降的 batch 大小
    # 因为 AND 数据集只有 4 个样本，所以这里设置为 2
    mini_batch_size = 2

    # ======================================================
    # 实验一：判断原始感知机代码采用的梯度下降法种类
    # ======================================================
    show_line("实验一：原始感知机代码采用的梯度下降法种类判断")

    print("判断结果：")
    print("原始感知机代码是每读取一个样本，就立即更新一次权重和偏置。")
    print("因此，它属于顺序随机梯度下降思想下的在线更新方式。")
    print("但是原始代码没有在每个 epoch 打乱数据顺序，")
    print("所以它不是严格意义上的随机打乱 SGD。")

    original_model = SimplePerceptron(2, step_function)

    original_model.fit_original_order(
        x_train,
        y_train,
        max_epoch,
        alpha
    )

    show_result(original_model)

    # ======================================================
    # 实验二：批量梯度下降法训练感知机
    # ======================================================
    show_line("实验二：批量梯度下降 BGD 训练感知机")

    bgd_model = SimplePerceptron(2, step_function)

    bgd_model.fit_batch_gradient(
        x_train,
        y_train,
        max_epoch,
        alpha
    )

    show_result(bgd_model)

    # ======================================================
    # 实验三：随机梯度下降法训练感知机
    # ======================================================
    show_line("实验三：随机梯度下降 SGD 训练感知机")

    random.seed(2026)

    sgd_model = SimplePerceptron(2, step_function)

    sgd_model.fit_random_gradient(
        x_train,
        y_train,
        max_epoch,
        alpha
    )

    show_result(sgd_model)

    # ======================================================
    # 实验四：小批量梯度下降法训练感知机
    # ======================================================
    show_line("实验四：小批量梯度下降 MBGD 训练感知机")

    random.seed(2026)

    mbgd_model = SimplePerceptron(2, step_function)

    mbgd_model.fit_mini_batch_gradient(
        x_train,
        y_train,
        max_epoch,
        alpha,
        mini_batch_size
    )

    show_result(mbgd_model)