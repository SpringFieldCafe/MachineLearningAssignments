
# -*- coding: utf-8 -*-
"""
机器学习实验七&八：全连接神经网络回归实验

实验内容：
1. 生成 y = 3x^2 + 2x + 1 + noise 的合成数据
2. 划分训练集、验证集、测试集：60% / 20% / 20%
3. 手动实现一个隐藏层全连接神经网络
4. 隐藏层使用 sigmoid 激活函数
5. 手动实现前向传播、MSE损失、反向传播、SGD更新
6. 绘制训练损失、验证损失，以及测试集预测对比图

说明：
本代码没有调用 sklearn、torch、tensorflow 等机器学习框架。
神经网络的前向计算和误差反向传播均由 numpy 手写完成。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

assignment_path=Path(__file__).resolve()
assignmemt8_dir=assignment_path.resolve().parent

def make_regression_points(sample_total, random_seed):
    """
    生成实验数据。
    x 在 [-10, 10] 均匀采样；
    y = 3x^2 + 2x + 1 + eps；
    eps 服从 N(0, 0.1)。
    """
    np.random.seed(random_seed)

    x_col = np.random.uniform(-10.0, 10.0, size=(sample_total, 1))
    disturb = np.random.normal(0.0, 0.1, size=(sample_total, 1))
    y_col = 3.0 * x_col * x_col + 2.0 * x_col + 1.0 + disturb

    return x_col, y_col


def split_by_ratio(x_all, y_all, train_ratio, val_ratio, random_seed):
    """
    将数据集打乱后分为训练集、验证集、测试集。
    """
    np.random.seed(random_seed)

    total_num = x_all.shape[0]
    order = np.arange(total_num)
    np.random.shuffle(order)

    train_end = int(total_num * train_ratio)
    val_end = int(total_num * (train_ratio + val_ratio))

    train_ids = order[:train_end]
    val_ids = order[train_end:val_end]
    test_ids = order[val_end:]

    x_train = x_all[train_ids]
    y_train = y_all[train_ids]

    x_val = x_all[val_ids]
    y_val = y_all[val_ids]

    x_test = x_all[test_ids]
    y_test = y_all[test_ids]

    return x_train, y_train, x_val, y_val, x_test, y_test


def standardize_array(arr, center_value=None, scale_value=None):
    """
    标准化数据。
    训练阶段根据训练集计算均值和标准差；
    验证集、测试集复用训练集的均值和标准差。
    """
    if center_value is None:
        center_value = np.mean(arr, axis=0)

    if scale_value is None:
        scale_value = np.std(arr, axis=0)

    scale_value = np.where(scale_value == 0, 1.0, scale_value)
    new_arr = (arr - center_value) / scale_value

    return new_arr, center_value, scale_value


def sigmoid_func(z):
    """
    sigmoid 激活函数。
    使用 clip 限制输入范围，避免 exp 溢出。
    """
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_grad(sigmoid_value):
    """
    sigmoid 的导数。
    这里直接使用 sigmoid 的输出值计算导数。
    """
    return sigmoid_value * (1.0 - sigmoid_value)


def mse_value(pred, label):
    """
    均方误差。
    """
    diff = pred - label
    return np.mean(diff * diff)


def init_weight(input_count, hidden_count, output_count, random_seed):
    """
    初始化一层隐藏层网络的参数。
    """
    np.random.seed(random_seed)

    pack = dict()

    # 输入层到隐藏层
    pack["w1"] = np.random.randn(input_count, hidden_count) * 0.5
    pack["b1"] = np.zeros((1, hidden_count))

    # 隐藏层到输出层
    pack["w2"] = np.random.randn(hidden_count, output_count) * 0.5
    pack["b2"] = np.zeros((1, output_count))

    return pack


def forward_compute(x_batch, param_box):
    """
    前向传播：
    输入 -> 隐藏层线性变换 -> sigmoid -> 输出层线性变换。
    """
    z1 = np.dot(x_batch, param_box["w1"]) + param_box["b1"]
    a1 = sigmoid_func(z1)
    z2 = np.dot(a1, param_box["w2"]) + param_box["b2"]

    cache = dict()
    cache["x"] = x_batch
    cache["z1"] = z1
    cache["a1"] = a1
    cache["out"] = z2

    return z2, cache


def backward_compute(y_batch, param_box, cache):
    """
    反向传播。
    损失函数为 MSE = mean((pred - y)^2)。
    """
    x_batch = cache["x"]
    a1 = cache["a1"]
    pred = cache["out"]

    batch_size = x_batch.shape[0]

    # 输出层梯度
    d_out = 2.0 * (pred - y_batch) / batch_size
    dw2 = np.dot(a1.T, d_out)
    db2 = np.sum(d_out, axis=0, keepdims=True)

    # 传回隐藏层
    d_a1 = np.dot(d_out, param_box["w2"].T)
    d_z1 = d_a1 * sigmoid_grad(a1)

    dw1 = np.dot(x_batch.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    grad_box = dict()
    grad_box["w1"] = dw1
    grad_box["b1"] = db1
    grad_box["w2"] = dw2
    grad_box["b2"] = db2

    return grad_box


def update_by_sgd(param_box, grad_box, step_size):
    """
    SGD 参数更新。
    """
    param_box["w1"] = param_box["w1"] - step_size * grad_box["w1"]
    param_box["b1"] = param_box["b1"] - step_size * grad_box["b1"]

    param_box["w2"] = param_box["w2"] - step_size * grad_box["w2"]
    param_box["b2"] = param_box["b2"] - step_size * grad_box["b2"]

    return param_box


def train_network(x_train, y_train, x_val, y_val, hidden_count, learn_rate, epoch_count, batch_count, random_seed):
    """
    训练网络，并记录训练集和验证集的 MSE。
    """
    param_box = init_weight(1, hidden_count, 1, random_seed)

    train_loss_list = []
    val_loss_list = []

    sample_num = x_train.shape[0]

    for epoch in range(epoch_count):
        idx = np.arange(sample_num)
        np.random.shuffle(idx)

        # 小批量 SGD
        begin = 0
        while begin < sample_num:
            end = begin + batch_count
            batch_ids = idx[begin:end]

            xb = x_train[batch_ids]
            yb = y_train[batch_ids]

            _, mid_cache = forward_compute(xb, param_box)
            grad_pack = backward_compute(yb, param_box, mid_cache)
            param_box = update_by_sgd(param_box, grad_pack, learn_rate)

            begin = end

        # 每个 epoch 记录一次完整训练集和验证集损失
        train_pred, _ = forward_compute(x_train, param_box)
        val_pred, _ = forward_compute(x_val, param_box)

        train_mse = mse_value(train_pred, y_train)
        val_mse = mse_value(val_pred, y_val)

        train_loss_list.append(train_mse)
        val_loss_list.append(val_mse)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print("epoch:", epoch + 1, "train_mse:", round(train_mse, 6), "val_mse:", round(val_mse, 6))

    return param_box, train_loss_list, val_loss_list


def predict_value(x_in, param_box):
    """
    网络预测。
    """
    pred, _ = forward_compute(x_in, param_box)
    return pred


def draw_loss(train_list, val_list, save_path):
    """
    绘制训练集和验证集损失曲线。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_list, label="train mse")
    plt.plot(val_list, label="validation mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()


def draw_compare(x_test_raw, y_test_raw, y_pred_raw, save_path):
    """
    绘制测试集真实值与预测值对比。
    为了图像更清楚，这里按 x 从小到大排序后画图。
    """
    order = np.argsort(x_test_raw.reshape(-1))
    xs = x_test_raw[order].reshape(-1)
    ys = y_test_raw[order].reshape(-1)
    ps = y_pred_raw[order].reshape(-1)

    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, s=16, label="real value")
    plt.plot(xs, ps, label="predicted value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Prediction Result on Test Set")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()


def save_result_csv(x_test_raw, y_test_raw, y_pred_raw, save_path):
    """
    保存测试集预测结果，方便写实验报告。
    """
    result = pd.DataFrame()
    result["x"] = x_test_raw.reshape(-1)
    result["true_y"] = y_test_raw.reshape(-1)
    result["predict_y"] = y_pred_raw.reshape(-1)
    result["abs_error"] = np.abs(result["true_y"] - result["predict_y"])
    result.to_csv(save_path, index=False, encoding="utf-8-sig")


def main():
    # ========= 1. 超参数设置 =========
    sample_total = 1000
    hidden_count = 24
    epoch_count = 5000
    learn_rate = 0.03
    batch_count = 32
    random_seed = 7

    # ========= 2. 生成数据 =========
    x_all, y_all = make_regression_points(sample_total, random_seed)

    # ========= 3. 划分数据集 =========
    x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = split_by_ratio(
        x_all, y_all, 0.6, 0.2, random_seed + 1
    )

    # ========= 4. 标准化 =========
    # 只用训练集统计量，避免测试集信息泄漏。
    x_train, x_mean, x_std = standardize_array(x_train_raw)
    x_val, _, _ = standardize_array(x_val_raw, x_mean, x_std)
    x_test, _, _ = standardize_array(x_test_raw, x_mean, x_std)

    y_train, y_mean, y_std = standardize_array(y_train_raw)
    y_val, _, _ = standardize_array(y_val_raw, y_mean, y_std)
    y_test, _, _ = standardize_array(y_test_raw, y_mean, y_std)

    print("训练集数量:", x_train.shape[0])
    print("验证集数量:", x_val.shape[0])
    print("测试集数量:", x_test.shape[0])

    # ========= 5. 训练模型 =========
    param_box, train_loss_list, val_loss_list = train_network(
        x_train, y_train, x_val, y_val,
        hidden_count, learn_rate, epoch_count, batch_count, random_seed + 2
    )

    # ========= 6. 测试集评估 =========
    y_pred_std = predict_value(x_test, param_box)

    # 将标准化后的预测值还原到原始 y 尺度
    y_pred_raw = y_pred_std * y_std + y_mean

    test_mse_raw = mse_value(y_pred_raw, y_test_raw)
    print("测试集原始尺度 MSE:", test_mse_raw)

    # ========= 7. 保存结果和图像 =========
    save_result_csv(x_test_raw, y_test_raw, y_pred_raw, str(assignmemt8_dir/"test_prediction_result.csv"))
    draw_loss(train_loss_list, val_loss_list, str(assignmemt8_dir/"loss_curve.png"))
    draw_compare(x_test_raw, y_test_raw, y_pred_raw, str(assignmemt8_dir/"test_prediction_compare.png"))


if __name__ == "__main__":
    main()
