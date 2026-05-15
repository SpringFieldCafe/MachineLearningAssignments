# gd_all.py
# 机器学习实验五：线性回归 + 三种梯度下降法
# 包含：批量梯度下降 BGD、随机梯度下降 SGD、小批量梯度下降 Mini-Batch GD

import numpy as np
import random


# =========================
# 1. 构造实验数据
# =========================
# 这里模拟函数：y = theta1 * x1 + theta2 * x2
# 根据数据可以看出，真实参数大约为 theta = [3, 4]
input_x = np.array([
    [1, 4],
    [2, 5],
    [5, 1],
    [4, 2]
], dtype=float)

y = np.array([19, 26, 19, 20], dtype=float)


# =========================
# 2. 公共函数
# =========================
def predict(x, theta):
    # 线性回归预测函数：h(x) = theta1*x1 + theta2*x2
    return np.dot(x, theta)


def compute_loss(x, y, theta):
    # 损失函数：J(theta) = 1 / (2m) * sum((h(x)-y)^2)
    m = len(y)
    pred_y = predict(x, theta)
    loss = np.sum((pred_y - y) ** 2) / (2 * m)
    return loss


# =========================
# 3. 批量梯度下降法 BGD
# =========================
def batch_gradient_descent(x, y, learning_rate=0.001, eps=0.0001, max_iters=10000):
    # 批量梯度下降：每次使用全部样本计算一次总梯度，然后更新参数
    m, n = x.shape
    theta = np.ones(n)
    iter_count = 0

    while iter_count < max_iters:
        pred_y = predict(x, theta)
        error = pred_y - y

        # 关键：BGD 使用全部样本一起计算梯度
        gradient = np.dot(x.T, error) / m

        # 参数更新公式：theta = theta - learning_rate * gradient
        theta = theta - learning_rate * gradient

        loss = compute_loss(x, y, theta)
        iter_count += 1

        if iter_count <= 5 or iter_count % 100 == 0:
            print("BGD iters:", iter_count, "loss:", loss)

        if loss < eps:
            break

    return theta, loss, iter_count


# =========================
# 4. 随机梯度下降法 SGD
# =========================
def stochastic_gradient_descent(x, y, learning_rate=0.001, eps=0.0001, max_iters=10000):
    # 随机梯度下降：每次只使用一个样本计算梯度，并立即更新参数
    m, n = x.shape
    theta = np.ones(n)
    iter_count = 0

    while iter_count < max_iters:
        index_list = list(range(m))

        # 关键：SGD 每一轮通常要随机打乱样本顺序
        random.shuffle(index_list)

        for i in index_list:
            pred_y = predict(x[i], theta)
            error = pred_y - y[i]

            # 关键：SGD 用单个样本的误差更新 theta
            gradient = error * x[i]

            # 每看一个样本就更新一次参数
            theta = theta - learning_rate * gradient

        loss = compute_loss(x, y, theta)
        iter_count += 1

        if iter_count <= 5 or iter_count % 100 == 0:
            print("SGD iters:", iter_count, "loss:", loss)

        if loss < eps:
            break

    return theta, loss, iter_count


# =========================
# 5. 小批量梯度下降法 Mini-Batch GD
# =========================
def mini_batch_gradient_descent(x, y, batch_size=2, learning_rate=0.001, eps=0.0001, max_iters=10000):
    # 小批量梯度下降：每次使用一小批样本计算梯度，然后更新参数
    m, n = x.shape
    theta = np.ones(n)
    iter_count = 0

    while iter_count < max_iters:
        index_list = list(range(m))

        # 关键：每一轮训练前随机打乱数据
        random.shuffle(index_list)

        x_shuffle = x[index_list]
        y_shuffle = y[index_list]

        # 关键：每次取 batch_size 个样本作为一个小批量
        for start in range(0, m, batch_size):
            end = start + batch_size
            x_batch = x_shuffle[start:end]
            y_batch = y_shuffle[start:end]

            pred_y = predict(x_batch, theta)
            error = pred_y - y_batch

            # 关键：Mini-Batch 用一小批样本计算平均梯度
            gradient = np.dot(x_batch.T, error) / len(y_batch)

            # 每处理一个小批量就更新一次参数
            theta = theta - learning_rate * gradient

        loss = compute_loss(x, y, theta)
        iter_count += 1

        if iter_count <= 5 or iter_count % 100 == 0:
            print("Mini-Batch GD iters:", iter_count, "loss:", loss)

        if loss < eps:
            break

    return theta, loss, iter_count


# =========================
# 6. 主程序入口
# =========================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    learning_rate = 0.001
    eps = 0.0001
    max_iters = 10000

    print("==============================")
    print("批量梯度下降法 BGD")
    print("==============================")
    bgd_theta, bgd_loss, bgd_iters = batch_gradient_descent(
        input_x,
        y,
        learning_rate=learning_rate,
        eps=eps,
        max_iters=max_iters
    )
    print("BGD theta:", bgd_theta)
    print("BGD final loss:", bgd_loss)
    print("BGD iters:", bgd_iters)

    print("\n==============================")
    print("随机梯度下降法 SGD")
    print("==============================")
    sgd_theta, sgd_loss, sgd_iters = stochastic_gradient_descent(
        input_x,
        y,
        learning_rate=learning_rate,
        eps=eps,
        max_iters=max_iters
    )
    print("SGD theta:", sgd_theta)
    print("SGD final loss:", sgd_loss)
    print("SGD iters:", sgd_iters)

    print("\n==============================")
    print("小批量梯度下降法 Mini-Batch GD")
    print("==============================")
    mini_theta, mini_loss, mini_iters = mini_batch_gradient_descent(
        input_x,
        y,
        batch_size=2,
        learning_rate=learning_rate,
        eps=eps,
        max_iters=max_iters
    )
    print("Mini-Batch GD theta:", mini_theta)
    print("Mini-Batch GD final loss:", mini_loss)
    print("Mini-Batch GD iters:", mini_iters)