import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 机器学习实验三：一元线性回归实验
# 只需要这一个 main.py 文件
#
# 文件放置方式：
# 当前文件夹/
# ├── main.py
# └── ex1data1.txt
#
# 运行方式：
# python main.py
# ============================================================


# =========================
# 1. 参数配置
# =========================

DATA_PATH = "ex1data1.txt"

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1500
PRINT_EVERY = 100

OUTPUT_DIR = "outputs"


# =========================
# 2. 数据读取函数
# =========================

def load_data(data_path):
    """
    读取实验数据。

    ex1data1.txt 的数据格式是：
        x,y

    例如：
        6.1101,17.592
        5.5277,9.1302

    返回：
        x: 输入特征
        y: 真实标签
    """

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"找不到数据文件：{data_path}\n"
            f"请把 ex1data1.txt 和 main.py 放在同一个文件夹下。"
        )

    data = np.loadtxt(data_path, delimiter=",")

    x = data[:, 0]
    y = data[:, 1]

    return x, y


# =========================
# 3. 预测函数
# =========================

def predict(x, w, b):
    """
    一元线性回归模型：

        f(x) = w * x + b

    参数：
        w: 权重
        b: 偏置
    """

    return w * x + b


# =========================
# 4. 损失函数
# =========================

def compute_cost(x, y, w, b):
    """
    计算线性回归的损失函数。

    损失函数：

        J(w, b) = 1 / (2m) * sum((w * x_i + b - y_i)^2)

    这里使用 1 / (2m)，是因为求导的时候可以抵消平方项产生的 2。
    """

    m = len(x)

    y_pred = predict(x, w, b)

    cost = np.sum((y_pred - y) ** 2) / (2 * m)

    return cost


def compute_mse(y_true, y_pred):
    """
    计算均方误差 MSE。

    MSE = mean((y_pred - y_true)^2)
    """

    return np.mean((y_pred - y_true) ** 2)


# =========================
# 5. 梯度下降法
# =========================

def gradient_descent(x, y, learning_rate=0.01, num_iterations=1500):
    """
    使用梯度下降法求一元线性回归参数。

    模型：

        y_pred = w * x + b

    损失函数：

        J(w, b) = 1 / (2m) * sum((y_pred - y)^2)

    对 w 求偏导：

        dw = 1 / m * sum((y_pred - y) * x)

    对 b 求偏导：

        db = 1 / m * sum(y_pred - y)

    参数更新：

        w = w - learning_rate * dw
        b = b - learning_rate * db
    """

    m = len(x)

    w = 0.0
    b = 0.0

    cost_history = []

    for i in range(num_iterations):
        y_pred = predict(x, w, b)

        error = y_pred - y

        dw = np.sum(error * x) / m
        db = np.sum(error) / m

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % PRINT_EVERY == 0 or i == num_iterations - 1:
            print(
                f"[梯度下降] 第 {i:4d} 次迭代 | "
                f"cost = {cost:.6f} | "
                f"w = {w:.6f} | "
                f"b = {b:.6f}"
            )

    return w, b, cost_history


# =========================
# 6. 最小二乘法：一元公式形式
# =========================

def least_squares_formula(x, y):
    """
    使用一元线性回归的最小二乘法闭式解。

    模型：

        y = w * x + b

    公式：

        w = sum(y_i * (x_i - x_mean))
            /
            [sum(x_i^2) - 1/m * (sum(x_i))^2]

        b = 1/m * sum(y_i - w * x_i)

    这种方法不需要迭代，可以直接算出最优参数。
    """

    m = len(x)

    x_mean = np.mean(x)

    numerator = np.sum(y * (x - x_mean))

    denominator = np.sum(x ** 2) - (1 / m) * (np.sum(x) ** 2)

    if abs(denominator) < 1e-12:
        raise ValueError("最小二乘法失败：分母接近 0，可能所有 x 都相同。")

    w = numerator / denominator

    b = np.mean(y - w * x)

    return w, b


# =========================
# 7. 最小二乘法：矩阵形式
# =========================

def least_squares_matrix(x, y):
    """
    使用矩阵形式求最小二乘解。

    原模型：

        y = w * x + b

    写成矩阵形式：

        y = X * theta

    其中：

        X = [[1, x1],
             [1, x2],
             ...
             [1, xm]]

        theta = [b, w]^T

    闭式解：

        theta = (X^T X)^(-1) X^T y

    这里使用 np.linalg.pinv 求伪逆，
    比直接使用 np.linalg.inv 更稳定。
    """

    x_column = x.reshape(-1, 1)

    ones = np.ones((len(x), 1))

    X = np.hstack([ones, x_column])

    theta = np.linalg.pinv(X.T @ X) @ X.T @ y

    b = theta[0]
    w = theta[1]

    return w, b


# =========================
# 8. 绘制拟合结果
# =========================

def plot_regression_result(x, y, w_gd, b_gd, w_ls, b_ls):
    """
    绘制训练数据散点图和两种方法得到的拟合直线。
    """

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    x_sorted = np.sort(x)

    y_gd = predict(x_sorted, w_gd, b_gd)
    y_ls = predict(x_sorted, w_ls, b_ls)

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, marker="x", label="Training Data")

    plt.plot(x_sorted, y_gd, label="Gradient Descent")

    plt.plot(x_sorted, y_ls, linestyle="--", label="Least Squares")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Result")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "regression_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n[图片保存] 拟合结果图已保存到：{save_path}")


# =========================
# 9. 绘制损失函数下降曲线
# =========================

def plot_cost_curve(cost_history):
    """
    绘制梯度下降过程中 cost 的变化曲线。
    """

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    plt.figure(figsize=(8, 6))

    plt.plot(range(len(cost_history)), cost_history)

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Cost Curve")
    plt.grid(True)

    save_path = os.path.join(OUTPUT_DIR, "cost_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[图片保存] 损失函数曲线已保存到：{save_path}")


# =========================
# 10. 主函数
# =========================

def main():
    print("=" * 70)
    print("机器学习实验三：一元线性回归实验")
    print("=" * 70)

    # 读取数据
    x, y = load_data(DATA_PATH)

    print("\n[数据读取完成]")
    print(f"样本数量：{len(x)}")
    print(f"x 前 5 个数据：{x[:5]}")
    print(f"y 前 5 个数据：{y[:5]}")

    # --------------------------------------------------------
    # 方法一：梯度下降法
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("方法一：梯度下降法")
    print("=" * 70)

    w_gd, b_gd, cost_history = gradient_descent(
        x,
        y,
        learning_rate=LEARNING_RATE,
        num_iterations=NUM_ITERATIONS,
    )

    y_pred_gd = predict(x, w_gd, b_gd)

    cost_gd = compute_cost(x, y, w_gd, b_gd)
    mse_gd = compute_mse(y, y_pred_gd)

    print("\n[梯度下降法最终结果]")
    print(f"w = {w_gd:.6f}")
    print(f"b = {b_gd:.6f}")
    print(f"cost = {cost_gd:.6f}")
    print(f"MSE = {mse_gd:.6f}")

    # --------------------------------------------------------
    # 方法二：最小二乘法，一元公式形式
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("方法二：最小二乘法")
    print("=" * 70)

    w_ls, b_ls = least_squares_formula(x, y)

    y_pred_ls = predict(x, w_ls, b_ls)

    cost_ls = compute_cost(x, y, w_ls, b_ls)
    mse_ls = compute_mse(y, y_pred_ls)

    print("\n[最小二乘法最终结果]")
    print(f"w = {w_ls:.6f}")
    print(f"b = {b_ls:.6f}")
    print(f"cost = {cost_ls:.6f}")
    print(f"MSE = {mse_ls:.6f}")

    # --------------------------------------------------------
    # 补充：矩阵形式最小二乘法
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("补充验证：矩阵形式最小二乘法")
    print("=" * 70)

    w_mat, b_mat = least_squares_matrix(x, y)

    y_pred_mat = predict(x, w_mat, b_mat)

    cost_mat = compute_cost(x, y, w_mat, b_mat)
    mse_mat = compute_mse(y, y_pred_mat)

    print("\n[矩阵形式最小二乘法结果]")
    print(f"w = {w_mat:.6f}")
    print(f"b = {b_mat:.6f}")
    print(f"cost = {cost_mat:.6f}")
    print(f"MSE = {mse_mat:.6f}")

    # --------------------------------------------------------
    # 两种方法结果比较
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("实验结果比较")
    print("=" * 70)

    print(f"{'方法':<20}{'w':>15}{'b':>15}{'cost':>15}{'MSE':>15}")
    print("-" * 80)

    print(
        f"{'梯度下降法':<20}"
        f"{w_gd:>15.6f}"
        f"{b_gd:>15.6f}"
        f"{cost_gd:>15.6f}"
        f"{mse_gd:>15.6f}"
    )

    print(
        f"{'最小二乘法':<20}"
        f"{w_ls:>15.6f}"
        f"{b_ls:>15.6f}"
        f"{cost_ls:>15.6f}"
        f"{mse_ls:>15.6f}"
    )

    print(
        f"{'矩阵最小二乘':<20}"
        f"{w_mat:>15.6f}"
        f"{b_mat:>15.6f}"
        f"{cost_mat:>15.6f}"
        f"{mse_mat:>15.6f}"
    )

    print("\n[参数差异]")
    print(f"|w_gd - w_ls| = {abs(w_gd - w_ls):.10f}")
    print(f"|b_gd - b_ls| = {abs(b_gd - b_ls):.10f}")

    print("\n说明：")
    print("1. 最小二乘法是闭式解，可以直接得到最优参数。")
    print("2. 梯度下降法是迭代优化，迭代次数足够多时会逐渐接近最小二乘法结果。")
    print("3. 两者结果越接近，说明梯度下降收敛效果越好。")

    # --------------------------------------------------------
    # 示例预测
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("示例预测")
    print("=" * 70)

    test_values = [3.5, 7.0]

    for value in test_values:
        pred_gd = predict(value, w_gd, b_gd)
        pred_ls = predict(value, w_ls, b_ls)

        print(f"\nx = {value}")
        print(f"梯度下降法预测 y = {pred_gd:.6f}")
        print(f"最小二乘法预测 y = {pred_ls:.6f}")

    # --------------------------------------------------------
    # 绘图
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("绘制实验图像")
    print("=" * 70)

    plot_regression_result(x, y, w_gd, b_gd, w_ls, b_ls)
    plot_cost_curve(cost_history)

    print("\n实验完成。")


# =========================
# 11. 程序入口
# =========================

if __name__ == "__main__":
    main()