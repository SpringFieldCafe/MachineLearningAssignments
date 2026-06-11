#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os  # 使用标准库 os 处理路径和环境信息。
import sys  # 使用标准库 sys 进行 stdout 快速输出和命令行参数读取。
import math  # 使用标准库 math 计算指数、对数、开方等数学函数。
import gzip  # 使用标准库 gzip 兼容未解压的 MNIST .gz 文件。
import time  # 使用标准库 time 记录程序运行耗时。
import struct  # 使用标准库 struct 按 IDX 二进制格式解析 MNIST 文件头。
import random  # 使用标准库 random 进行权重初始化和样本打乱。
import matplotlib  # 只额外使用 matplotlib 作为画图库，不使用 numpy、pandas、sklearn、torch。
matplotlib.use("Agg")  # 使用无界面后端保存图片，避免服务器或命令行环境没有图形界面时报错。
import matplotlib.pyplot as plt  # 只导入 matplotlib.pyplot 负责生成作业需要的 PNG 可视化图像。
from pathlib import Path  # 使用标准库 pathlib 处理跨平台文件路径。


# ========================== 基础输出与参数区 ========================== # 本脚本核心算法和数据读取只依赖 Python 标准库，图片可视化只额外使用 matplotlib。


def out(text=""):  # 定义统一 stdout 输出函数，避免频繁使用 print 的额外包装。
    sys.stdout.write(str(text) + "\n")  # 直接写入标准输出，便于在命令行中快速显示进度。
    sys.stdout.flush()  # 每次输出后立即刷新，防止训练时长时间无显示。


def parse_args():  # 定义一个极简命令行参数解析函数，避免额外依赖第三方库。
    args = {}  # 创建参数字典保存配置项。
    args["data_dir"] = str(Path(__file__).resolve().parent / "src")  # 默认在代码同目录下的 src 文件夹寻找四个 MNIST 解压文件或 .gz 文件。
    args["out_dir"] = str(Path(__file__).resolve().parent / "res")  # 默认把可视化、混淆矩阵、汇总结果写到代码同目录下的 res 文件夹。
    args["seed"] = 7  # 默认随机种子，保证每次实验结果尽量可复现。
    args["train_limit"] = 5000  # 默认只取 5000 张训练图以便纯 Python 能较快跑完，设为 0 表示使用全部训练集。
    args["test_limit"] = 1000  # 默认只取 1000 张测试图以便纯 Python 能较快跑完，设为 0 表示使用全部测试集。
    args["linear_epochs"] = 8  # 线性 softmax 模型默认训练轮数。
    args["perceptron_epochs"] = 8  # 多分类感知机默认训练轮数。
    args["nn_epochs"] = 6  # 单隐层神经网络默认训练轮数。
    args["deep_epochs"] = 6  # 多隐层神经网络默认训练轮数。
    args["tree_depth"] = 10  # 决策树默认最大深度，避免纯 Python 在 MNIST 上过慢或过拟合。
    args["logic_only"] = False  # 是否只运行与或非异或数据集。
    args["mnist_only"] = False  # 是否只运行 MNIST 数据集。
    i = 1  # 从第 1 个命令行参数开始读取。
    while i < len(sys.argv):  # 遍历全部命令行参数。
        key = sys.argv[i]  # 取出当前参数名。
        if key == "--logic-only":  # 如果用户指定只跑逻辑数据集。
            args["logic_only"] = True  # 设置只跑逻辑实验。
            i += 1  # 移动到下一个参数。
        elif key == "--mnist-only":  # 如果用户指定只跑 MNIST。
            args["mnist_only"] = True  # 设置只跑 MNIST 实验。
            i += 1  # 移动到下一个参数。
        elif key.startswith("--") and i + 1 < len(sys.argv):  # 如果是带值参数并且后面还有一个值。
            name = key[2:].replace("-", "_")  # 把 --train-limit 这类名称转成 train_limit。
            value = sys.argv[i + 1]  # 取出参数值。
            if name in ["seed", "train_limit", "test_limit", "linear_epochs", "perceptron_epochs", "nn_epochs", "deep_epochs", "tree_depth"]:  # 判断是否为整数参数。
                args[name] = int(value)  # 把整数参数转换为 int。
            elif name in ["data_dir", "out_dir"]:  # 判断是否为路径参数。
                args[name] = value  # 保存路径字符串。
            else:  # 如果遇到未知参数。
                out("忽略未知参数：" + key)  # 通过 stdout 提示但不中断程序。
            i += 2  # 跳过参数名和参数值。
        else:  # 如果参数格式不正确。
            out("忽略无法解析的参数：" + key)  # 输出提示信息。
            i += 1  # 继续处理后续参数。
    return args  # 返回最终配置字典。


def ensure_dir(path):  # 定义创建输出目录的辅助函数。
    p = Path(path)  # 把字符串路径转换为 Path 对象。
    p.mkdir(parents=True, exist_ok=True)  # 若目录不存在则递归创建。
    return p  # 返回 Path 对象便于后续拼接文件名。


# ========================== 通用数学工具区 ========================== # 下面函数均为从零实现的基础算法工具。


def argmax(values):  # 返回列表中最大值的下标。
    best_i = 0  # 初始化最大值下标为 0。
    best_v = values[0]  # 初始化最大值为第一个元素。
    for i in range(1, len(values)):  # 从第二个元素开始遍历。
        if values[i] > best_v:  # 如果当前值更大。
            best_i = i  # 更新最大值下标。
            best_v = values[i]  # 更新最大值。
    return best_i  # 返回最大值下标。


def dot(a, b):  # 计算两个等长列表的点积。
    s = 0.0  # 初始化累加和。
    for i in range(len(a)):  # 遍历向量的每一维。
        s += a[i] * b[i]  # 累加对应元素乘积。
    return s  # 返回点积结果。


def softmax(scores):  # 手写 softmax 函数，避免调用任何第三方库。
    m = max(scores)  # 减去最大值提高数值稳定性。
    exps = []  # 创建指数值列表。
    total = 0.0  # 初始化指数和。
    for s in scores:  # 遍历每个原始分数。
        e = math.exp(s - m)  # 计算平移后的指数。
        exps.append(e)  # 保存指数值。
        total += e  # 累加指数和。
    probs = []  # 创建概率列表。
    for e in exps:  # 遍历每个指数值。
        probs.append(e / total)  # 归一化得到概率。
    return probs  # 返回类别概率。


def relu(x):  # 定义 ReLU 激活函数。
    if x > 0.0:  # 如果输入为正。
        return x  # 返回原值。
    return 0.0  # 如果输入非正则返回 0。


def relu_grad(x):  # 定义 ReLU 的导数。
    if x > 0.0:  # 如果预激活值为正。
        return 1.0  # 导数为 1。
    return 0.0  # 如果预激活值非正则导数为 0。


def accuracy(model, X, y):  # 计算模型在数据集上的分类精度。
    if len(X) == 0:  # 防止空数据集导致除零错误。
        return 0.0  # 空数据集精度记为 0。
    correct = 0  # 初始化正确样本数。
    for i in range(len(X)):  # 遍历全部样本。
        pred = model.predict_one(X[i])  # 调用模型预测单个样本。
        if pred == y[i]:  # 判断预测是否等于真实标签。
            correct += 1  # 正确样本数加一。
    return correct / len(X)  # 返回正确率。


def confusion_matrix(model, X, y, class_count):  # 计算多分类混淆矩阵。
    mat = []  # 创建矩阵列表。
    for _ in range(class_count):  # 为每个真实类别创建一行。
        mat.append([0 for _ in range(class_count)])  # 初始化一行全 0。
    for i in range(len(X)):  # 遍历全部样本。
        pred = model.predict_one(X[i])  # 得到预测类别。
        truth = y[i]  # 得到真实类别。
        if 0 <= truth < class_count and 0 <= pred < class_count:  # 保证下标合法。
            mat[truth][pred] += 1  # 累加真实类到预测类的计数。
    return mat  # 返回混淆矩阵。


# ========================== 数据集构造区 ========================== # 第一类逻辑数据直接在代码中构造，第二类 MNIST 从四个文件读取。


def make_logic_datasets():  # 构造与、或、非、异或四个小数据集。
    datasets = {}  # 创建数据集字典。
    datasets["and"] = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 0, 0, 1])  # 与门数据集。
    datasets["or"] = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 1])  # 或门数据集。
    datasets["xor"] = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 0])  # 异或数据集。
    datasets["not"] = ([[0.0], [1.0]], [1, 0])  # 非门数据集，其中输入 0 输出 1，输入 1 输出 0。
    return datasets  # 返回全部逻辑数据集。


def find_data_file(root, names):  # 在代码同目录或用户指定目录下寻找 MNIST 文件。
    root = Path(root)  # 把根目录转换为 Path 对象。
    for name in names:  # 优先检查候选文件名是否直接存在。
        direct = root / name  # 拼接直接路径。
        if direct.is_file():  # 如果直接路径就是文件。
            return direct  # 返回该文件路径。
        if direct.is_dir():  # 如果用户把解压结果放成同名文件夹。
            for child in direct.rglob("*"):  # 在该文件夹内部递归寻找真实 IDX 文件。
                if child.is_file() and (child.name in names or child.name.replace(".", "-") in names):  # 判断文件名是否匹配候选名称。
                    return child  # 返回找到的真实文件路径。
    for child in root.rglob("*"):  # 如果直接检查失败，则在根目录递归搜索。
        if child.is_file() and child.name in names:  # 判断文件名是否为候选名称。
            return child  # 返回找到的文件。
    return None  # 如果仍未找到则返回 None。


def open_maybe_gzip(path):  # 打开普通 IDX 文件或 gzip 压缩 IDX 文件。
    if str(path).endswith(".gz"):  # 判断是否为 .gz 文件。
        return gzip.open(path, "rb")  # 用 gzip 按二进制方式打开。
    return open(path, "rb")  # 否则用普通二进制文件方式打开。


def read_idx_images(path, limit):  # 读取 MNIST 图片 IDX 文件，返回每张图片的原始 bytes。
    images = []  # 创建图片列表。
    with open_maybe_gzip(path) as f:  # 打开 IDX 图片文件。
        header = f.read(16)  # 读取 16 字节文件头。
        magic, count, rows, cols = struct.unpack(">IIII", header)  # 按大端格式解析 magic、图片数、行数、列数。
        if magic != 2051:  # MNIST 图片文件 magic 应为 2051。
            raise ValueError("图片文件 magic 不正确：" + str(path))  # magic 不对说明文件放错。
        use_count = count if limit == 0 else min(count, limit)  # limit 为 0 时读取全部，否则读取指定数量。
        size = rows * cols  # 每张图片的像素数量，MNIST 为 28*28。
        for _ in range(use_count):  # 逐张读取图片。
            data = f.read(size)  # 读取一张图片的原始像素 bytes。
            if len(data) != size:  # 如果长度不足说明文件损坏或提前结束。
                break  # 停止读取。
            images.append(data)  # 保存图片 bytes，避免把 784 个像素都转成 int 后占用大量内存。
    return images, rows, cols  # 返回图片列表和尺寸。


def read_idx_labels(path, limit):  # 读取 MNIST 标签 IDX 文件。
    labels = []  # 创建标签列表。
    with open_maybe_gzip(path) as f:  # 打开 IDX 标签文件。
        header = f.read(8)  # 读取 8 字节标签文件头。
        magic, count = struct.unpack(">II", header)  # 解析 magic 和标签数量。
        if magic != 2049:  # MNIST 标签文件 magic 应为 2049。
            raise ValueError("标签文件 magic 不正确：" + str(path))  # magic 不对说明文件放错。
        use_count = count if limit == 0 else min(count, limit)  # limit 为 0 时读取全部，否则读取指定数量。
        data = f.read(use_count)  # 一次读取指定数量的标签字节。
        for b in data:  # 遍历每个标签字节。
            labels.append(int(b))  # 转换为 Python 整数类别。
    return labels  # 返回标签列表。


def image_to_features(image, rows, cols, grid=8):  # 把 28*28 灰度图片压缩为 grid*grid 的平均池化特征。
    features = []  # 创建特征列表。
    for gy in range(grid):  # 遍历池化网格的行。
        y0 = gy * rows // grid  # 计算当前网格块的起始行。
        y1 = (gy + 1) * rows // grid  # 计算当前网格块的结束行。
        for gx in range(grid):  # 遍历池化网格的列。
            x0 = gx * cols // grid  # 计算当前网格块的起始列。
            x1 = (gx + 1) * cols // grid  # 计算当前网格块的结束列。
            s = 0.0  # 初始化像素和。
            n = 0  # 初始化像素计数。
            for yy in range(y0, y1):  # 遍历当前块的每一行。
                base = yy * cols  # 计算当前行在一维 bytes 中的起点。
                for xx in range(x0, x1):  # 遍历当前块的每一列。
                    s += image[base + xx] / 255.0  # 累加归一化灰度值。
                    n += 1  # 累加像素数量。
            features.append(s / n if n else 0.0)  # 保存平均灰度作为一个特征。
    return features  # 返回低维特征。


def images_to_features(images, rows, cols):  # 批量把图片 bytes 转为低维特征列表。
    X = []  # 创建特征矩阵。
    for img in images:  # 遍历每张图片。
        X.append(image_to_features(img, rows, cols, 8))  # 使用 8*8 平均池化得到 64 维特征。
    return X  # 返回特征矩阵。


def load_mnist(data_dir, train_limit, test_limit):  # 从四个 MNIST 文件读取训练集和测试集。
    root = Path(data_dir)  # 把数据目录转换为 Path。
    train_img_names = ["train-images-idx3-ubyte", "train-images.idx3-ubyte", "train-images-idx3-ubyte.gz", "train-images.idx3-ubyte.gz"]  # 训练图片文件候选名。
    train_lbl_names = ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte", "train-labels-idx1-ubyte.gz", "train-labels.idx1-ubyte.gz"]  # 训练标签文件候选名。
    test_img_names = ["t10k-images-idx3-ubyte", "t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte.gz", "t10k-images.idx3-ubyte.gz"]  # 测试图片文件候选名。
    test_lbl_names = ["t10k-labels-idx1-ubyte", "t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte.gz", "t10k-labels.idx1-ubyte.gz"]  # 测试标签文件候选名。
    train_img_path = find_data_file(root, train_img_names)  # 在代码同目录寻找训练图片文件或文件夹。
    train_lbl_path = find_data_file(root, train_lbl_names)  # 在代码同目录寻找训练标签文件或文件夹。
    test_img_path = find_data_file(root, test_img_names)  # 在代码同目录寻找测试图片文件或文件夹。
    test_lbl_path = find_data_file(root, test_lbl_names)  # 在代码同目录寻找测试标签文件或文件夹。
    paths = [train_img_path, train_lbl_path, test_img_path, test_lbl_path]  # 汇总四个路径便于统一检查。
    if any(p is None for p in paths):  # 如果任意一个路径没有找到。
        out("未找到完整 MNIST 四文件，MNIST 实验将跳过。")  # 输出跳过提示。
        out("请把 train-images-idx3-ubyte、train-labels-idx1-ubyte、t10k-images-idx3-ubyte、t10k-labels-idx1-ubyte 放到代码同目录。")  # 提示四个文件名。
        return None  # 返回 None 表示无法运行 MNIST。
    out("MNIST 训练图片：" + str(train_img_path))  # 输出训练图片路径。
    out("MNIST 训练标签：" + str(train_lbl_path))  # 输出训练标签路径。
    out("MNIST 测试图片：" + str(test_img_path))  # 输出测试图片路径。
    out("MNIST 测试标签：" + str(test_lbl_path))  # 输出测试标签路径。
    train_images, rows, cols = read_idx_images(train_img_path, train_limit)  # 读取训练图片。
    train_labels = read_idx_labels(train_lbl_path, train_limit)  # 读取训练标签。
    test_images, test_rows, test_cols = read_idx_images(test_img_path, test_limit)  # 读取测试图片。
    test_labels = read_idx_labels(test_lbl_path, test_limit)  # 读取测试标签。
    n_train = min(len(train_images), len(train_labels))  # 对齐训练图片与标签数量。
    n_test = min(len(test_images), len(test_labels))  # 对齐测试图片与标签数量。
    train_images = train_images[:n_train]  # 截断训练图片到对齐长度。
    train_labels = train_labels[:n_train]  # 截断训练标签到对齐长度。
    test_images = test_images[:n_test]  # 截断测试图片到对齐长度。
    test_labels = test_labels[:n_test]  # 截断测试标签到对齐长度。
    X_train = images_to_features(train_images, rows, cols)  # 把训练图片转换为 64 维平均池化特征。
    X_test = images_to_features(test_images, test_rows, test_cols)  # 把测试图片转换为 64 维平均池化特征。
    return X_train, train_labels, X_test, test_labels, test_images, rows, cols  # 返回特征、标签和测试图片。


# ========================== 模型一：线性模型 ========================== # 使用 softmax 回归作为第 3 章线性模型代表。


class SoftmaxLinear:  # 定义手写多分类 softmax 线性模型。
    def __init__(self, input_dim, class_count, lr=0.1, epochs=10, seed=7):  # 初始化模型超参数。
        self.input_dim = input_dim  # 保存输入维度。
        self.class_count = class_count  # 保存类别数。
        self.lr = lr  # 保存学习率。
        self.epochs = epochs  # 保存训练轮数。
        random.seed(seed)  # 设置随机种子。
        self.W = []  # 创建权重矩阵。
        self.b = []  # 创建偏置向量。
        scale = 1.0 / math.sqrt(max(1, input_dim))  # 计算小随机初始化范围。
        for _ in range(class_count):  # 为每个类别创建一组权重。
            self.W.append([(random.random() * 2.0 - 1.0) * scale for _ in range(input_dim)])  # 初始化该类别权重。
            self.b.append(0.0)  # 初始化该类别偏置。

    def scores(self, x):  # 计算每个类别的线性打分。
        res = []  # 创建分数列表。
        for k in range(self.class_count):  # 遍历每个类别。
            res.append(dot(self.W[k], x) + self.b[k])  # 计算 w_k*x+b_k。
        return res  # 返回全部类别分数。

    def predict_one(self, x):  # 预测单个样本类别。
        return argmax(self.scores(x))  # 分数最大的类别即预测结果。

    def train(self, X, y, name="linear"):  # 使用随机梯度下降训练 softmax 回归。
        order = list(range(len(X)))  # 创建样本下标列表。
        for epoch in range(self.epochs):  # 遍历每一轮训练。
            random.shuffle(order)  # 打乱样本顺序以提高 SGD 效果。
            lr = self.lr / (1.0 + 0.05 * epoch)  # 使用轻微学习率衰减。
            loss_sum = 0.0  # 初始化本轮损失和。
            correct = 0  # 初始化本轮正确数。
            for idx in order:  # 遍历打乱后的样本。
                x = X[idx]  # 取出特征。
                label = y[idx]  # 取出标签。
                probs = softmax(self.scores(x))  # 计算 softmax 概率。
                pred = argmax(probs)  # 得到预测类别。
                if pred == label:  # 判断是否预测正确。
                    correct += 1  # 正确数加一。
                loss_sum += -math.log(max(probs[label], 1e-12))  # 累加交叉熵损失。
                for k in range(self.class_count):  # 遍历每个类别更新参数。
                    g = probs[k] - (1.0 if k == label else 0.0)  # 计算 softmax+交叉熵梯度。
                    for j in range(self.input_dim):  # 遍历每个特征维度。
                        self.W[k][j] -= lr * g * x[j]  # 更新权重。
                    self.b[k] -= lr * g  # 更新偏置。
            out(name + " epoch " + str(epoch + 1) + "/" + str(self.epochs) + " loss=" + format(loss_sum / max(1, len(X)), ".4f") + " acc=" + format(correct / max(1, len(X)), ".4f"))  # 输出训练进度。


# ========================== 模型二：决策树 ========================== # 手写 CART 风格二叉决策树，连续特征用阈值划分。


class TreeNode:  # 定义决策树结点。
    def __init__(self):  # 初始化结点。
        self.is_leaf = True  # 默认先视为叶子结点。
        self.label = 0  # 叶子结点默认预测类别。
        self.feature = -1  # 内部结点使用的特征下标。
        self.threshold = 0.0  # 内部结点使用的阈值。
        self.left = None  # 小于等于阈值的左子树。
        self.right = None  # 大于阈值的右子树。


class DecisionTree:  # 定义手写决策树分类器。
    def __init__(self, class_count, max_depth=8, min_samples=2, max_thresholds=8):  # 初始化决策树超参数。
        self.class_count = class_count  # 保存类别数。
        self.max_depth = max_depth  # 保存最大深度。
        self.min_samples = min_samples  # 保存最小分裂样本数。
        self.max_thresholds = max_thresholds  # 每个特征最多尝试的阈值数量。
        self.root = None  # 初始化根结点为空。
        self.X = None  # 训练时暂存特征矩阵。
        self.y = None  # 训练时暂存标签。

    def majority_label(self, rows):  # 计算当前样本集合的多数类。
        counts = [0 for _ in range(self.class_count)]  # 初始化类别计数。
        for idx in rows:  # 遍历样本下标。
            counts[self.y[idx]] += 1  # 累加该类别数量。
        return argmax(counts)  # 返回数量最多的类别。

    def gini_from_counts(self, counts):  # 根据类别计数计算 Gini 不纯度。
        total = sum(counts)  # 计算样本总数。
        if total == 0:  # 如果没有样本。
            return 0.0  # 不纯度记为 0。
        s = 1.0  # Gini 初始为 1。
        for c in counts:  # 遍历每个类别计数。
            p = c / total  # 计算该类别比例。
            s -= p * p  # 减去类别比例平方。
        return s  # 返回 Gini 不纯度。

    def candidate_thresholds(self, rows, feat):  # 为一个连续特征生成候选阈值。
        vals = []  # 创建特征值列表。
        for idx in rows:  # 遍历样本下标。
            vals.append(self.X[idx][feat])  # 保存该特征值。
        vals = sorted(set(vals))  # 去重并排序。
        if len(vals) <= 1:  # 如果只有一个取值则无法划分。
            return []  # 返回空阈值列表。
        mids = []  # 创建中点阈值列表。
        if len(vals) <= self.max_thresholds + 1:  # 如果不同取值很少。
            for i in range(len(vals) - 1):  # 遍历相邻取值。
                mids.append((vals[i] + vals[i + 1]) * 0.5)  # 用相邻取值中点作为阈值。
        else:  # 如果不同取值很多。
            for t in range(1, self.max_thresholds + 1):  # 按分位位置选有限个阈值。
                pos = t * (len(vals) - 1) // (self.max_thresholds + 1)  # 计算分位下标。
                mids.append((vals[pos] + vals[pos + 1]) * 0.5)  # 保存分位附近中点阈值。
        return mids  # 返回候选阈值。

    def best_split(self, rows):  # 在当前样本集合寻找最佳特征和阈值。
        input_dim = len(self.X[0])  # 获取输入特征维度。
        base_counts = [0 for _ in range(self.class_count)]  # 初始化当前结点类别计数。
        for idx in rows:  # 遍历当前样本集合。
            base_counts[self.y[idx]] += 1  # 累加类别计数。
        base_gini = self.gini_from_counts(base_counts)  # 计算当前结点 Gini。
        best_gain = 0.0  # 初始化最佳增益。
        best_feat = -1  # 初始化最佳特征。
        best_thr = 0.0  # 初始化最佳阈值。
        for feat in range(input_dim):  # 遍历每一个特征维度。
            thresholds = self.candidate_thresholds(rows, feat)  # 生成该特征候选阈值。
            for thr in thresholds:  # 遍历候选阈值。
                left_counts = [0 for _ in range(self.class_count)]  # 初始化左子集计数。
                right_counts = [0 for _ in range(self.class_count)]  # 初始化右子集计数。
                left_n = 0  # 初始化左子集样本数。
                right_n = 0  # 初始化右子集样本数。
                for idx in rows:  # 遍历当前样本。
                    if self.X[idx][feat] <= thr:  # 判断是否进入左子集。
                        left_counts[self.y[idx]] += 1  # 左子集类别计数加一。
                        left_n += 1  # 左子集样本数加一。
                    else:  # 否则进入右子集。
                        right_counts[self.y[idx]] += 1  # 右子集类别计数加一。
                        right_n += 1  # 右子集样本数加一。
                if left_n == 0 or right_n == 0:  # 如果某侧为空则无效。
                    continue  # 跳过该阈值。
                left_gini = self.gini_from_counts(left_counts)  # 计算左子集 Gini。
                right_gini = self.gini_from_counts(right_counts)  # 计算右子集 Gini。
                weighted = (left_n * left_gini + right_n * right_gini) / len(rows)  # 计算划分后加权 Gini。
                gain = base_gini - weighted  # 计算 Gini 降低量。
                if gain > best_gain:  # 如果当前划分更好。
                    best_gain = gain  # 更新最佳增益。
                    best_feat = feat  # 更新最佳特征。
                    best_thr = thr  # 更新最佳阈值。
        return best_feat, best_thr, best_gain  # 返回最佳划分。

    def build(self, rows, depth):  # 递归构建决策树。
        node = TreeNode()  # 创建新结点。
        node.label = self.majority_label(rows)  # 默认把当前结点设为多数类叶子。
        if depth >= self.max_depth:  # 如果达到最大深度。
            return node  # 返回叶子结点。
        if len(rows) < self.min_samples:  # 如果样本数太少。
            return node  # 返回叶子结点。
        same = True  # 假设当前结点全部同类。
        first = self.y[rows[0]]  # 取第一个样本标签。
        for idx in rows:  # 遍历样本下标。
            if self.y[idx] != first:  # 如果存在不同类别。
                same = False  # 标记不是纯结点。
                break  # 停止检查。
        if same:  # 如果当前样本全属于同一类别。
            return node  # 直接返回叶子。
        feat, thr, gain = self.best_split(rows)  # 寻找最佳划分。
        if feat < 0 or gain <= 1e-12:  # 如果找不到有效划分。
            return node  # 返回叶子结点。
        left_rows = []  # 创建左子集下标列表。
        right_rows = []  # 创建右子集下标列表。
        for idx in rows:  # 遍历样本下标。
            if self.X[idx][feat] <= thr:  # 如果特征值小于等于阈值。
                left_rows.append(idx)  # 放入左子集。
            else:  # 否则放入右子集。
                right_rows.append(idx)  # 放入右子集。
        if len(left_rows) == 0 or len(right_rows) == 0:  # 防止退化划分。
            return node  # 返回叶子结点。
        node.is_leaf = False  # 将当前结点改为内部结点。
        node.feature = feat  # 保存划分特征。
        node.threshold = thr  # 保存划分阈值。
        node.left = self.build(left_rows, depth + 1)  # 递归构建左子树。
        node.right = self.build(right_rows, depth + 1)  # 递归构建右子树。
        return node  # 返回构建完成的结点。

    def train(self, X, y, name="tree"):  # 训练决策树模型。
        self.X = X  # 暂存训练特征矩阵。
        self.y = y  # 暂存训练标签。
        rows = list(range(len(X)))  # 创建全部训练样本下标。
        self.root = self.build(rows, 0)  # 从根结点开始递归建树。
        out(name + " train finished")  # 输出训练完成信息。

    def predict_one(self, x):  # 预测单个样本类别。
        node = self.root  # 从根结点开始。
        while node is not None and not node.is_leaf:  # 只要当前不是叶子就继续向下。
            if x[node.feature] <= node.threshold:  # 判断是否进入左子树。
                node = node.left  # 移动到左子树。
            else:  # 否则进入右子树。
                node = node.right  # 移动到右子树。
        if node is None:  # 理论上不应为空，仍做保护。
            return 0  # 返回默认类别。
        return node.label  # 返回叶子结点多数类标签。


# ========================== 模型三：单层感知机 ========================== # 使用多分类感知机处理二分类逻辑和十分类 MNIST。


class MulticlassPerceptron:  # 定义手写多分类感知机。
    def __init__(self, input_dim, class_count, lr=0.1, epochs=10, seed=7):  # 初始化感知机。
        self.input_dim = input_dim  # 保存输入维度。
        self.class_count = class_count  # 保存类别数。
        self.lr = lr  # 保存学习率。
        self.epochs = epochs  # 保存训练轮数。
        random.seed(seed)  # 设置随机种子。
        self.W = []  # 创建权重矩阵。
        self.b = []  # 创建偏置向量。
        for _ in range(class_count):  # 为每个类别创建一组权重。
            self.W.append([0.0 for _ in range(input_dim)])  # 感知机权重初始化为 0。
            self.b.append(0.0)  # 感知机偏置初始化为 0。

    def scores(self, x):  # 计算每个类别的线性打分。
        res = []  # 创建分数列表。
        for k in range(self.class_count):  # 遍历类别。
            res.append(dot(self.W[k], x) + self.b[k])  # 计算类别得分。
        return res  # 返回全部类别得分。

    def predict_one(self, x):  # 预测单个样本。
        return argmax(self.scores(x))  # 选择得分最高类别。

    def train(self, X, y, name="perceptron"):  # 使用感知机更新规则训练。
        order = list(range(len(X)))  # 创建样本下标。
        for epoch in range(self.epochs):  # 遍历训练轮数。
            random.shuffle(order)  # 打乱样本顺序。
            wrong = 0  # 初始化错误数。
            lr = self.lr / (1.0 + 0.05 * epoch)  # 使用轻微学习率衰减。
            for idx in order:  # 遍历样本。
                x = X[idx]  # 取出特征。
                label = y[idx]  # 取出真实类别。
                pred = self.predict_one(x)  # 预测类别。
                if pred != label:  # 只有预测错误时更新。
                    wrong += 1  # 错误数加一。
                    for j in range(self.input_dim):  # 遍历特征维度。
                        self.W[label][j] += lr * x[j]  # 增加真实类别权重。
                        self.W[pred][j] -= lr * x[j]  # 减少错误预测类别权重。
                    self.b[label] += lr  # 增加真实类别偏置。
                    self.b[pred] -= lr  # 减少错误预测类别偏置。
            out(name + " epoch " + str(epoch + 1) + "/" + str(self.epochs) + " acc=" + format(1.0 - wrong / max(1, len(X)), ".4f"))  # 输出训练进度。


# ========================== 模型四与五：全连接神经网络 ========================== # 手写反向传播，任务四为单隐层，任务五为多隐层。


class MLP:  # 定义手写全连接神经网络。
    def __init__(self, input_dim, hidden_sizes, class_count, lr=0.05, epochs=8, seed=7):  # 初始化 MLP。
        self.input_dim = input_dim  # 保存输入维度。
        self.hidden_sizes = hidden_sizes[:]  # 保存隐藏层维度列表。
        self.class_count = class_count  # 保存类别数。
        self.lr = lr  # 保存学习率。
        self.epochs = epochs  # 保存训练轮数。
        random.seed(seed)  # 设置随机种子。
        sizes = [input_dim] + hidden_sizes + [class_count]  # 汇总每一层神经元数量。
        self.W = []  # 创建每一层权重矩阵列表。
        self.b = []  # 创建每一层偏置向量列表。
        for l in range(len(sizes) - 1):  # 遍历相邻层。
            fan_in = sizes[l]  # 当前层输入维度。
            fan_out = sizes[l + 1]  # 当前层输出维度。
            scale = math.sqrt(2.0 / max(1, fan_in))  # 使用适合 ReLU 的小随机初始化范围。
            W_l = []  # 创建当前层权重矩阵。
            for _ in range(fan_out):  # 为当前层每个输出神经元创建权重。
                W_l.append([(random.random() * 2.0 - 1.0) * scale for _ in range(fan_in)])  # 初始化一行权重。
            self.W.append(W_l)  # 保存当前层权重。
            self.b.append([0.0 for _ in range(fan_out)])  # 保存当前层偏置。

    def forward(self, x):  # 前向传播，返回激活值、预激活值和输出概率。
        activations = [x]  # 第 0 层激活就是输入特征。
        preacts = []  # 保存每一层的线性预激活值。
        a = x  # 当前激活初始化为输入。
        for l in range(len(self.W)):  # 遍历每一层。
            z = []  # 创建当前层预激活列表。
            for o in range(len(self.W[l])):  # 遍历当前层输出神经元。
                z.append(dot(self.W[l][o], a) + self.b[l][o])  # 计算当前神经元线性输出。
            preacts.append(z)  # 保存当前层预激活。
            if l == len(self.W) - 1:  # 如果是输出层。
                a = softmax(z)  # 输出层使用 softmax 得到类别概率。
            else:  # 如果是隐藏层。
                a = [relu(v) for v in z]  # 隐藏层使用 ReLU 激活。
            activations.append(a)  # 保存当前层激活。
        return activations, preacts, activations[-1]  # 返回全部中间结果和输出概率。

    def predict_one(self, x):  # 预测单个样本类别。
        _, _, probs = self.forward(x)  # 执行前向传播。
        return argmax(probs)  # 概率最大的类别为预测类别。

    def train(self, X, y, name="mlp"):  # 使用 SGD 和反向传播训练 MLP。
        order = list(range(len(X)))  # 创建样本下标列表。
        for epoch in range(self.epochs):  # 遍历训练轮数。
            random.shuffle(order)  # 打乱样本顺序。
            lr = self.lr / (1.0 + 0.05 * epoch)  # 使用轻微学习率衰减。
            loss_sum = 0.0  # 初始化本轮损失。
            correct = 0  # 初始化本轮正确数。
            for idx in order:  # 遍历样本。
                x = X[idx]  # 取出样本特征。
                label = y[idx]  # 取出真实标签。
                activations, preacts, probs = self.forward(x)  # 前向传播。
                pred = argmax(probs)  # 得到预测类别。
                if pred == label:  # 判断是否预测正确。
                    correct += 1  # 正确数加一。
                loss_sum += -math.log(max(probs[label], 1e-12))  # 累加交叉熵损失。
                deltas = [None for _ in range(len(self.W))]  # 创建每层误差项列表。
                out_delta = probs[:]  # 输出层 delta 初值为 softmax 概率。
                out_delta[label] -= 1.0  # softmax+交叉熵对真实类减 1。
                deltas[-1] = out_delta  # 保存输出层 delta。
                for l in range(len(self.W) - 2, -1, -1):  # 从倒数第二层向前反传。
                    next_delta = deltas[l + 1]  # 获取后一层 delta。
                    cur_delta = []  # 创建当前隐藏层 delta。
                    for i in range(len(self.W[l])):  # 遍历当前隐藏层神经元。
                        s = 0.0  # 初始化来自后一层的梯度和。
                        for o in range(len(next_delta)):  # 遍历后一层神经元。
                            s += next_delta[o] * self.W[l + 1][o][i]  # 按后一层旧权重累加误差。
                        cur_delta.append(s * relu_grad(preacts[l][i]))  # 乘以 ReLU 导数得到当前层 delta。
                    deltas[l] = cur_delta  # 保存当前层 delta。
                for l in range(len(self.W)):  # 遍历每一层更新参数。
                    a_prev = activations[l]  # 获取当前层输入激活。
                    for o in range(len(self.W[l])):  # 遍历当前层输出神经元。
                        d = deltas[l][o]  # 获取当前神经元 delta。
                        for j in range(len(a_prev)):  # 遍历输入维度。
                            self.W[l][o][j] -= lr * d * a_prev[j]  # 更新当前权重。
                        self.b[l][o] -= lr * d  # 更新当前偏置。
            out(name + " epoch " + str(epoch + 1) + "/" + str(self.epochs) + " loss=" + format(loss_sum / max(1, len(X)), ".4f") + " acc=" + format(correct / max(1, len(X)), ".4f"))  # 输出训练进度。


# ========================== 可视化输出区 ========================== # 本版只在画图环节使用 matplotlib，不使用 numpy、pandas、sklearn、torch。 


def color_for_class(c):  # 给类别分配可视化颜色，供散点和说明使用。
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:cyan", "tab:pink", "tab:gray", "tab:olive", "tab:brown"]  # 使用 matplotlib 内置颜色名。
    return colors[c % len(colors)]  # 按类别编号循环取颜色。


def write_logic_matplotlib(path, model, X, y, title):  # 使用 matplotlib 输出逻辑数据集二维分类边界图。
    dim = len(X[0])  # 获取输入维度，非门是一维，其余逻辑门是二维。
    xmin = -0.25  # 设置横轴显示最小值。
    xmax = 1.25  # 设置横轴显示最大值。
    ymin = -0.25  # 设置纵轴显示最小值。
    ymax = 1.25  # 设置纵轴显示最大值。
    grid_n = 90  # 设置背景网格密度，数值越大边界越细腻但画图越慢。
    grid = []  # 创建背景类别网格。
    for gy in range(grid_n):  # 遍历背景网格的每一行。
        row = []  # 创建当前行的类别列表。
        vy = ymin + gy / (grid_n - 1) * (ymax - ymin)  # 计算当前行对应的数据纵坐标。
        for gx in range(grid_n):  # 遍历背景网格的每一列。
            vx = xmin + gx / (grid_n - 1) * (xmax - xmin)  # 计算当前列对应的数据横坐标。
            sample = [vx] if dim == 1 else [vx, vy]  # 非门只输入 x1，其他逻辑门输入 x1 和 x2。
            row.append(model.predict_one(sample))  # 把当前网格点的预测类别写入背景。
        grid.append(row)  # 保存当前行类别。
    plt.figure(figsize=(5.2, 4.8))  # 创建一张适合报告粘贴的图片。
    plt.imshow(grid, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="Pastel1", alpha=0.75, aspect="auto")  # 绘制分类背景色。
    for i in range(len(X)):  # 遍历真实样本点。
        vx = X[i][0]  # 取出样本的第一维特征。
        vy = 0.5 if dim == 1 else X[i][1]  # 非门固定画在 y=0.5，二维逻辑门使用第二维特征。
        plt.scatter([vx], [vy], s=120, c=color_for_class(y[i]), edgecolors="black", linewidths=1.0, marker="o")  # 绘制真实样本散点。
        plt.text(vx + 0.03, vy + 0.03, "class=" + str(y[i]), fontsize=9)  # 在样本点旁标注真实类别。
    plt.axhline(0.0, color="black", linewidth=0.8)  # 绘制横轴辅助线。
    plt.axvline(0.0, color="black", linewidth=0.8)  # 绘制纵轴辅助线。
    plt.xlim(xmin, xmax)  # 设置横轴范围。
    plt.ylim(ymin, ymax)  # 设置纵轴范围。
    plt.xlabel("x1")  # 设置横轴标题。
    plt.ylabel("x2 / display axis")  # 设置纵轴标题。
    plt.title(title)  # 设置图片标题。
    plt.grid(True, linewidth=0.3, alpha=0.4)  # 添加浅色网格线，方便观察分类边界。
    plt.tight_layout()  # 自动压缩边距，避免标题或坐标轴被裁剪。
    plt.savefig(path, dpi=180)  # 保存 PNG 图片。
    plt.close()  # 关闭当前图，避免多图叠加和内存占用。


def write_confusion_txt(path, mat):  # 把混淆矩阵写为文本表格，方便报告中引用具体数字。
    lines = []  # 创建文本行列表。
    lines.append("rows=true labels, columns=predicted labels")  # 第一行说明行列含义。
    header = "true\\pred " + " ".join([str(i).rjust(5) for i in range(len(mat))])  # 创建表头。
    lines.append(header)  # 保存表头。
    for i in range(len(mat)):  # 遍历矩阵的每一行。
        line = str(i).rjust(9) + " " + " ".join([str(v).rjust(5) for v in mat[i]])  # 把一行数字格式化为等宽文本。
        lines.append(line)  # 保存当前行。
    Path(path).write_text("\n".join(lines), encoding="utf-8")  # 写入文本文件。


def write_confusion_matplotlib(path, mat, title):  # 使用 matplotlib 输出 MNIST 混淆矩阵热力图。
    n = len(mat)  # 获取类别数量。
    plt.figure(figsize=(7.0, 6.2))  # 创建混淆矩阵画布。
    im = plt.imshow(mat, cmap="Blues")  # 使用蓝色热力图显示混淆矩阵计数。
    plt.colorbar(im, fraction=0.046, pad=0.04)  # 添加颜色条说明计数大小。
    plt.xticks(list(range(n)), [str(i) for i in range(n)])  # 设置横轴类别刻度。
    plt.yticks(list(range(n)), [str(i) for i in range(n)])  # 设置纵轴类别刻度。
    plt.xlabel("Predicted label")  # 设置横轴标题。
    plt.ylabel("True label")  # 设置纵轴标题。
    plt.title(title)  # 设置图标题。
    max_v = 1  # 初始化最大计数，防止全零时除零。
    for row in mat:  # 遍历矩阵每一行。
        for v in row:  # 遍历每个计数。
            if v > max_v:  # 如果当前计数更大。
                max_v = v  # 更新最大计数。
    for i in range(n):  # 遍历真实类别。
        for j in range(n):  # 遍历预测类别。
            color = "white" if mat[i][j] > max_v * 0.55 else "black"  # 深色背景上用白字，浅色背景上用黑字。
            plt.text(j, i, str(mat[i][j]), ha="center", va="center", color=color, fontsize=8)  # 在每个格子中写入计数。
    plt.tight_layout()  # 自动调整边距。
    plt.savefig(path, dpi=180)  # 保存 PNG 图片。
    plt.close()  # 关闭当前图。


def write_per_class_accuracy_matplotlib(path, mat, title):  # 使用 matplotlib 输出每个数字类别的测试准确率柱状图。
    labels = []  # 创建类别标签列表。
    values = []  # 创建每类准确率列表。
    for i in range(len(mat)):  # 遍历每个真实类别。
        total = sum(mat[i])  # 计算当前真实类别的样本总数。
        acc = mat[i][i] / total if total > 0 else 0.0  # 计算当前类别被正确分类的比例。
        labels.append(str(i))  # 保存类别名称。
        values.append(acc * 100.0)  # 保存百分制准确率。
    plt.figure(figsize=(7.0, 4.2))  # 创建柱状图画布。
    xs = list(range(len(labels)))  # 创建横轴位置列表。
    plt.bar(xs, values)  # 绘制每个类别的准确率柱状图。
    plt.xticks(xs, labels)  # 设置横轴类别刻度。
    plt.ylim(0, 100)  # 设置纵轴范围为 0 到 100%。
    plt.xlabel("Digit class")  # 设置横轴标题。
    plt.ylabel("Accuracy (%)")  # 设置纵轴标题。
    plt.title(title)  # 设置图标题。
    for i in range(len(values)):  # 遍历每个柱子。
        plt.text(xs[i], values[i] + 1.0, format(values[i], ".1f"), ha="center", fontsize=8)  # 在柱子上方标注数值。
    plt.grid(axis="y", linewidth=0.3, alpha=0.4)  # 添加横向网格线。
    plt.tight_layout()  # 自动调整边距。
    plt.savefig(path, dpi=180)  # 保存 PNG 图片。
    plt.close()  # 关闭当前图。


def write_mnist_grid_matplotlib(path, images, rows, cols, count=100):  # 使用 matplotlib 输出 MNIST 测试样本拼图。
    use = min(count, len(images))  # 确定实际绘制的图片数量。
    grid = int(math.ceil(math.sqrt(use)))  # 计算拼图网格边长。
    gap = 2  # 设置图片之间的白色间隔。
    out_w = grid * cols + (grid + 1) * gap  # 计算拼图宽度。
    out_h = grid * rows + (grid + 1) * gap  # 计算拼图高度。
    canvas = []  # 创建二维画布列表。
    for _ in range(out_h):  # 遍历画布每一行。
        canvas.append([255 for _ in range(out_w)])  # 初始化整行为白色背景。
    for idx in range(use):  # 遍历要绘制的 MNIST 图片。
        gy = idx // grid  # 计算当前图片所在的网格行。
        gx = idx % grid  # 计算当前图片所在的网格列。
        ox = gap + gx * (cols + gap)  # 计算当前图片左上角 x 坐标。
        oy = gap + gy * (rows + gap)  # 计算当前图片左上角 y 坐标。
        img = images[idx]  # 取出当前图片的原始字节。
        for r in range(rows):  # 遍历图片每一行。
            for c in range(cols):  # 遍历图片每一列。
                canvas[oy + r][ox + c] = 255 - img[r * cols + c]  # 反色显示，让数字笔迹为黑色。
    plt.figure(figsize=(7.0, 7.0))  # 创建样本拼图画布。
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=255)  # 使用灰度图显示拼图。
    plt.axis("off")  # 关闭坐标轴，让图片更适合放入报告。
    plt.title("MNIST test samples")  # 设置图标题。
    plt.tight_layout()  # 自动调整边距。
    plt.savefig(path, dpi=180)  # 保存 PNG 图片。
    plt.close()  # 关闭当前图。


def write_summary_visuals(out_dir, logic_results, mnist_results):  # 生成任务六对比分析所需的扩展可视化图像。
    if len(logic_results) > 0:  # 如果有逻辑数据集结果。
        labels = []  # 创建逻辑结果横轴标签。
        values = []  # 创建逻辑结果准确率。
        for ds_name, model_name, acc in logic_results:  # 遍历每个逻辑实验结果。
            labels.append(ds_name + "\n" + model_name.replace("task", "T"))  # 保存数据集和模型名称。
            values.append(acc * 100.0)  # 保存百分制准确率。
        plt.figure(figsize=(max(8.0, len(labels) * 0.62), 4.5))  # 根据柱子数量创建画布。
        xs = list(range(len(labels)))  # 创建横轴位置。
        plt.bar(xs, values)  # 绘制逻辑数据集模型精度柱状图。
        plt.xticks(xs, labels, rotation=45, ha="right", fontsize=8)  # 设置横轴标签并旋转避免重叠。
        plt.ylim(0, 105)  # 设置纵轴范围。
        plt.ylabel("Accuracy (%)")  # 设置纵轴标题。
        plt.title("Logic datasets accuracy comparison")  # 设置图标题。
        plt.grid(axis="y", linewidth=0.3, alpha=0.4)  # 添加横向网格。
        plt.tight_layout()  # 自动调整边距。
        plt.savefig(out_dir / "summary_logic_accuracy.png", dpi=180)  # 保存逻辑数据集对比图。
        plt.close()  # 关闭当前图。
    if len(mnist_results) > 0:  # 如果有 MNIST 结果。
        names = []  # 创建模型名称列表。
        train_values = []  # 创建训练准确率列表。
        test_values = []  # 创建测试准确率列表。
        for item in mnist_results:  # 遍历 MNIST 实验结果。
            _, model_name, train_acc, test_acc, _, _ = item  # 解包结果项。
            names.append(model_name.replace("task", "T"))  # 保存模型名称。
            train_values.append(train_acc * 100.0)  # 保存百分制训练准确率。
            test_values.append(test_acc * 100.0)  # 保存百分制测试准确率。
        xs = list(range(len(names)))  # 创建横轴位置。
        width = 0.36  # 设置并列柱宽度。
        plt.figure(figsize=(8.0, 4.8))  # 创建训练和测试对比画布。
        plt.bar([x - width / 2 for x in xs], train_values, width=width, label="Train")  # 绘制训练准确率柱。
        plt.bar([x + width / 2 for x in xs], test_values, width=width, label="Test")  # 绘制测试准确率柱。
        plt.xticks(xs, names, rotation=25, ha="right")  # 设置横轴模型名称。
        plt.ylim(0, 100)  # 设置纵轴范围。
        plt.ylabel("Accuracy (%)")  # 设置纵轴标题。
        plt.title("MNIST train and test accuracy comparison")  # 设置图标题。
        plt.legend()  # 添加图例。
        plt.grid(axis="y", linewidth=0.3, alpha=0.4)  # 添加横向网格。
        plt.tight_layout()  # 自动调整边距。
        plt.savefig(out_dir / "summary_mnist_train_test_accuracy.png", dpi=180)  # 保存训练测试对比图。
        plt.close()  # 关闭当前图。
        plt.figure(figsize=(8.0, 4.5))  # 创建单独测试精度对比画布。
        plt.bar(xs, test_values)  # 绘制测试准确率柱状图。
        plt.xticks(xs, names, rotation=25, ha="right")  # 设置横轴模型名称。
        plt.ylim(0, 100)  # 设置纵轴范围。
        plt.ylabel("Test accuracy (%)")  # 设置纵轴标题。
        plt.title("MNIST model test accuracy ranking")  # 设置图标题。
        for i in range(len(test_values)):  # 遍历每个模型。
            plt.text(xs[i], test_values[i] + 1.0, format(test_values[i], ".1f"), ha="center", fontsize=8)  # 标注测试准确率数值。
        plt.grid(axis="y", linewidth=0.3, alpha=0.4)  # 添加横向网格。
        plt.tight_layout()  # 自动调整边距。
        plt.savefig(out_dir / "summary_mnist_test_accuracy.png", dpi=180)  # 保存测试精度排序图。
        plt.close()  # 关闭当前图。


# ========================== 实验运行区 ========================== # 分别运行五个任务，并输出任务六所需的精度对比。


def run_logic_experiments(args, out_dir):  # 运行与、或、非、异或四个数据集实验。
    out("========== 第一类数据集：与、或、非、异或 ==========")  # 输出阶段标题。
    results = []  # 创建结果列表。
    datasets = make_logic_datasets()  # 构造四个逻辑数据集。
    for ds_name in ["and", "or", "not", "xor"]:  # 按固定顺序遍历逻辑数据集。
        X, y = datasets[ds_name]  # 取出当前数据集。
        input_dim = len(X[0])  # 获取输入维度。
        class_count = 2  # 逻辑数据集都是二分类。
        models = []  # 创建模型列表。
        models.append(("task1_linear", SoftmaxLinear(input_dim, class_count, lr=0.2, epochs=max(20, args["linear_epochs"]), seed=args["seed"])))  # 添加任务一线性模型。
        models.append(("task2_tree", DecisionTree(class_count, max_depth=3, min_samples=1, max_thresholds=4)))  # 添加任务二决策树。
        models.append(("task3_perceptron", MulticlassPerceptron(input_dim, class_count, lr=0.2, epochs=max(20, args["perceptron_epochs"]), seed=args["seed"])))  # 添加任务三感知机。
        models.append(("task4_one_hidden_nn", MLP(input_dim, [4], class_count, lr=0.1, epochs=max(1000, args["nn_epochs"]), seed=args["seed"])))  # 添加任务四单隐层全连接网络，较长训练用于解决异或非线性问题。
        for model_name, model in models:  # 遍历当前数据集上的每个模型。
            out("开始训练 " + ds_name + " / " + model_name)  # 输出训练开始信息。
            model.train(X, y, ds_name + "_" + model_name)  # 训练模型。
            acc = accuracy(model, X, y)  # 计算训练集精度。
            results.append((ds_name, model_name, acc))  # 保存结果。
            png_path = out_dir / ("logic_" + ds_name + "_" + model_name + ".png")  # 构造决策边界 PNG 路径。
            write_logic_matplotlib(png_path, model, X, y, ds_name + " / " + model_name + " / acc=" + format(acc, ".3f"))  # 使用 matplotlib 输出逻辑数据可视化。
            out("完成 " + ds_name + " / " + model_name + " acc=" + format(acc, ".4f") + " vis=" + str(png_path))  # 输出结果。
    return results  # 返回逻辑实验结果。


def train_and_eval_mnist_model(model_name, model, X_train, y_train, X_test, y_test, out_dir):  # 训练并评估一个 MNIST 模型。
    out("开始训练 MNIST / " + model_name)  # 输出训练开始信息。
    start = time.time()  # 记录开始时间。
    model.train(X_train, y_train, "mnist_" + model_name)  # 训练模型。
    train_acc = accuracy(model, X_train, y_train)  # 计算训练精度。
    test_acc = accuracy(model, X_test, y_test)  # 计算测试精度。
    mat = confusion_matrix(model, X_test, y_test, 10)  # 计算测试集混淆矩阵。
    txt_path = out_dir / ("mnist_confusion_" + model_name + ".txt")  # 构造混淆矩阵文本路径。
    png_path = out_dir / ("mnist_confusion_" + model_name + ".png")  # 构造混淆矩阵 PNG 路径。
    class_acc_path = out_dir / ("mnist_per_class_accuracy_" + model_name + ".png")  # 构造每类数字准确率 PNG 路径。
    write_confusion_txt(txt_path, mat)  # 写出混淆矩阵文本。
    write_confusion_matplotlib(png_path, mat, "MNIST " + model_name + " test_acc=" + format(test_acc, ".4f"))  # 使用 matplotlib 写出混淆矩阵可视化。
    write_per_class_accuracy_matplotlib(class_acc_path, mat, "MNIST per-class accuracy / " + model_name)  # 使用 matplotlib 写出每类数字准确率。
    used = time.time() - start  # 计算耗时。
    out("完成 MNIST / " + model_name + " train_acc=" + format(train_acc, ".4f") + " test_acc=" + format(test_acc, ".4f") + " time=" + format(used, ".1f") + "s")  # 输出评估结果。
    return ("mnist", model_name, train_acc, test_acc, str(txt_path), str(png_path))  # 返回汇总结果。


def run_mnist_experiments(args, out_dir):  # 运行 MNIST 五个任务实验。
    out("========== 第二类数据集：MNIST 手写数字 ==========")  # 输出阶段标题。
    loaded = load_mnist(args["data_dir"], args["train_limit"], args["test_limit"])  # 读取 MNIST 四文件。
    if loaded is None:  # 如果 MNIST 文件不完整。
        return []  # 直接返回空结果。
    X_train, y_train, X_test, y_test, test_images, rows, cols = loaded  # 解包数据。
    out("MNIST train=" + str(len(X_train)) + " test=" + str(len(X_test)) + " feature_dim=" + str(len(X_train[0]) if X_train else 0))  # 输出数据规模。
    sample_path = out_dir / "mnist_test_samples.png"  # 构造测试样本可视化 PNG 路径。
    write_mnist_grid_matplotlib(sample_path, test_images, rows, cols, 100)  # 使用 matplotlib 输出前 100 张测试图的 PNG 拼图。
    out("MNIST 样本可视化已保存：" + str(sample_path))  # 输出可视化路径。
    input_dim = len(X_train[0])  # 获取 MNIST 特征维度。
    results = []  # 创建结果列表。
    linear = SoftmaxLinear(input_dim, 10, lr=0.08, epochs=args["linear_epochs"], seed=args["seed"])  # 创建任务一线性 softmax 模型。
    results.append(train_and_eval_mnist_model("task1_linear", linear, X_train, y_train, X_test, y_test, out_dir))  # 训练并评估任务一。
    tree = DecisionTree(10, max_depth=args["tree_depth"], min_samples=10, max_thresholds=8)  # 创建任务二决策树模型。
    results.append(train_and_eval_mnist_model("task2_tree", tree, X_train, y_train, X_test, y_test, out_dir))  # 训练并评估任务二。
    perceptron = MulticlassPerceptron(input_dim, 10, lr=0.08, epochs=args["perceptron_epochs"], seed=args["seed"])  # 创建任务三多分类感知机。
    results.append(train_and_eval_mnist_model("task3_perceptron", perceptron, X_train, y_train, X_test, y_test, out_dir))  # 训练并评估任务三。
    one_hidden = MLP(input_dim, [64], 10, lr=0.03, epochs=args["nn_epochs"], seed=args["seed"])  # 创建任务四单隐层全连接网络。
    results.append(train_and_eval_mnist_model("task4_one_hidden_nn", one_hidden, X_train, y_train, X_test, y_test, out_dir))  # 训练并评估任务四。
    deep = MLP(input_dim, [96, 48], 10, lr=0.02, epochs=args["deep_epochs"], seed=args["seed"])  # 创建任务五多隐层全连接网络。
    results.append(train_and_eval_mnist_model("task5_deep_nn", deep, X_train, y_train, X_test, y_test, out_dir))  # 训练并评估任务五。
    return results  # 返回 MNIST 实验结果。


def write_summary(out_dir, logic_results, mnist_results):  # 写出任务六需要的总体精度对比文件。
    path = out_dir / "summary_results.csv"  # 构造汇总 CSV 路径。
    lines = []  # 创建 CSV 行列表。
    lines.append("dataset,model,train_or_logic_acc,test_acc,extra_file,extra_vis")  # 写入表头。
    for ds_name, model_name, acc in logic_results:  # 遍历逻辑数据集结果。
        lines.append(ds_name + "," + model_name + "," + format(acc, ".6f") + ",,,")  # 写入逻辑结果行。
    for item in mnist_results:  # 遍历 MNIST 结果。
        ds_name, model_name, train_acc, test_acc, txt_path, svg_path = item  # 解包结果元组。
        lines.append(ds_name + "," + model_name + "," + format(train_acc, ".6f") + "," + format(test_acc, ".6f") + "," + txt_path + "," + svg_path)  # 写入 MNIST 结果行。
    path.write_text("\n".join(lines), encoding="utf-8")  # 写入 CSV 文件。
    out("汇总结果已保存：" + str(path))  # 输出保存路径。
    out("任务六分析建议：线性模型和感知机难以处理异或等非线性边界；决策树能用分裂组合表达异或；单隐层和多隐层神经网络能学习非线性映射；MNIST 上模型效果通常随表达能力增强而提高，但也受训练轮数、特征提取和样本量限制。")  # 输出可直接写入报告的分析提示。


def main():  # 定义程序入口函数。
    args = parse_args()  # 解析命令行参数。
    random.seed(args["seed"])  # 设置全局随机种子。
    out_dir = ensure_dir(args["out_dir"])  # 创建输出目录。
    out("本脚本的核心算法和数据读取只使用 Python 标准库，图片可视化只额外使用 matplotlib。")  # 输出环境说明。
    out("data_dir=" + args["data_dir"] + " out_dir=" + str(out_dir))  # 输出数据目录和输出目录。
    out("提示：把四个 MNIST 解压文件夹或解压文件放到本代码同一目录，或用 --data-dir 指定其所在目录。")  # 在代码相关位置用注释和 stdout 指明四文件放置方法。
    logic_results = []  # 初始化逻辑实验结果。
    mnist_results = []  # 初始化 MNIST 实验结果。
    if not args["mnist_only"]:  # 如果不是只跑 MNIST。
        logic_results = run_logic_experiments(args, out_dir)  # 运行逻辑数据集实验。
    if not args["logic_only"]:  # 如果不是只跑逻辑数据集。
        mnist_results = run_mnist_experiments(args, out_dir)  # 运行 MNIST 实验。
    write_summary(out_dir, logic_results, mnist_results)  # 写出结果汇总。
    write_summary_visuals(out_dir, logic_results, mnist_results)  # 使用 matplotlib 写出任务六精度对比扩展图。
    out("全部任务完成。")  # 输出结束信息。


if __name__ == "__main__":  # 判断是否从命令行直接运行本脚本。
    main()  # 调用主函数。
