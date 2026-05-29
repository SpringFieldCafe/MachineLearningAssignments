# -*- coding: utf-8 -*-
"""
机器学习实验九：决策树 C4.5 算法实验

实验内容：
1. 使用 C4.5 决策树实现西瓜数据集 2.0 的分类，随机十次取平均。
2. 使用 C4.5 决策树实现鸢尾花数据集的分类，随机十次取平均。
3. 画出十次实验的精度变化曲线图。

说明：
1. 本代码不直接调用 sklearn 的决策树，而是手动实现 C4.5 的主要划分逻辑。
2. 输出不使用 print，统一使用 sys.stdout.write。
3. 路径统一使用 pathlib.Path，减少 Windows 和 Linux 路径差异问题。
"""

from pathlib import Path

import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# 用于避免除零错误的小常数。
EPSILON = 1e-12


def configure_stdout() -> None:
    """配置标准输出编码，尽量避免 Windows 终端中文乱码。"""

    # Python 3.7 及以上版本支持 reconfigure。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def write_line(message: str = "") -> None:
    """使用 sys.stdout.write 输出一行内容，避免直接使用 print。"""

    # 将任意内容转换为字符串。
    text = str(message)

    # 写入标准输出。
    sys.stdout.write(text + "\n")

    # 立即刷新输出，方便实时查看训练过程。
    sys.stdout.flush()


@dataclass
class SplitResult:
    """保存候选属性的一次划分结果。"""

    # 当前候选属性的下标。
    feature_index: int

    # 当前候选属性的名称。
    feature_name: str

    # 当前属性是否为连续属性。
    is_continuous: bool

    # 连续属性的二分阈值；离散属性为 None。
    threshold: Optional[float]

    # 当前划分的信息增益。
    info_gain: float

    # 当前划分的划分信息。
    split_info: float

    # 当前划分的增益率。
    gain_ratio: float


@dataclass
class DecisionNode:
    """保存决策树中的一个结点。"""

    # 是否为叶结点。
    is_leaf: bool

    # 当前结点预测类别。
    prediction: str

    # 当前结点默认类别，用于处理未知分支。
    default_label: str

    # 当前结点使用的划分属性下标。
    feature_index: Optional[int] = None

    # 当前结点使用的划分属性名称。
    feature_name: Optional[str] = None

    # 连续属性的二分阈值。
    threshold: Optional[float] = None

    # 当前结点划分属性是否为连续属性。
    is_continuous: bool = False

    # 当前结点的子结点字典。
    children: Dict[Any, "DecisionNode"] = field(default_factory=dict)


class C45DecisionTree:
    """手动实现的 C4.5 决策树，支持离散属性和连续属性。"""

    def __init__(
        self,
        feature_names: Sequence[str],
        feature_types: Sequence[str],
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
    ) -> None:
        """初始化 C4.5 决策树。"""

        # 保存特征名称。
        self.feature_names = list(feature_names)

        # 保存特征类型，categorical 表示离散属性，continuous 表示连续属性。
        self.feature_types = list(feature_types)

        # 保存最大深度，None 表示不限制深度。
        self.max_depth = max_depth

        # 保存最小划分样本数。
        self.min_samples_split = min_samples_split

        # 初始化根结点。
        self.root: Optional[DecisionNode] = None

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> None:
        """训练决策树模型。"""

        # 将标签转成 numpy 数组，便于布尔索引。
        label_array = np.asarray(labels, dtype=object)

        # 初始时全部特征均可作为候选划分属性。
        available_features = list(range(features.shape[1]))

        # 从根结点开始递归建树。
        self.root = self._build_tree(
            features=features,
            labels=label_array,
            available_features=available_features,
            depth=0,
        )

    def predict(self, features: np.ndarray) -> List[str]:
        """对测试样本进行预测。"""

        # 如果模型尚未训练，则抛出异常。
        if self.root is None:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法。")

        # 逐个样本预测类别。
        predictions = [
            self._predict_one(sample=sample, node=self.root)
            for sample in features
        ]

        # 返回预测结果。
        return predictions

    def _build_tree(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        available_features: List[int],
        depth: int,
    ) -> DecisionNode:
        """递归生成决策树。"""

        # 当前结点的多数类标签。
        majority_label = self._majority_label(labels)

        # 情况 1：当前样本全部属于同一类别，直接生成叶结点。
        if len(set(labels.tolist())) == 1:
            return DecisionNode(
                is_leaf=True,
                prediction=str(labels[0]),
                default_label=str(labels[0]),
            )

        # 情况 2：没有可用划分属性，生成多数类叶结点。
        if len(available_features) == 0:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        # 情况 3：达到最大深度，生成多数类叶结点。
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        # 情况 4：样本数过少，生成多数类叶结点。
        if len(labels) < self.min_samples_split:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        # 按 C4.5 规则选择最佳划分属性。
        best_split = self._choose_best_split(
            features=features,
            labels=labels,
            available_features=available_features,
        )

        # 如果找不到有效划分，则生成多数类叶结点。
        if best_split is None or best_split.info_gain <= EPSILON:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        # 创建当前内部结点。
        current_node = DecisionNode(
            is_leaf=False,
            prediction=majority_label,
            default_label=majority_label,
            feature_index=best_split.feature_index,
            feature_name=best_split.feature_name,
            threshold=best_split.threshold,
            is_continuous=best_split.is_continuous,
        )

        # 连续属性按照阈值划分为两个分支。
        if best_split.is_continuous:
            # 取出当前连续属性列。
            current_values = features[:, best_split.feature_index].astype(float)

            # 小于等于阈值的样本进入左分支。
            left_mask = current_values <= float(best_split.threshold)

            # 大于阈值的样本进入右分支。
            right_mask = ~left_mask

            # 连续属性在子树中仍可继续使用。
            next_available_features = list(available_features)

            # 递归生成左分支。
            current_node.children["<="] = self._build_tree(
                features=features[left_mask],
                labels=labels[left_mask],
                available_features=next_available_features,
                depth=depth + 1,
            )

            # 递归生成右分支。
            current_node.children[">"] = self._build_tree(
                features=features[right_mask],
                labels=labels[right_mask],
                available_features=next_available_features,
                depth=depth + 1,
            )

        # 离散属性按照每个属性取值生成一个分支。
        else:
            # 找出当前离散属性出现过的所有取值。
            feature_values = sorted(set(features[:, best_split.feature_index].tolist()))

            # 离散属性使用后，不再进入后续候选属性集合。
            next_available_features = [
                index
                for index in available_features
                if index != best_split.feature_index
            ]

            # 为每个属性取值建立分支。
            for feature_value in feature_values:
                # 找到属性值等于当前取值的样本。
                value_mask = features[:, best_split.feature_index] == feature_value

                # 如果该分支为空，则使用多数类作为叶结点。
                if not np.any(value_mask):
                    current_node.children[feature_value] = DecisionNode(
                        is_leaf=True,
                        prediction=majority_label,
                        default_label=majority_label,
                    )

                # 如果该分支不为空，则递归生成子树。
                else:
                    current_node.children[feature_value] = self._build_tree(
                        features=features[value_mask],
                        labels=labels[value_mask],
                        available_features=next_available_features,
                        depth=depth + 1,
                    )

        # 返回当前结点。
        return current_node

    def _choose_best_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        available_features: List[int],
    ) -> Optional[SplitResult]:
        """按照 C4.5 启发式规则选择最佳划分属性。"""

        # 保存所有候选划分结果。
        candidate_splits: List[SplitResult] = []

        # 遍历所有可用属性。
        for feature_index in available_features:
            # 连续属性需要枚举候选阈值。
            if self.feature_types[feature_index] == "continuous":
                split_result = self._best_continuous_split(
                    features=features,
                    labels=labels,
                    feature_index=feature_index,
                )

            # 离散属性直接按属性取值划分。
            else:
                split_result = self._categorical_split(
                    features=features,
                    labels=labels,
                    feature_index=feature_index,
                )

            # 保存有效划分结果。
            if split_result is not None:
                candidate_splits.append(split_result)

        # 没有候选划分时返回 None。
        if len(candidate_splits) == 0:
            return None

        # C4.5 第一步：计算候选属性的信息增益平均值。
        average_info_gain = (
            sum(item.info_gain for item in candidate_splits)
            / len(candidate_splits)
        )

        # C4.5 第二步：保留信息增益高于平均水平的属性。
        high_gain_splits = [
            item
            for item in candidate_splits
            if item.info_gain > average_info_gain and item.split_info > EPSILON
        ]

        # 如果没有属性高于平均水平，则退化为从全部候选属性中选择。
        valid_splits = high_gain_splits if len(high_gain_splits) > 0 else candidate_splits

        # C4.5 第三步：选择增益率最高的属性。
        return max(valid_splits, key=lambda item: (item.gain_ratio, item.info_gain))

    def _categorical_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_index: int,
    ) -> SplitResult:
        """计算离散属性的信息增益和增益率。"""

        # 取出当前离散属性列。
        feature_values = features[:, feature_index]

        # 计算划分前的信息熵。
        base_entropy = self._entropy(labels)

        # 初始化条件熵。
        conditional_entropy = 0.0

        # 初始化划分信息。
        split_info = 0.0

        # 遍历当前属性的所有取值。
        for feature_value in set(feature_values.tolist()):
            # 找到当前属性取值对应的样本。
            value_mask = feature_values == feature_value

            # 计算该分支样本比例。
            value_probability = float(np.mean(value_mask))

            # 累加条件熵。
            conditional_entropy += value_probability * self._entropy(labels[value_mask])

            # 累加划分信息。
            split_info -= value_probability * math.log2(value_probability)

        # 信息增益等于划分前熵减去划分后条件熵。
        info_gain = base_entropy - conditional_entropy

        # 增益率等于信息增益除以划分信息。
        gain_ratio = info_gain / (split_info + EPSILON)

        # 返回当前离散属性的划分结果。
        return SplitResult(
            feature_index=feature_index,
            feature_name=self.feature_names[feature_index],
            is_continuous=False,
            threshold=None,
            info_gain=info_gain,
            split_info=split_info,
            gain_ratio=gain_ratio,
        )

    def _best_continuous_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_index: int,
    ) -> Optional[SplitResult]:
        """枚举连续属性候选阈值，寻找最佳二分划分。"""

        # 将当前属性列转换为浮点数组。
        feature_values = features[:, feature_index].astype(float)

        # 获取有序唯一取值。
        unique_values = sorted(set(feature_values.tolist()))

        # 如果只有一个取值，则无法划分。
        if len(unique_values) <= 1:
            return None

        # 连续属性候选阈值取相邻取值中点。
        candidate_thresholds = [
            (unique_values[index] + unique_values[index + 1]) / 2.0
            for index in range(len(unique_values) - 1)
        ]

        # 计算划分前的信息熵。
        base_entropy = self._entropy(labels)

        # 初始化最佳结果。
        best_result: Optional[SplitResult] = None

        # 遍历所有候选阈值。
        for threshold in candidate_thresholds:
            # 小于等于阈值的样本。
            left_mask = feature_values <= threshold

            # 大于阈值的样本。
            right_mask = ~left_mask

            # 跳过空分支。
            if not np.any(left_mask) or not np.any(right_mask):
                continue

            # 计算左分支样本比例。
            left_probability = float(np.mean(left_mask))

            # 计算右分支样本比例。
            right_probability = 1.0 - left_probability

            # 计算条件熵。
            conditional_entropy = (
                left_probability * self._entropy(labels[left_mask])
                + right_probability * self._entropy(labels[right_mask])
            )

            # 计算信息增益。
            info_gain = base_entropy - conditional_entropy

            # 计算划分信息。
            split_info = -(
                left_probability * math.log2(left_probability)
                + right_probability * math.log2(right_probability)
            )

            # 计算增益率。
            gain_ratio = info_gain / (split_info + EPSILON)

            # 保存当前阈值对应的划分结果。
            current_result = SplitResult(
                feature_index=feature_index,
                feature_name=self.feature_names[feature_index],
                is_continuous=True,
                threshold=threshold,
                info_gain=info_gain,
                split_info=split_info,
                gain_ratio=gain_ratio,
            )

            # 根据增益率和信息增益更新最佳结果。
            if best_result is None:
                best_result = current_result
            elif (current_result.gain_ratio, current_result.info_gain) > (
                best_result.gain_ratio,
                best_result.info_gain,
            ):
                best_result = current_result

        # 返回连续属性的最佳划分结果。
        return best_result

    def _predict_one(self, sample: np.ndarray, node: DecisionNode) -> str:
        """对单条样本进行预测。"""

        # 到达叶结点，直接返回类别。
        if node.is_leaf:
            return node.prediction

        # 如果结点缺少特征下标，则返回默认类别。
        if node.feature_index is None:
            return node.default_label

        # 连续属性按照阈值决定分支。
        if node.is_continuous:
            if float(sample[node.feature_index]) <= float(node.threshold):
                branch_key = "<="
            else:
                branch_key = ">"

        # 离散属性按照属性取值决定分支。
        else:
            branch_key = sample[node.feature_index]

        # 如果遇到训练集中没有出现过的分支，则返回默认类别。
        if branch_key not in node.children:
            return node.default_label

        # 递归进入子结点。
        return self._predict_one(sample=sample, node=node.children[branch_key])

    @staticmethod
    def _entropy(labels: Sequence[str]) -> float:
        """计算信息熵。"""

        # 样本为空时熵为 0。
        if len(labels) == 0:
            return 0.0

        # 统计类别出现次数。
        label_counter = Counter(labels)

        # 统计样本总数。
        sample_count = len(labels)

        # 初始化熵。
        entropy_value = 0.0

        # 按信息熵公式累加。
        for count in label_counter.values():
            probability = count / sample_count
            entropy_value -= probability * math.log2(probability)

        # 返回熵。
        return entropy_value

    @staticmethod
    def _majority_label(labels: Sequence[str]) -> str:
        """返回样本集合中的多数类。"""

        # 统计类别频数。
        counter = Counter(labels)

        # 按频数降序、标签升序排序，保证结果稳定。
        majority_item = sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0])),
        )[0]

        # 返回多数类标签。
        return str(majority_item[0])


def load_watermelon_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """构造西瓜数据集 2.0。"""

    # 定义西瓜数据集 2.0 的特征名称。
    feature_names = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]

    # 西瓜数据集 2.0 的六个属性均为离散属性。
    feature_types = ["categorical"] * len(feature_names)

    # 每行数据依次为：色泽、根蒂、敲声、纹理、脐部、触感、好瓜。
    raw_samples = [
        ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
        ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
        ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
        ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "是"],
        ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", "是"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", "是"],
        ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
        ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", "否"],
        ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", "否"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", "否"],
        ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", "否"],
        ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"],
        ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "否"],
        ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", "否"],
        ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
    ]

    # 前六列作为输入特征。
    features = np.asarray([sample[:-1] for sample in raw_samples], dtype=object)

    # 最后一列作为类别标签。
    labels = np.asarray([sample[-1] for sample in raw_samples], dtype=object)

    # 返回特征、标签、特征名称和特征类型。
    return features, labels, feature_names, feature_types


def load_iris_dataset(iris_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """读取 iris.data 鸢尾花数据集。"""

    # 判断文件是否存在，方便发现路径错误。
    if not iris_path.exists():
        raise FileNotFoundError(f"没有找到 iris.data 文件：{iris_path}")

    # 鸢尾花数据集的四个连续属性名称。
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # 四个属性均为连续属性。
    feature_types = ["continuous"] * len(feature_names)

    # 保存特征数据。
    feature_rows: List[List[float]] = []

    # 保存标签数据。
    label_rows: List[str] = []

    # 使用 Path.open 打开数据文件。
    with iris_path.open("r", encoding="utf-8") as file_object:
        # 使用 csv 读取逗号分隔数据。
        csv_reader = csv.reader(file_object)

        # 逐行读取数据。
        for row in csv_reader:
            # 跳过空行。
            if len(row) == 0:
                continue

            # 标准 iris.data 每行有 5 列。
            if len(row) != 5:
                raise ValueError(f"iris.data 存在非法行：{row}")

            # 前四列是连续数值特征。
            feature_rows.append([float(value) for value in row[:4]])

            # 第五列是类别标签。
            label_rows.append(row[4])

    # 返回特征、标签、特征名称和特征类型。
    return (
        np.asarray(feature_rows, dtype=float),
        np.asarray(label_rows, dtype=object),
        feature_names,
        feature_types,
    )


def stratified_train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_ratio: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层随机划分训练集和测试集。"""

    # 创建随机数生成器，保证实验可复现。
    random_generator = random.Random(random_seed)

    # 按类别保存样本下标。
    label_to_indices: Dict[str, List[int]] = defaultdict(list)

    # 遍历所有标签。
    for sample_index, label in enumerate(labels):
        label_to_indices[str(label)].append(sample_index)

    # 保存训练集下标。
    train_indices: List[int] = []

    # 保存测试集下标。
    test_indices: List[int] = []

    # 每个类别单独划分，避免类别分布偏移过大。
    for class_indices in label_to_indices.values():
        # 复制当前类别下标。
        shuffled_indices = list(class_indices)

        # 打乱当前类别下标。
        random_generator.shuffle(shuffled_indices)

        # 计算当前类别测试样本数。
        test_count = max(1, int(round(len(shuffled_indices) * test_ratio)))

        # 至少保留一个训练样本。
        test_count = min(test_count, len(shuffled_indices) - 1)

        # 分配测试集。
        test_indices.extend(shuffled_indices[:test_count])

        # 分配训练集。
        train_indices.extend(shuffled_indices[test_count:])

    # 打乱训练集下标。
    random_generator.shuffle(train_indices)

    # 打乱测试集下标。
    random_generator.shuffle(test_indices)

    # 返回划分结果。
    return (
        features[train_indices],
        features[test_indices],
        labels[train_indices],
        labels[test_indices],
    )


def calculate_accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """计算分类准确率。"""

    # 统计预测正确数量。
    correct_count = sum(
        str(true_label) == str(predicted_label)
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )

    # 返回准确率。
    return correct_count / len(true_labels)


def run_ten_random_experiments(
    dataset_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    feature_types: List[str],
    test_ratio: float,
    seed_start: int,
) -> List[float]:
    """对一个数据集进行十次随机实验。"""

    # 保存十次实验的精度。
    accuracy_list: List[float] = []

    # 生成十个随机种子。
    random_seeds = list(range(seed_start, seed_start + 10))

    # 逐次运行实验。
    for experiment_index, random_seed in enumerate(random_seeds, start=1):
        # 划分训练集和测试集。
        train_features, test_features, train_labels, test_labels = stratified_train_test_split(
            features=features,
            labels=labels,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        # 创建 C4.5 决策树模型。
        decision_tree = C45DecisionTree(
            feature_names=feature_names,
            feature_types=feature_types,
            max_depth=None,
            min_samples_split=2,
        )

        # 训练模型。
        decision_tree.fit(train_features, train_labels)

        # 预测测试集。
        predicted_labels = decision_tree.predict(test_features)

        # 计算当前实验精度。
        current_accuracy = calculate_accuracy(test_labels, predicted_labels)

        # 保存当前实验精度。
        accuracy_list.append(current_accuracy)

        # 输出当前实验结果。
        write_line(
            f"{dataset_name} | 第 {experiment_index:02d} 次 | "
            f"随机种子={random_seed} | "
            f"测试样本数={len(test_labels)} | "
            f"精度={current_accuracy:.4f}"
        )

    # 返回十次实验精度。
    return accuracy_list


def save_accuracy_csv(
    output_path: Path,
    watermelon_accuracies: Sequence[float],
    iris_accuracies: Sequence[float],
) -> None:
    """保存十次实验精度到 CSV 文件。"""

    # 使用 utf-8-sig，方便 Excel 正确显示中文。
    with output_path.open("w", newline="", encoding="utf-8-sig") as file_object:
        # 创建 CSV 写入器。
        csv_writer = csv.writer(file_object)

        # 写入表头。
        csv_writer.writerow(["实验次数", "西瓜数据集2.0精度", "鸢尾花数据集精度"])

        # 写入每次实验结果。
        for experiment_index, pair in enumerate(
            zip(watermelon_accuracies, iris_accuracies),
            start=1,
        ):
            watermelon_accuracy, iris_accuracy = pair

            csv_writer.writerow(
                [
                    experiment_index,
                    f"{watermelon_accuracy:.4f}",
                    f"{iris_accuracy:.4f}",
                ]
            )

        # 写入平均精度。
        csv_writer.writerow(
            [
                "平均精度",
                f"{float(np.mean(watermelon_accuracies)):.4f}",
                f"{float(np.mean(iris_accuracies)):.4f}",
            ]
        )


def plot_accuracy_curve(
    output_path: Path,
    watermelon_accuracies: Sequence[float],
    iris_accuracies: Sequence[float],
) -> None:
    """绘制十次实验精度变化曲线。"""

    # 设置中文字体候选，减少图中中文乱码。
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    # 解决负号显示问题。
    plt.rcParams["axes.unicode_minus"] = False

    # 横坐标为实验次数。
    experiment_indices = list(range(1, 11))

    # 创建画布。
    plt.figure(figsize=(9, 5))

    # 绘制西瓜数据集精度曲线。
    plt.plot(
        experiment_indices,
        watermelon_accuracies,
        marker="o",
        label="西瓜数据集2.0",
    )

    # 绘制鸢尾花数据集精度曲线。
    plt.plot(
        experiment_indices,
        iris_accuracies,
        marker="s",
        label="鸢尾花数据集",
    )

    # 设置横轴标签。
    plt.xlabel("随机实验次数")

    # 设置纵轴标签。
    plt.ylabel("分类精度")

    # 设置标题。
    plt.title("C4.5 决策树十次随机实验精度变化曲线")

    # 设置横坐标刻度。
    plt.xticks(experiment_indices)

    # 设置纵坐标范围。
    plt.ylim(0.0, 1.05)

    # 显示网格线。
    plt.grid(True, linestyle="--", alpha=0.5)

    # 显示图例。
    plt.legend()

    # 自动调整布局。
    plt.tight_layout()

    # 保存图像。
    plt.savefig(output_path, dpi=300)

    # 关闭图像。
    plt.close()


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    # 创建参数解析器。
    parser = argparse.ArgumentParser(description="机器学习实验九：C4.5 决策树分类实验")

    # 设置 iris.data 路径。
    parser.add_argument(
        "--iris_path",
        type=Path,
        default=Path("iris.data"),
        help="iris.data 文件路径",
    )

    # 设置输出目录。
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("result"),
        help="实验结果输出目录",
    )

    # 设置测试集比例。
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.3,
        help="测试集比例",
    )

    # 设置随机种子起点。
    parser.add_argument(
        "--seed_start",
        type=int,
        default=2024,
        help="十次随机实验的起始随机种子",
    )

    # 返回参数。
    return parser.parse_args()


def main() -> None:
    """主函数。"""

    # 配置标准输出编码。
    configure_stdout()

    # 解析命令行参数。
    args = parse_arguments()

    # 获取输出目录。
    output_dir: Path = args.output_dir

    # 如果输出目录不存在，则自动创建。
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取西瓜数据集 2.0。
    watermelon_features, watermelon_labels, watermelon_feature_names, watermelon_feature_types = (
        load_watermelon_dataset()
    )

    # 读取鸢尾花数据集。
    iris_features, iris_labels, iris_feature_names, iris_feature_types = load_iris_dataset(
        args.iris_path
    )

    # 输出实验基本信息。
    write_line("=" * 70)
    write_line("机器学习实验九：C4.5 决策树算法实验")
    write_line(f"当前工作目录：{Path.cwd()}")
    write_line(f"iris.data 路径：{args.iris_path.resolve()}")
    write_line(f"结果输出目录：{output_dir.resolve()}")
    write_line(f"西瓜数据集2.0样本数：{len(watermelon_labels)}")
    write_line(f"鸢尾花数据集样本数：{len(iris_labels)}")
    write_line(f"测试集比例：{args.test_ratio}")
    write_line("随机实验次数：10")
    write_line("=" * 70)

    # 对西瓜数据集进行十次随机实验。
    watermelon_accuracies = run_ten_random_experiments(
        dataset_name="西瓜数据集2.0",
        features=watermelon_features,
        labels=watermelon_labels,
        feature_names=watermelon_feature_names,
        feature_types=watermelon_feature_types,
        test_ratio=args.test_ratio,
        seed_start=args.seed_start,
    )

    # 输出分隔线。
    write_line("-" * 70)

    # 对鸢尾花数据集进行十次随机实验。
    iris_accuracies = run_ten_random_experiments(
        dataset_name="鸢尾花数据集",
        features=iris_features,
        labels=iris_labels,
        feature_names=iris_feature_names,
        feature_types=iris_feature_types,
        test_ratio=args.test_ratio,
        seed_start=args.seed_start,
    )

    # 计算平均精度。
    watermelon_mean_accuracy = float(np.mean(watermelon_accuracies))
    iris_mean_accuracy = float(np.mean(iris_accuracies))

    # 设置结果保存路径。
    accuracy_csv_path = output_dir / "accuracy_results.csv"
    accuracy_plot_path = output_dir / "accuracy_curve.png"

    # 保存 CSV 结果表。
    save_accuracy_csv(
        output_path=accuracy_csv_path,
        watermelon_accuracies=watermelon_accuracies,
        iris_accuracies=iris_accuracies,
    )

    # 保存精度变化曲线图。
    plot_accuracy_curve(
        output_path=accuracy_plot_path,
        watermelon_accuracies=watermelon_accuracies,
        iris_accuracies=iris_accuracies,
    )

    # 输出最终结果。
    write_line("=" * 70)
    write_line(f"西瓜数据集2.0十次平均精度：{watermelon_mean_accuracy:.4f}")
    write_line(f"鸢尾花数据集十次平均精度：{iris_mean_accuracy:.4f}")
    write_line(f"精度结果表已保存：{accuracy_csv_path.resolve()}")
    write_line(f"精度变化曲线已保存：{accuracy_plot_path.resolve()}")
    write_line("=" * 70)


if __name__ == "__main__":
    main()