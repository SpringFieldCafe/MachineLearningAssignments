# -*- coding: utf-8 -*-

import os
import sys
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


def out_line(message=""):
    sys.stdout.write(str(message) + "\n")


def out_block(lines):
    sys.stdout.write("\n".join(str(item) for item in lines) + "\n")


def get_local_dataset_path():
    current_folder = Path(__file__).resolve().parent

    candidate_names = [
        "iris.data",
        "iris.txt",
    ]

    for name in candidate_names:
        pathlib_guess = current_folder / name
        os_checked_path = os.path.join(str(current_folder), name)

        if pathlib_guess.exists() and os.path.isfile(os_checked_path):
            return pathlib_guess

    for file_item in current_folder.iterdir():
        if file_item.is_file():
            lowered_name = file_item.name.lower()
            if lowered_name.startswith("iris") and lowered_name.endswith((".data", ".txt", ".csv")):
                if os.path.exists(str(file_item)):
                    return file_item

    raise FileNotFoundError("没有找到 iris 数据集，请把 iris 数据文件和本代码放在同一文件夹。")


def calculate_accuracy(answer_list, guess_list):
    correct_counter = 0

    for real_value, predict_value in zip(answer_list, guess_list):
        if real_value == predict_value:
            correct_counter += 1

    if len(answer_list) == 0:
        return 0.0

    return correct_counter / len(answer_list)


def most_common_label(label_collection):
    label_box = {}

    for label in label_collection:
        label_box[label] = label_box.get(label, 0) + 1

    best_label = None
    best_amount = -1

    for label, amount in label_box.items():
        if amount > best_amount:
            best_label = label
            best_amount = amount

    return best_label


def random_cut(dataset, test_ratio, seed_value):
    temporary_bag = dataset[:]
    random.Random(seed_value).shuffle(temporary_bag)

    test_size = int(len(temporary_bag) * test_ratio)

    if test_size <= 0:
        test_size = 1

    testing_part = temporary_bag[:test_size]
    training_part = temporary_bag[test_size:]

    return training_part, testing_part


def load_iris_records():
    iris_path = get_local_dataset_path()
    record_box = []

    with open(iris_path, "r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()

            if line == "":
                continue

            pieces = line.split(",")

            if len(pieces) != 5:
                continue

            numbers = []

            for value in pieces[:4]:
                numbers.append(float(value))

            record_box.append({
                "x": numbers,
                "y": pieces[-1]
            })

    return record_box


def make_watermelon_records():
    table = [
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

    names = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
    result = []

    for row in table:
        feature_map = {}

        for position, name in enumerate(names):
            feature_map[name] = row[position]

        result.append({
            "x": feature_map,
            "y": row[-1]
        })

    return result


class MelonCompass:
    def __init__(self):
        self.direction_score = {}
        self.middle_gate = 0.0
        self.fallback = "否"

    def fit(self, samples):
        labels = [item["y"] for item in samples]
        self.fallback = most_common_label(labels)

        positive_total = labels.count("是")
        negative_total = labels.count("否")

        feature_names = list(samples[0]["x"].keys())
        score_memory = {}

        for feature_name in feature_names:
            value_counter = {}

            for item in samples:
                value = item["x"][feature_name]
                label = item["y"]

                if value not in value_counter:
                    value_counter[value] = {"是": 0, "否": 0}

                value_counter[value][label] += 1

            for value, counter in value_counter.items():
                positive_rate = (counter["是"] + 1) / (positive_total + 2)
                negative_rate = (counter["否"] + 1) / (negative_total + 2)
                score_memory[(feature_name, value)] = math.log(positive_rate / negative_rate)

        self.direction_score = score_memory

        positive_marks = []
        negative_marks = []

        for item in samples:
            mark = self._mark(item["x"])

            if item["y"] == "是":
                positive_marks.append(mark)
            else:
                negative_marks.append(mark)

        if positive_marks and negative_marks:
            left_center = sum(positive_marks) / len(positive_marks)
            right_center = sum(negative_marks) / len(negative_marks)
            self.middle_gate = (left_center + right_center) / 2
        else:
            self.middle_gate = 0.0

    def _mark(self, feature_map):
        total_mark = 0.0

        for key, value in feature_map.items():
            total_mark += self.direction_score.get((key, value), 0.0)

        return total_mark

    def predict_one(self, feature_map):
        mark = self._mark(feature_map)

        if mark >= self.middle_gate:
            return "是"

        return "否"

    def predict(self, feature_rows):
        result = []

        for feature_map in feature_rows:
            result.append(self.predict_one(feature_map))

        return result


class IrisOrbitRule:
    def __init__(self):
        self.anchor = {}
        self.width = {}
        self.label_sequence = []

    def fit(self, samples):
        grouped = {}

        for item in samples:
            grouped.setdefault(item["y"], []).append(item["x"])

        self.label_sequence = list(grouped.keys())

        for label, rows in grouped.items():
            dimension = len(rows[0])
            center = []

            for column in range(dimension):
                column_sum = 0.0

                for row in rows:
                    column_sum += row[column]

                center.append(column_sum / len(rows))

            loose_width = []

            for column in range(dimension):
                gap_sum = 0.0

                for row in rows:
                    gap_sum += abs(row[column] - center[column])

                average_gap = gap_sum / len(rows)

                if average_gap < 1e-9:
                    average_gap = 1.0

                loose_width.append(average_gap)

            self.anchor[label] = center
            self.width[label] = loose_width

    def predict_one(self, feature_row):
        chosen_label = None
        chosen_cost = None

        for label in self.label_sequence:
            center = self.anchor[label]
            loose_width = self.width[label]

            cost = 0.0

            for column in range(len(feature_row)):
                cost += abs(feature_row[column] - center[column]) / loose_width[column]

            if chosen_cost is None or cost < chosen_cost:
                chosen_cost = cost
                chosen_label = label

        return chosen_label

    def predict(self, feature_rows):
        result = []

        for row in feature_rows:
            result.append(self.predict_one(row))

        return result


def label_entropy(labels):
    amount_box = {}

    for label in labels:
        amount_box[label] = amount_box.get(label, 0) + 1

    total = len(labels)
    entropy_value = 0.0

    for amount in amount_box.values():
        rate = amount / total
        entropy_value -= rate * math.log(rate + 1e-12, 2)

    return entropy_value


def divide_by_threshold(samples, column, threshold):
    small_side = []
    large_side = []

    for item in samples:
        if item["x"][column] <= threshold:
            small_side.append(item)
        else:
            large_side.append(item)

    return small_side, large_side


class SmallBranchTree:
    def __init__(self, max_depth=4, min_leaf=3):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.root = None

    def fit(self, samples):
        self.root = self._grow(samples, 0)

    def _grow(self, samples, depth):
        labels = [item["y"] for item in samples]
        default_label = most_common_label(labels)

        if depth >= self.max_depth:
            return {"type": "leaf", "label": default_label}

        if len(set(labels)) == 1:
            return {"type": "leaf", "label": labels[0]}

        if len(samples) <= self.min_leaf:
            return {"type": "leaf", "label": default_label}

        base_entropy = label_entropy(labels)
        best_gain = -1.0
        best_column = None
        best_threshold = None
        best_small = None
        best_large = None

        column_count = len(samples[0]["x"])

        for column in range(column_count):
            values = sorted(set(item["x"][column] for item in samples))

            if len(values) <= 1:
                continue

            thresholds = []

            for index in range(len(values) - 1):
                thresholds.append((values[index] + values[index + 1]) / 2)

            for threshold in thresholds:
                small_side, large_side = divide_by_threshold(samples, column, threshold)

                if len(small_side) < self.min_leaf or len(large_side) < self.min_leaf:
                    continue

                small_entropy = label_entropy([item["y"] for item in small_side])
                large_entropy = label_entropy([item["y"] for item in large_side])

                mixed_entropy = len(small_side) / len(samples) * small_entropy
                mixed_entropy += len(large_side) / len(samples) * large_entropy

                gain = base_entropy - mixed_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_column = column
                    best_threshold = threshold
                    best_small = small_side
                    best_large = large_side

        if best_column is None:
            return {"type": "leaf", "label": default_label}

        return {
            "type": "node",
            "column": best_column,
            "threshold": best_threshold,
            "small": self._grow(best_small, depth + 1),
            "large": self._grow(best_large, depth + 1),
            "default": default_label
        }

    def predict_one(self, row):
        cursor = self.root

        while cursor["type"] != "leaf":
            column = cursor["column"]
            threshold = cursor["threshold"]

            if row[column] <= threshold:
                cursor = cursor["small"]
            else:
                cursor = cursor["large"]

        return cursor["label"]

    def predict(self, feature_rows):
        result = []

        for row in feature_rows:
            result.append(self.predict_one(row))

        return result


def run_three_times(records, model_builder, start_seed, test_ratio):
    score_list = []

    for round_id in range(3):
        seed_value = start_seed + round_id * 31
        train_part, test_part = random_cut(records, test_ratio, seed_value)

        model = model_builder()
        model.fit(train_part)

        feature_rows = [item["x"] for item in test_part]
        real_labels = [item["y"] for item in test_part]
        predicted_labels = model.predict(feature_rows)

        score = calculate_accuracy(real_labels, predicted_labels)
        score_list.append(score)

    return score_list


def show_result(title, scores):
    average_score = sum(scores) / len(scores)

    out_line(title)
    out_line("第1次：{:.2f}%".format(scores[0] * 100))
    out_line("第2次：{:.2f}%".format(scores[1] * 100))
    out_line("第3次：{:.2f}%".format(scores[2] * 100))
    out_line("平均值：{:.2f}%".format(average_score * 100))
    out_line("-" * 45)


def draw_curve(melon_scores, iris_rule_scores, iris_tree_scores):
    round_axis = [1, 2, 3]

    plt.figure(figsize=(8, 5))
    plt.plot(round_axis, melon_scores, marker="o", label="Watermelon Rule")
    plt.plot(round_axis, iris_rule_scores, marker="s", label="Iris Rule")
    plt.plot(round_axis, iris_tree_scores, marker="^", label="Iris Decision Tree")

    plt.title("Accuracy Curve of Three Random Experiments")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.xticks(round_axis)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    save_path = Path(__file__).resolve().parent / "experiment10_accuracy_curve.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return save_path


def main():
    melon_records = make_watermelon_records()
    iris_records = load_iris_records()

    melon_rule_scores = run_three_times(
        records=melon_records,
        model_builder=MelonCompass,
        start_seed=20261001,
        test_ratio=0.3
    )

    iris_rule_scores = run_three_times(
        records=iris_records,
        model_builder=IrisOrbitRule,
        start_seed=20261099,
        test_ratio=0.3
    )

    iris_tree_scores = run_three_times(
        records=iris_records,
        model_builder=lambda: SmallBranchTree(max_depth=4, min_leaf=3),
        start_seed=20261177,
        test_ratio=0.3
    )

    out_line()
    out_line("========== 机器学习实验十：启发式算法实验 ==========")
    out_line()

    show_result("西瓜数据集 2.0：自定义规则分类", melon_rule_scores)
    show_result("鸢尾花数据集：自定义规则分类", iris_rule_scores)
    show_result("鸢尾花数据集：手写决策树分类对比", iris_tree_scores)

    image_path = draw_curve(melon_rule_scores, iris_rule_scores, iris_tree_scores)

    out_line("精度变化曲线已保存：{}".format(image_path))


if __name__ == "__main__":
    main()