# -*- coding: utf-8 -*-
"""
Machine Learning Experiment 9: C4.5 Decision Tree

Task:
1. Use C4.5 decision tree to classify Watermelon Dataset 2.0.
2. Use C4.5 decision tree to classify Iris dataset.
3. Repeat random experiments 10 times and draw accuracy curves.

Path rule:
1. iris.data is loaded from the same folder as this Python file by default.
2. result files are saved into the result folder beside this Python file.
3. pathlib.Path is used to reduce path problems.
"""

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
from pathlib import Path

EPSILON = 1e-12

assignment9_dir = Path(__file__).resolve().parent
iris_path = assignment9_dir / "iris.data"
default_output_dir = assignment9_dir / "result"


def configure_stdout() -> None:
    """Configure stdout encoding."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def write_line(message: str = "") -> None:
    """Write one line by sys.stdout.write."""

    sys.stdout.write(str(message) + "\n")
    sys.stdout.flush()


@dataclass
class SplitResult:
    """Store split result of one candidate feature."""

    feature_index: int
    feature_name: str
    is_continuous: bool
    threshold: Optional[float]
    info_gain: float
    split_info: float
    gain_ratio: float


@dataclass
class DecisionNode:
    """Store one node in decision tree."""

    is_leaf: bool
    prediction: str
    default_label: str
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    is_continuous: bool = False
    children: Dict[Any, "DecisionNode"] = field(default_factory=dict)


class C45DecisionTree:
    """Manual C4.5 decision tree implementation."""

    def __init__(
        self,
        feature_names: Sequence[str],
        feature_types: Sequence[str],
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
    ) -> None:
        """Initialize decision tree."""

        self.feature_names = list(feature_names)
        self.feature_types = list(feature_types)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[DecisionNode] = None

    def fit(self, features: np.ndarray, labels: Sequence[str]) -> None:
        """Train decision tree."""

        label_array = np.asarray(labels, dtype=object)
        available_features = list(range(features.shape[1]))

        self.root = self._build_tree(
            features=features,
            labels=label_array,
            available_features=available_features,
            depth=0,
        )

    def predict(self, features: np.ndarray) -> List[str]:
        """Predict labels."""

        if self.root is None:
            raise RuntimeError("Model is not fitted. Please call fit first.")

        return [
            self._predict_one(sample=sample, node=self.root)
            for sample in features
        ]

    def _build_tree(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        available_features: List[int],
        depth: int,
    ) -> DecisionNode:
        """Build decision tree recursively."""

        majority_label = self._majority_label(labels)

        if len(set(labels.tolist())) == 1:
            return DecisionNode(
                is_leaf=True,
                prediction=str(labels[0]),
                default_label=str(labels[0]),
            )

        if len(available_features) == 0:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        if len(labels) < self.min_samples_split:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        best_split = self._choose_best_split(
            features=features,
            labels=labels,
            available_features=available_features,
        )

        if best_split is None or best_split.info_gain <= EPSILON:
            return DecisionNode(
                is_leaf=True,
                prediction=majority_label,
                default_label=majority_label,
            )

        current_node = DecisionNode(
            is_leaf=False,
            prediction=majority_label,
            default_label=majority_label,
            feature_index=best_split.feature_index,
            feature_name=best_split.feature_name,
            threshold=best_split.threshold,
            is_continuous=best_split.is_continuous,
        )

        if best_split.is_continuous:
            current_values = features[:, best_split.feature_index].astype(float)
            left_mask = current_values <= float(best_split.threshold)
            right_mask = ~left_mask
            next_available_features = list(available_features)

            current_node.children["<="] = self._build_tree(
                features=features[left_mask],
                labels=labels[left_mask],
                available_features=next_available_features,
                depth=depth + 1,
            )

            current_node.children[">"] = self._build_tree(
                features=features[right_mask],
                labels=labels[right_mask],
                available_features=next_available_features,
                depth=depth + 1,
            )

        else:
            feature_values = sorted(set(features[:, best_split.feature_index].tolist()))

            next_available_features = [
                index
                for index in available_features
                if index != best_split.feature_index
            ]

            for feature_value in feature_values:
                value_mask = features[:, best_split.feature_index] == feature_value

                if not np.any(value_mask):
                    current_node.children[feature_value] = DecisionNode(
                        is_leaf=True,
                        prediction=majority_label,
                        default_label=majority_label,
                    )
                else:
                    current_node.children[feature_value] = self._build_tree(
                        features=features[value_mask],
                        labels=labels[value_mask],
                        available_features=next_available_features,
                        depth=depth + 1,
                    )

        return current_node

    def _choose_best_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        available_features: List[int],
    ) -> Optional[SplitResult]:
        """Choose best split by C4.5 rule."""

        candidate_splits: List[SplitResult] = []

        for feature_index in available_features:
            if self.feature_types[feature_index] == "continuous":
                split_result = self._best_continuous_split(
                    features=features,
                    labels=labels,
                    feature_index=feature_index,
                )
            else:
                split_result = self._categorical_split(
                    features=features,
                    labels=labels,
                    feature_index=feature_index,
                )

            if split_result is not None:
                candidate_splits.append(split_result)

        if len(candidate_splits) == 0:
            return None

        average_info_gain = (
            sum(item.info_gain for item in candidate_splits)
            / len(candidate_splits)
        )

        high_gain_splits = [
            item
            for item in candidate_splits
            if item.info_gain > average_info_gain and item.split_info > EPSILON
        ]

        valid_splits = high_gain_splits if len(high_gain_splits) > 0 else candidate_splits

        return max(valid_splits, key=lambda item: (item.gain_ratio, item.info_gain))

    def _categorical_split(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_index: int,
    ) -> SplitResult:
        """Calculate information gain and gain ratio for categorical feature."""

        feature_values = features[:, feature_index]
        base_entropy = self._entropy(labels)
        conditional_entropy = 0.0
        split_info = 0.0

        for feature_value in set(feature_values.tolist()):
            value_mask = feature_values == feature_value
            value_probability = float(np.mean(value_mask))
            conditional_entropy += value_probability * self._entropy(labels[value_mask])
            split_info -= value_probability * math.log2(value_probability)

        info_gain = base_entropy - conditional_entropy
        gain_ratio = info_gain / (split_info + EPSILON)

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
        """Find best threshold for continuous feature."""

        feature_values = features[:, feature_index].astype(float)
        unique_values = sorted(set(feature_values.tolist()))

        if len(unique_values) <= 1:
            return None

        candidate_thresholds = [
            (unique_values[index] + unique_values[index + 1]) / 2.0
            for index in range(len(unique_values) - 1)
        ]

        base_entropy = self._entropy(labels)
        best_result: Optional[SplitResult] = None

        for threshold in candidate_thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask

            if not np.any(left_mask) or not np.any(right_mask):
                continue

            left_probability = float(np.mean(left_mask))
            right_probability = 1.0 - left_probability

            conditional_entropy = (
                left_probability * self._entropy(labels[left_mask])
                + right_probability * self._entropy(labels[right_mask])
            )

            info_gain = base_entropy - conditional_entropy

            split_info = -(
                left_probability * math.log2(left_probability)
                + right_probability * math.log2(right_probability)
            )

            gain_ratio = info_gain / (split_info + EPSILON)

            current_result = SplitResult(
                feature_index=feature_index,
                feature_name=self.feature_names[feature_index],
                is_continuous=True,
                threshold=threshold,
                info_gain=info_gain,
                split_info=split_info,
                gain_ratio=gain_ratio,
            )

            if best_result is None:
                best_result = current_result
            elif (current_result.gain_ratio, current_result.info_gain) > (
                best_result.gain_ratio,
                best_result.info_gain,
            ):
                best_result = current_result

        return best_result

    def _predict_one(self, sample: np.ndarray, node: DecisionNode) -> str:
        """Predict one sample."""

        if node.is_leaf:
            return node.prediction

        if node.feature_index is None:
            return node.default_label

        if node.is_continuous:
            if float(sample[node.feature_index]) <= float(node.threshold):
                branch_key = "<="
            else:
                branch_key = ">"
        else:
            branch_key = sample[node.feature_index]

        if branch_key not in node.children:
            return node.default_label

        return self._predict_one(sample=sample, node=node.children[branch_key])

    @staticmethod
    def _entropy(labels: Sequence[str]) -> float:
        """Calculate entropy."""

        if len(labels) == 0:
            return 0.0

        label_counter = Counter(labels)
        sample_count = len(labels)
        entropy_value = 0.0

        for count in label_counter.values():
            probability = count / sample_count
            entropy_value -= probability * math.log2(probability)

        return entropy_value

    @staticmethod
    def _majority_label(labels: Sequence[str]) -> str:
        """Return majority label."""

        counter = Counter(labels)

        majority_item = sorted(
            counter.items(),
            key=lambda item: (-item[1], str(item[0])),
        )[0]

        return str(majority_item[0])


def load_watermelon_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load Watermelon Dataset 2.0."""

    feature_names = ["color", "root", "sound", "texture", "navel", "touch"]
    feature_types = ["categorical"] * len(feature_names)

    raw_samples = [
        ["green", "curl", "dull", "clear", "sunken", "hard", "yes"],
        ["black", "curl", "muffled", "clear", "sunken", "hard", "yes"],
        ["black", "curl", "dull", "clear", "sunken", "hard", "yes"],
        ["green", "curl", "muffled", "clear", "sunken", "hard", "yes"],
        ["white", "curl", "dull", "clear", "sunken", "hard", "yes"],
        ["green", "slight", "dull", "clear", "slight", "soft", "yes"],
        ["black", "slight", "dull", "blur", "slight", "soft", "yes"],
        ["black", "slight", "dull", "clear", "slight", "hard", "yes"],
        ["black", "slight", "muffled", "blur", "slight", "hard", "no"],
        ["green", "stiff", "clear_sound", "clear", "flat", "soft", "no"],
        ["white", "stiff", "clear_sound", "blur", "flat", "hard", "no"],
        ["white", "curl", "dull", "blur", "flat", "soft", "no"],
        ["green", "slight", "dull", "blur", "sunken", "hard", "no"],
        ["white", "slight", "muffled", "blur", "sunken", "hard", "no"],
        ["black", "slight", "dull", "clear", "slight", "soft", "no"],
        ["white", "curl", "dull", "blur", "flat", "hard", "no"],
        ["green", "curl", "muffled", "blur", "slight", "hard", "no"],
    ]

    features = np.asarray([sample[:-1] for sample in raw_samples], dtype=object)
    labels = np.asarray([sample[-1] for sample in raw_samples], dtype=object)

    return features, labels, feature_names, feature_types


def load_iris_dataset(input_iris_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load Iris dataset from iris.data."""

    if not input_iris_path.exists():
        raise FileNotFoundError(f"iris.data not found: {input_iris_path}")

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    feature_types = ["continuous"] * len(feature_names)

    feature_rows: List[List[float]] = []
    label_rows: List[str] = []

    with input_iris_path.open("r", encoding="utf-8") as file_object:
        csv_reader = csv.reader(file_object)

        for row in csv_reader:
            if len(row) == 0:
                continue

            if len(row) != 5:
                raise ValueError(f"Invalid row in iris.data: {row}")

            feature_rows.append([float(value) for value in row[:4]])
            label_rows.append(row[4])

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
    """Split train and test data by label."""

    random_generator = random.Random(random_seed)
    label_to_indices: Dict[str, List[int]] = defaultdict(list)

    for sample_index, label in enumerate(labels):
        label_to_indices[str(label)].append(sample_index)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for class_indices in label_to_indices.values():
        shuffled_indices = list(class_indices)
        random_generator.shuffle(shuffled_indices)

        test_count = max(1, int(round(len(shuffled_indices) * test_ratio)))
        test_count = min(test_count, len(shuffled_indices) - 1)

        test_indices.extend(shuffled_indices[:test_count])
        train_indices.extend(shuffled_indices[test_count:])

    random_generator.shuffle(train_indices)
    random_generator.shuffle(test_indices)

    return (
        features[train_indices],
        features[test_indices],
        labels[train_indices],
        labels[test_indices],
    )


def calculate_accuracy(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> float:
    """Calculate accuracy."""

    correct_count = sum(
        str(true_label) == str(predicted_label)
        for true_label, predicted_label in zip(true_labels, predicted_labels)
    )

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
    """Run 10 random experiments."""

    accuracy_list: List[float] = []
    random_seeds = list(range(seed_start, seed_start + 10))

    for experiment_index, random_seed in enumerate(random_seeds, start=1):
        train_features, test_features, train_labels, test_labels = stratified_train_test_split(
            features=features,
            labels=labels,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        decision_tree = C45DecisionTree(
            feature_names=feature_names,
            feature_types=feature_types,
            max_depth=None,
            min_samples_split=2,
        )

        decision_tree.fit(train_features, train_labels)
        predicted_labels = decision_tree.predict(test_features)
        current_accuracy = calculate_accuracy(test_labels, predicted_labels)
        accuracy_list.append(current_accuracy)

        write_line(
            f"{dataset_name} | run {experiment_index:02d} | "
            f"seed={random_seed} | "
            f"test_size={len(test_labels)} | "
            f"accuracy={current_accuracy:.4f}"
        )

    return accuracy_list


def save_accuracy_csv(
    output_path: Path,
    watermelon_accuracies: Sequence[float],
    iris_accuracies: Sequence[float],
) -> None:
    """Save accuracy results to CSV."""

    with output_path.open("w", newline="", encoding="utf-8-sig") as file_object:
        csv_writer = csv.writer(file_object)

        csv_writer.writerow(["run", "watermelon_accuracy", "iris_accuracy"])

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

        csv_writer.writerow(
            [
                "mean",
                f"{float(np.mean(watermelon_accuracies)):.4f}",
                f"{float(np.mean(iris_accuracies)):.4f}",
            ]
        )


def plot_accuracy_curve(
    output_path: Path,
    watermelon_accuracies: Sequence[float],
    iris_accuracies: Sequence[float],
) -> None:
    """Plot accuracy curve."""

    experiment_indices = list(range(1, 11))

    plt.figure(figsize=(9, 5))

    plt.plot(
        experiment_indices,
        watermelon_accuracies,
        marker="o",
        label="Watermelon 2.0",
    )

    plt.plot(
        experiment_indices,
        iris_accuracies,
        marker="s",
        label="Iris",
    )

    plt.xlabel("Random Run")
    plt.ylabel("Accuracy")
    plt.title("C4.5 Decision Tree Accuracy Curve")
    plt.xticks(experiment_indices)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="C4.5 decision tree experiment")

    parser.add_argument(
        "--iris_path",
        type=Path,
        default=iris_path,
        help="Path of iris.data",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_output_dir,
        help="Output folder",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.3,
        help="Test set ratio",
    )

    parser.add_argument(
        "--seed_start",
        type=int,
        default=2024,
        help="Start seed of 10 random runs",
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""

    configure_stdout()
    args = parse_arguments()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    watermelon_features, watermelon_labels, watermelon_feature_names, watermelon_feature_types = (
        load_watermelon_dataset()
    )

    iris_features, iris_labels, iris_feature_names, iris_feature_types = load_iris_dataset(
        args.iris_path
    )

    write_line("=" * 70)
    write_line("Machine Learning Experiment 9: C4.5 Decision Tree")
    write_line(f"Current working directory: {Path.cwd()}")
    write_line(f"Program directory: {assignment9_dir}")
    write_line(f"Iris data path: {args.iris_path.resolve()}")
    write_line(f"Output directory: {output_dir.resolve()}")
    write_line(f"Watermelon sample count: {len(watermelon_labels)}")
    write_line(f"Iris sample count: {len(iris_labels)}")
    write_line(f"Test ratio: {args.test_ratio}")
    write_line("Random runs: 10")
    write_line("=" * 70)

    watermelon_accuracies = run_ten_random_experiments(
        dataset_name="Watermelon 2.0",
        features=watermelon_features,
        labels=watermelon_labels,
        feature_names=watermelon_feature_names,
        feature_types=watermelon_feature_types,
        test_ratio=args.test_ratio,
        seed_start=args.seed_start,
    )

    write_line("-" * 70)

    iris_accuracies = run_ten_random_experiments(
        dataset_name="Iris",
        features=iris_features,
        labels=iris_labels,
        feature_names=iris_feature_names,
        feature_types=iris_feature_types,
        test_ratio=args.test_ratio,
        seed_start=args.seed_start,
    )

    watermelon_mean_accuracy = float(np.mean(watermelon_accuracies))
    iris_mean_accuracy = float(np.mean(iris_accuracies))

    accuracy_csv_path = output_dir / "accuracy_results.csv"
    accuracy_plot_path = output_dir / "accuracy_curve.png"

    save_accuracy_csv(
        output_path=accuracy_csv_path,
        watermelon_accuracies=watermelon_accuracies,
        iris_accuracies=iris_accuracies,
    )

    plot_accuracy_curve(
        output_path=accuracy_plot_path,
        watermelon_accuracies=watermelon_accuracies,
        iris_accuracies=iris_accuracies,
    )

    write_line("=" * 70)
    write_line(f"Watermelon mean accuracy: {watermelon_mean_accuracy:.4f}")
    write_line(f"Iris mean accuracy: {iris_mean_accuracy:.4f}")
    write_line(f"CSV saved to: {accuracy_csv_path.resolve()}")
    write_line(f"Curve saved to: {accuracy_plot_path.resolve()}")
    write_line("=" * 70)


if __name__ == "__main__":
    main()