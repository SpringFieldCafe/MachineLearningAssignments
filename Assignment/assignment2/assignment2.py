import cv2
import pathlib
import numpy as np
import random


# =========================
# 参数设置
# =========================
IMG_SIZE = (128, 128)      # 统一图像大小：宽，高
TRAIN_RATIO = 0.7          # 训练集比例
RANDOM_SEED = 42           # 随机种子，保证每次划分结果一致

# 当前 assignment2.py 所在目录
CURRENT_DIR = pathlib.Path(__file__).resolve().parent

# 项目根目录：Assignment 的上一级
PROJECT_DIR = CURRENT_DIR.parent

# 人脸库目录
FACE_ROOT = PROJECT_DIR / "人脸库"

# 两个人脸数据库路径
DATASETS = {
    "ABERDEEN": FACE_ROOT / "ABERDEEN人脸数据库",
    "FERET": FACE_ROOT / "FERET人脸库"
}

# 支持的图像格式
IMG_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif", ".tiff"]


# =========================
# 中文路径读取图像
# =========================
def imread_chinese_path(img_path):
    """
    解决 cv2.imread 读取中文路径可能失败的问题
    """
    img_path = str(img_path)
    data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# =========================
# imresize 函数
# =========================
def imresize(img, size=(128, 128)):
    """
    将图像调整为统一大小
    size: (width, height)
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# =========================
# 读取数据集
# =========================
def load_face_data():
    X = []
    y = []

    label_map = {
        "ABERDEEN": 0,
        "FERET": 1
    }

    for dataset_name, dataset_path in DATASETS.items():
        if not dataset_path.exists():
            print(f"警告：路径不存在：{dataset_path}")
            continue

        label = label_map[dataset_name]

        # 递归读取所有图片
        img_paths = []
        for suffix in IMG_SUFFIX:
            img_paths.extend(dataset_path.rglob(f"*{suffix}"))

        print(f"{dataset_name} 数据库读取到 {len(img_paths)} 张图像")

        for img_path in img_paths:
            img = imread_chinese_path(img_path)

            if img is None:
                print(f"读取失败：{img_path}")
                continue

            # 调整为统一大小
            img = imresize(img, IMG_SIZE)

            # 转为灰度图
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 归一化到 0~1
            img_gray = img_gray.astype(np.float32) / 255.0

            X.append(img_gray)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


# =========================
# 随机划分训练集和测试集
# =========================
def train_test_split(X, y, train_ratio=0.7):
    total_num = len(X)

    index_list = list(range(total_num))
    random.seed(RANDOM_SEED)
    random.shuffle(index_list)

    train_num = int(total_num * train_ratio)

    train_index = index_list[:train_num]
    test_index = index_list[train_num:]

    Xtrain = X[train_index]
    Xtest = X[test_index]

    Xlabel = y[train_index]
    Ytest = y[test_index]

    return Xtrain, Xtest, Xlabel, Ytest


# =========================
# 主函数
# =========================
def main():
    print("开始读取人脸图像数据...")

    X, y = load_face_data()

    if len(X) == 0:
        print("没有读取到任何图像，请检查路径是否正确。")
        return

    print("图像数据读取完成")
    print("总样本数量：", X.shape[0])
    print("单张图像大小：", X.shape[1:])

    Xtrain, Xtest, Xlabel, Ytest = train_test_split(X, y, TRAIN_RATIO)

    print("训练集 Xtrain 大小：", Xtrain.shape)
    print("测试集 Xtest 大小：", Xtest.shape)
    print("训练标签 Xlabel 大小：", Xlabel.shape)
    print("测试标签 Ytest 大小：", Ytest.shape)

    # 保存数据
    np.save(CURRENT_DIR / "Xtrain.npy", Xtrain)
    np.save(CURRENT_DIR / "Xtest.npy", Xtest)
    np.save(CURRENT_DIR / "Xlabel.npy", Xlabel)
    np.save(CURRENT_DIR / "Ytest.npy", Ytest)

    print("数据已保存到 Assignment 文件夹：")
    print("Xtrain.npy")
    print("Xtest.npy")
    print("Xlabel.npy")
    print("Ytest.npy")


if __name__ == "__main__":
    main()