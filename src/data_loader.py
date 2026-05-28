import pandas as pd
import os

DATA_DIR = r"E:\QST\文档\routa\xinxueguan"
TRAIN_FILE = "历史患者数据信息.csv"
PREDICT_FILE = "当前患者数据信息.csv"

COLUMNS = [
    "患者编号", "年龄", "性别", "身高", "体重(KG)",
    "收缩压", "舒张压", "胆固醇", "葡萄糖",
    "是否吸烟", "是否饮酒", "是否锻炼"
]
TARGET = "是否有心血管疾病"


def load_data(data_dir=None, encoding="utf-8-sig"):
    if data_dir is None:
        data_dir = DATA_DIR
    train_path = os.path.join(data_dir, TRAIN_FILE)
    predict_path = os.path.join(data_dir, PREDICT_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练数据文件不存在: {train_path}")
    if not os.path.exists(predict_path):
        raise FileNotFoundError(f"预测数据文件不存在: {predict_path}")

    df_train = pd.read_csv(train_path, encoding=encoding)
    df_predict = pd.read_csv(predict_path, encoding=encoding)

    print(f"[数据加载] 训练集: {df_train.shape}, 预测集: {df_predict.shape}")
    print(f"[数据加载] 训练集列数: {list(df_train.columns)}")
    print(f"[数据加载] 预测集列数: {list(df_predict.columns)}")

    if TARGET not in df_train.columns:
        raise ValueError(f"训练集缺少目标列 '{TARGET}'")

    return df_train, df_predict


def validate_data(df_train, df_predict):
    issues = []
    if df_train.isnull().sum().sum() > 0:
        issues.append(f"训练集存在 {df_train.isnull().sum().sum()} 个缺失值")
    if df_predict.isnull().sum().sum() > 0:
        issues.append(f"预测集存在 {df_predict.isnull().sum().sum()} 个缺失值")

    common_cols = [c for c in COLUMNS if c in df_train.columns and c in df_predict.columns]
    for col in common_cols:
        train_vals = set(df_train[col].dropna().unique())
        predict_vals = set(df_predict[col].dropna().unique())
        diff = predict_vals - train_vals
        if diff:
            issues.append(f"预测集 '{col}' 包含训练集中未出现的值: {diff}")

    print(f"[数据校验] 训练集目标分布:\n{df_train[TARGET].value_counts()}")
    print(f"[数据校验] 训练集描述:\n{df_train[common_cols].describe()}")

    if issues:
        print(f"[数据校验] 警告: {len(issues)} 个问题")
        for iss in issues:
            print(f"  - {iss}")
    else:
        print("[数据校验] 通过")

    return issues


if __name__ == "__main__":
    train, predict = load_data()
    validate_data(train, predict)
