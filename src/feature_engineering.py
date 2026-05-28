import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

CAT_COLS = ["性别", "胆固醇", "葡萄糖", "是否吸烟", "是否饮酒", "是否锻炼"]
NUM_COLS = ["年龄", "身高", "体重(KG)", "收缩压", "舒张压"]
TARGET = "是否有心血管疾病"
ID_COL = "患者编号"

ENCODING_MAP = {
    "胆固醇": {"正常": 0, "高于正常": 1, "远高于正常": 2},
    "葡萄糖": {"正常": 0, "高于正常": 1, "远高于正常": 2},
    "性别": {"男": 0, "女": 1},
    "是否吸烟": {"不吸烟": 0, "吸烟": 1},
    "是否饮酒": {"不饮酒": 0, "饮酒": 1},
    "是否锻炼": {"不锻炼": 0, "锻炼": 1},
}

TARGET_MAP = {"无": 0, "有": 1}


def encode_features(df):
    df_enc = df.copy()
    for col, mapping in ENCODING_MAP.items():
        if col in df_enc.columns:
            df_enc[col] = df_enc[col].map(mapping)
            unknown = df_enc[col].isnull().sum()
            if unknown > 0:
                print(f"[特征工程] '{col}' 存在 {unknown} 个未知值，使用 -1 填充")
                df_enc[col] = df_enc[col].fillna(-1).astype(int)
    if TARGET in df_enc.columns:
        df_enc[TARGET] = df_enc[TARGET].map(TARGET_MAP)
        unknown_tgt = df_enc[TARGET].isnull().sum()
        if unknown_tgt > 0:
            raise ValueError(f"目标列存在 {unknown_tgt} 个无法映射的值")
    return df_enc


def scale_features(df, scaler=None, fit=True):
    num_cols_present = [c for c in NUM_COLS if c in df.columns]
    if not num_cols_present:
        return df, scaler
    if fit:
        scaler = StandardScaler()
        df[num_cols_present] = scaler.fit_transform(df[num_cols_present])
    else:
        df[num_cols_present] = scaler.transform(df[num_cols_present])
    return df, scaler


def prepare_data(df_train, df_predict, test_size=0.2, random_state=42):
    print("=" * 50)
    print("开始特征工程")
    print("=" * 50)

    df_train_enc = encode_features(df_train)
    df_predict_enc = encode_features(df_predict)

    feature_cols = [c for c in (ID_COL, *NUM_COLS, *CAT_COLS) if c in df_train_enc.columns]
    train_ids = df_train_enc[ID_COL].values if ID_COL in df_train_enc.columns else None
    predict_ids = df_predict_enc[ID_COL].values if ID_COL in df_predict_enc.columns else None

    X = df_train_enc[[c for c in feature_cols if c != ID_COL]].values
    y = df_train_enc[TARGET].values

    X_predict = df_predict_enc[[c for c in feature_cols if c != ID_COL]].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, scaler = scale_features(pd.DataFrame(X_train), fit=True)
    X_test_scaled, _ = scale_features(pd.DataFrame(X_test), scaler=scaler, fit=False)
    X_predict_scaled, _ = scale_features(pd.DataFrame(X_predict), scaler=scaler, fit=False)

    os.makedirs("output", exist_ok=True)
    joblib.dump(scaler, "output/scaler.joblib")

    print(f"[特征工程] 训练集: {X_train.shape}, 测试集: {X_test_scaled.shape}")
    print(f"[特征工程] 预测集: {X_predict_scaled.shape}")
    print(f"[特征工程] 特征数量: {X_train.shape[1]}")
    print(f"[特征工程] 标准化器已保存: output/scaler.joblib")

    result = {
        "X_train": X_train,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "X_predict": X_predict_scaled,
        "train_ids": train_ids,
        "predict_ids": predict_ids,
        "feature_cols": [c for c in feature_cols if c != ID_COL],
        "scaler": scaler,
    }
    return result


if __name__ == "__main__":
    from data_loader import load_data
    train, predict = load_data()
    result = prepare_data(train, predict)
    print("[特征工程] 完成")
