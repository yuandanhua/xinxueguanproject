import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"
CAT_COLS = ["性别", "胆固醇", "葡萄糖", "是否吸烟", "是否饮酒", "是否锻炼"]
NUM_COLS = ["年龄", "身高", "体重(KG)", "收缩压", "舒张压"]
TARGET = "是否有心血管疾病"


def _ensure_output():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_missing_values(df, prefix):
    _ensure_output()
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print(f"[EDA] {prefix}: 无缺失值")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.sort_values().plot(kind="bar", ax=ax)
    ax.set_title(f"{prefix} - 缺失值分布")
    ax.set_ylabel("缺失数量")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_missing_values.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 缺失值图表已保存: {path}")


def plot_outliers_box(df, prefix):
    _ensure_output()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUM_COLS):
        if col not in df.columns:
            continue
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f"{col} 箱线图")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_outliers_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 箱线图已保存: {path}")


def plot_distributions(df, prefix):
    _ensure_output()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(NUM_COLS):
        if col not in df.columns:
            continue
        df[col].hist(bins=30, ax=axes[i], edgecolor="black")
        axes[i].set_title(f"{col} 分布")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 分布图已保存: {path}")


def plot_correlation_heatmap(df, prefix):
    _ensure_output()
    corr_cols = [c for c in NUM_COLS if c in df.columns]
    if TARGET in df.columns:
        df_enc = df[corr_cols + [TARGET]].copy()
        df_enc[TARGET] = pd.factorize(df_enc[TARGET])[0]
    else:
        df_enc = df[corr_cols].copy()
    corr = df_enc.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title(f"{prefix} - 相关性热力图")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 相关性热力图已保存: {path}")


def plot_target_distribution(df, prefix):
    _ensure_output()
    if TARGET not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[TARGET].value_counts()
    counts.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title(f"{prefix} - 目标变量分布")
    ax.set_xlabel(TARGET)
    ax.set_ylabel("数量")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha="center")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_target_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 目标变量分布已保存: {path}")


def plot_categorical_analysis(df, prefix):
    _ensure_output()
    plot_cols = [c for c in CAT_COLS if c in df.columns]
    if not plot_cols:
        return
    n = len(plot_cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(plot_cols):
        counts = df[col].value_counts()
        counts.plot(kind="bar", ax=axes[i], edgecolor="black")
        axes[i].set_title(f"{col} 分布")
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_categorical_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] 分类特征分析图已保存: {path}")


def run_eda(df_train, df_predict=None):
    print("=" * 50)
    print("开始探索性数据分析 (EDA)")
    print("=" * 50)

    plot_missing_values(df_train, "train")
    plot_outliers_box(df_train, "train")
    plot_distributions(df_train, "train")
    plot_correlation_heatmap(df_train, "train")
    plot_target_distribution(df_train, "train")
    plot_categorical_analysis(df_train, "train")

    if df_predict is not None:
        plot_missing_values(df_predict, "predict")
        plot_outliers_box(df_predict, "predict")
        plot_distributions(df_predict, "predict")
        plot_correlation_heatmap(df_predict, "predict")
        plot_categorical_analysis(df_predict, "predict")

    print("[EDA] 完成")
    return


if __name__ == "__main__":
    from data_loader import load_data
    train, predict = load_data()
    run_eda(train, predict)
