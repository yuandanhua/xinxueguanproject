import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"


def evaluate_model(model, X_test, y_test, prefix="test"):
    print("=" * 50)
    print("模型评估")
    print("=" * 50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"[评估] 准确率 (Accuracy):  {accuracy:.4f}")
    print(f"[评估] 精确率 (Precision): {precision:.4f}")
    print(f"[评估] 召回率 (Recall):    {recall:.4f}")
    print(f"[评估] F1 分数:           {f1:.4f}")
    print(f"[评估] AUC:               {auc:.4f}")

    print(f"\n[评估] 分类报告:\n{classification_report(y_test, y_pred, target_names=['无', '有'])}")

    _plot_confusion_matrix(y_test, y_pred, prefix)
    _plot_roc_curve(y_test, y_prob, prefix, auc)

    report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["无", "有"], output_dict=True
        ),
    }

    _save_report(report, prefix)
    return report, y_pred, y_prob


def predict_new(model, X_predict, ids=None):
    print("=" * 50)
    print("对新数据进行预测")
    print("=" * 50)

    y_pred = model.predict(X_predict)
    y_prob = model.predict_proba(X_predict)[:, 1]

    pred_labels = np.where(y_pred == 1, "有", "无")
    risk_labels = np.where(y_prob >= 0.7, "高风险",
                           np.where(y_prob >= 0.4, "中风险", "低风险"))

    result_df = pd.DataFrame({
        "prediction": pred_labels,
        "probability": y_prob,
        "risk_level": risk_labels,
    })
    if ids is not None:
        result_df.insert(0, "患者编号", ids)

    print(f"[预测] 完成 {len(result_df)} 条预测")
    print(f"[预测] 风险分布:\n{result_df['risk_level'].value_counts()}")
    print(f"[预测] 预测结果分布:\n{result_df['prediction'].value_counts()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "predictions.csv")
    result_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[预测] 预测结果已保存: {path}")

    return result_df


def _plot_confusion_matrix(y_test, y_pred, prefix):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["无", "有"], yticklabels=["无", "有"])
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵 ({prefix})")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[评估] 混淆矩阵已保存: {path}")


def _plot_roc_curve(y_test, y_prob, prefix, auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"KNN (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="随机分类器")
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真阳性率 (TPR)")
    ax.set_title(f"ROC 曲线 ({prefix})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[评估] ROC 曲线已保存: {path}")


def _save_report(report, prefix):
    report_df = pd.DataFrame({
        "指标": ["准确率", "精确率", "召回率", "F1分数", "AUC"],
        "数值": [
            f"{report['accuracy']:.4f}",
            f"{report['precision']:.4f}",
            f"{report['recall']:.4f}",
            f"{report['f1']:.4f}",
            f"{report['auc']:.4f}",
        ],
    })
    path = os.path.join(OUTPUT_DIR, f"{prefix}_evaluation_report.csv")
    report_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[评估] 评估报告已保存: {path}")


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_data
    from model_training import train_knn
    train, predict = load_data()
    result = prepare_data(train, predict)
    model, _ = train_knn(result["X_train"], result["y_train"])
    evaluate_model(model, result["X_test"], result["y_test"])
    predict_new(model, result["X_predict"], result["predict_ids"])
    print("[评估] 完成")
