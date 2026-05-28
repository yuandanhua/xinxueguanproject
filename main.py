"""
心血管疾病发病风险预测分析 — KNN 模型实验
=========================================
完整流程：数据加载 → EDA → 特征工程 → KNN 训练与调优 → 评估 → 预测 → 报告生成
每步可独立运行（模块化设计）
"""

import sys
import os

os.makedirs("output", exist_ok=True)


def step1_load_data():
    print("\n" + "=" * 60)
    print("步骤 1/7: 数据加载")
    print("=" * 60)
    from src.data_loader import load_data, validate_data
    df_train, df_predict = load_data()
    validate_data(df_train, df_predict)
    return df_train, df_predict


def step2_eda(df_train, df_predict):
    print("\n" + "=" * 60)
    print("步骤 2/7: 探索性数据分析 (EDA)")
    print("=" * 60)
    from src.eda import run_eda
    run_eda(df_train, df_predict)
    return


def step3_feature_engineering(df_train, df_predict):
    print("\n" + "=" * 60)
    print("步骤 3/7: 特征工程")
    print("=" * 60)
    from src.feature_engineering import prepare_data
    result = prepare_data(df_train, df_predict)
    return result


def step4_train_knn(feature_result):
    print("\n" + "=" * 60)
    print("步骤 4/7: KNN 模型训练与超参数调优")
    print("=" * 60)
    from src.model_training import train_knn
    model, grid_search = train_knn(
        feature_result["X_train"], feature_result["y_train"]
    )
    return model, grid_search


def step5_evaluate(model, feature_result):
    print("\n" + "=" * 60)
    print("步骤 5/7: 模型评估")
    print("=" * 60)
    from src.evaluation import evaluate_model
    report, y_pred, y_prob = evaluate_model(
        model, feature_result["X_test"], feature_result["y_test"]
    )
    return report, y_pred, y_prob


def step6_predict(model, feature_result):
    print("\n" + "=" * 60)
    print("步骤 6/7: 对新数据进行预测")
    print("=" * 60)
    from src.evaluation import predict_new
    result_df = predict_new(
        model, feature_result["X_predict"], feature_result["predict_ids"]
    )
    return result_df


def step7_generate_report(metrics):
    print("\n" + "=" * 60)
    print("步骤 7/7: 生成实验手册")
    print("=" * 60)
    from src.generate_report import create_report
    eda_images = [
        f for f in os.listdir("output")
        if f.endswith(".png") and not f.endswith("confusion_matrix.png")
        and not f.endswith("roc_curve.png")
    ]
    report_path = create_report(metrics, eda_images)
    return report_path


def main():
    print("=" * 60)
    print("心血管疾病发病风险预测分析（KNN模型实验）")
    print("=" * 60)

    try:
        df_train, df_predict = step1_load_data()
        step2_eda(df_train, df_predict)
        feature_result = step3_feature_engineering(df_train, df_predict)
        model, grid_search = step4_train_knn(feature_result)
        report, y_pred, y_prob = step5_evaluate(model, feature_result)
        result_df = step6_predict(model, feature_result)
        report_path = step7_generate_report(report)

        print("\n" + "=" * 60)
        print("全流程完成！")
        print(f"- 模型: output/knn_model.joblib")
        print(f"- 标准化器: output/scaler.joblib")
        print(f"- 评估报告: output/test_evaluation_report.csv")
        print(f"- 预测结果: output/predictions.csv")
        print(f"- EDA 图表: output/*.png")
        print(f"- 实验手册: {report_path}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[错误] 执行失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
