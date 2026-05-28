import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os


def train_knn(X_train, y_train, cv=5, n_jobs=-1):
    print("=" * 50)
    print("开始 KNN 模型训练与超参数调优")
    print("=" * 50)

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 19, 23],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }

    base_knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=base_knn,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"[KNN训练] 最佳参数: {grid_search.best_params_}")
    print(f"[KNN训练] 最佳交叉验证 F1: {grid_search.best_score_:.4f}")

    cv_results = grid_search.cv_results_
    print(f"[KNN训练] 评估的参数组合数: {len(cv_results['params'])}")

    os.makedirs("output", exist_ok=True)
    joblib.dump(grid_search.best_estimator_, "output/knn_model.joblib")
    print("[KNN训练] 最佳模型已保存: output/knn_model.joblib")

    return grid_search.best_estimator_, grid_search, grid_search.best_params_, grid_search.best_score_


def evaluate_cv(X_train, y_train, best_params, cv=5):
    model = KNeighborsClassifier(**best_params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"[KNN交叉验证] 准确率: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores


if __name__ == "__main__":
    from data_loader import load_data
    from feature_engineering import prepare_data
    train, predict = load_data()
    result = prepare_data(train, predict)
    model, _ = train_knn(result["X_train"], result["y_train"])
    print("[KNN训练] 完成")
