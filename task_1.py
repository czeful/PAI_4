# iris_classification.py
# Полный цикл: загрузка -> split -> обучение -> метрики -> графики -> таблица результатов
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
)
import os

# ---------- Настройки ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
USE_GRIDSEARCH = False  # <-- Поставьте True если хотите лёгкий GridSearch (увеличит время)
OUT_DIR = "iris_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1. Загрузка и первичный анализ ----------
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("X shape:", X.shape)
print("y shape:", y.shape)
print("features:", feature_names)
print("classes:", target_names)
print("class balance:", np.bincount(y))

# ---------- 2. Разбиение выборки ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- 3. Пайплайны моделей ----------
pipe_dt = Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
pipe_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])
pipe_knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])

# ---------- 4. (Опционально) GridSearch (легкий) ----------
if USE_GRIDSEARCH:
    print("Running light GridSearch (5-fold)...")
    param_grid_dt = {"clf__max_depth": [2,3,4,None], "clf__min_samples_split":[2,4,6]}
    param_grid_lr = {"clf__C":[0.1,1,10], "clf__penalty":["l2"], "clf__solver":["lbfgs"]}
    param_grid_knn = {"clf__n_neighbors":[3,5,7,9]}

    gs_dt = GridSearchCV(pipe_dt, param_grid_dt, cv=5, n_jobs=-1)
    gs_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, n_jobs=-1)
    gs_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=5, n_jobs=-1)

    gs_dt.fit(X_train, y_train)
    gs_lr.fit(X_train, y_train)
    gs_knn.fit(X_train, y_train)

    print("Best DT params:", gs_dt.best_params_)
    print("Best LR params:", gs_lr.best_params_)
    print("Best KNN params:", gs_knn.best_params_)

    clf_dt = gs_dt.best_estimator_
    clf_lr = gs_lr.best_estimator_
    clf_knn = gs_knn.best_estimator_
else:
    # Зафиксируем разумные дефолты (быстро и детерминировано)
    clf_dt = Pipeline([("clf", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE))])
    clf_lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE))])
    clf_knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))])

# ---------- 5. Обучение финальных моделей (и измерение времени) ----------
models = {
    "Decision Tree": clf_dt,
    "Logistic Regression": clf_lr,
    "KNN": clf_knn
}
timings = {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    t = time.time() - t0
    timings[name] = {"train_time": t}

# ---------- 6. Предсказания и метрики ----------
y_test_binarized = label_binarize(y_test, classes=[0,1,2])
results_list = []
for name, model in models.items():
    t0 = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - t0
    timings[name]["predict_time"] = predict_time

    # Получаем вероятности для ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        # fallback
        y_score = np.zeros_like(y_test_binarized)
        for i,p in enumerate(y_pred):
            y_score[i,p] = 1.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC OvR per class
    fpr = {}; tpr = {}; roc_auc = {}
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    auc_macro = np.mean(list(roc_auc.values()))

    results_list.append({
        "Model": name,
        "Accuracy": acc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1,
        "AUC_macro": auc_macro,
        "Train_time_sec": timings[name]["train_time"],
        "Predict_time_sec": timings[name]["predict_time"],
        "Confusion_matrix": cm,
        "Classification_report": classification_report(y_test, y_pred, target_names=target_names, zero_division=0),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc_per_class": roc_auc
    })

    # 7. Визуализация confusion matrix
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix — {name}")
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"confusion_{name.replace(' ','_')}.png"))
    plt.close()

    # 8. ROC curves (OvR)
    plt.figure(figsize=(6,5))
    for i in range(len(target_names)):
        plt.plot(fpr[i], tpr[i], label=f"Class {target_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title(f"ROC curves (OvR) — {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"roc_{name.replace(' ','_')}.png"))
    plt.close()

# ---------- 9. Сводная таблица ----------
df_summary = pd.DataFrame([{
    "Model": r["Model"],
    "Accuracy": r["Accuracy"],
    "Precision_macro": r["Precision_macro"],
    "Recall_macro": r["Recall_macro"],
    "F1_macro": r["F1_macro"],
    "AUC_macro": r["AUC_macro"],
    "Train_time_sec": r["Train_time_sec"],
    "Predict_time_sec": r["Predict_time_sec"]
} for r in results_list]).set_index("Model")

print("\nСводная таблица метрик:")
print(df_summary)
df_summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"))

# Сохранение подробных текстовых отчётов
for r in results_list:
    with open(os.path.join(OUT_DIR, f"report_{r['Model'].replace(' ','_')}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Model: {r['Model']}\n\n")
        f.write("Classification report:\n")
        f.write(r["Classification_report"])
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(r["Confusion_matrix"]))
        f.write("\n\nROC AUC per class:\n")
        for k,v in r["roc_auc_per_class"].items():
            f.write(f"Class {target_names[k]}: {v:.4f}\n")

print(f"\nВсе графики и таблицы сохранены в папке: {OUT_DIR}")

