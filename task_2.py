

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Разделение данных с сохранением пропорций классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

importances = tree.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances (Decision Tree)')
plt.xlabel('Важность признака')
plt.ylabel('Признак')
plt.tight_layout()
plt.show()

print("\n--- Важность признаков ---")
print(importance_df.sort_values(by="Importance", ascending=False))

y_pred_tree = tree.predict(X_test)
cm = confusion_matrix(y_test, y_pred_tree)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Confusion Matrix — Decision Tree')
plt.tight_layout()
plt.show()

tree_acc = accuracy_score(y_test, y_pred_tree)
print(f"\nAccuracy Decision Tree: {tree_acc:.3f}")

k_values = range(1, 21)
knn_accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_knn)
    knn_accuracies.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_accuracies, marker='o')
plt.title('Зависимость Accuracy от k (KNN)')
plt.xlabel('Количество соседей (k)')
plt.ylabel('Точность (Accuracy)')
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(knn_accuracies)]
best_knn_acc = max(knn_accuracies)
print(f"\nОптимальное k для KNN: {best_k}, точность = {best_knn_acc:.3f}")


depths = range(1, 11)
train_acc = []
test_acc = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_acc, marker='o', label='Train Accuracy')
plt.plot(depths, test_acc, marker='s', label='Test Accuracy')
plt.title('Зависимость Accuracy от max_depth (Decision Tree)')
plt.xlabel('Глубина дерева (max_depth)')
plt.ylabel('Точность (Accuracy)')
plt.legend()
plt.grid(True)
plt.show()

best_depth = depths[np.argmax(test_acc)]
best_depth_acc = max(test_acc)
print(f"\nОптимальная глубина дерева: {best_depth}, точность = {best_depth_acc:.3f}")


print("\n=======================")
print("     ИНТЕРПРЕТАЦИЯ")
print("=======================")

# --- 5.1 Сравнение визуализаций ---
print("\n1️⃣ Сравнение визуализаций:")
print("• На графике важности признаков видно, что наибольший вклад в классификацию дают длина и ширина лепестка (petal length, petal width).")
print("• Confusion matrix показывает, что модель почти не путает классы setosa, но иногда ошибается между versicolor и virginica — это логично, т.к. они похожи по параметрам.")

# --- 5.2 Важные признаки ---
top_features = importance_df.sort_values(by='Importance', ascending=False).head(2)
print("\n2️⃣ Наиболее значимые признаки:")
for _, row in top_features.iterrows():
    print(f"• {row['Feature']} — важность {row['Importance']:.3f}")
print("→ Эти признаки наиболее информативны, так как хорошо разделяют классы, особенно petal length и petal width, которые явно различаются у видов iris.")

# --- 5.3 Лучшие гиперпараметры ---
print("\n3️⃣ Оптимальные гиперпараметры:")
print(f"• KNN: k = {best_k}, точность = {best_knn_acc:.3f}")
print(f"• Decision Tree: max_depth = {best_depth}, точность = {best_depth_acc:.3f}")

