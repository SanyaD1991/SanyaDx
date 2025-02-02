import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# 1. Загрузка данных
# Скачайте датасет с сайта UCI ML и укажите путь к файлу
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(data_url)

# 2. Изучение данных
print(data.head())
print(data.info())
print(data['status'].value_counts())  # 'status' - целевая переменная

# 3. Предобработка данных
# Разделение на признаки и целевую переменную
X = data.drop(['name', 'status'], axis=1)  # Удаляем столбец 'name', так как он не несет полезной информации
y = data['status']

# Преобразование целевой переменной в числовой формат, если это необходимо
# В данном случае 'status' уже в числовом формате: 0 - здоров, 1 - болен

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Обучение модели
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Оценка модели
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Дополнительная оценка
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))