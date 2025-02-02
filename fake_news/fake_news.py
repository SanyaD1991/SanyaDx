import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# 1. Загрузка данных
try:
    data = pd.read_csv('fake_news.csv')
    print("Данные успешно загружены.")
except FileNotFoundError:
    print("Файл 'fake_news.csv' не найден. Пожалуйста, проверьте путь к файлу.")
    exit()

# 2. Проверка колонок
if 'text' not in data.columns or 'label' not in data.columns:
    print("Файл должен содержать колонки 'text' и 'label'.")
    exit()

# 3. Предобработка текста
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Удаление знаков препинания и чисел
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Лемматизация
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

X = data['text'].apply(preprocess)
y = data['label']

# Проверка баланса классов
print("Распределение классов:")
print(y.value_counts())

# 4. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {X_train.shape[0]} записей")
print(f"Размер тестовой выборки: {X_test.shape[0]} записей")

# 5. Создание пайплайна и подбор гиперпараметров
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('pac', PassiveAggressiveClassifier())
])

param_grid = {
    'tfidf__max_df': [0.7, 0.8, 0.9],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'pac__C': [0.1, 1, 10],
    'pac__max_iter': [50, 100]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f'Best parameters: {grid.best_params_}')
print(f'Best cross-validation accuracy: {grid.best_score_:.2f}')

# 6. Оценка модели
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 7. Построение матрицы ошибок
cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])
print('Confusion Matrix:')
print(cm)

# 8. Визуализация матрицы ошибок
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()