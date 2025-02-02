import os
import pandas as pd
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
path = ''
images_dir = ''
df = None

def download_data():
    # Скачать последнюю версию
    global path
    path = kagglehub.dataset_download("rrighart/jarlids")
    print("Path to dataset files:", path)

def load_data():
    global path
    global images_dir
    global  df
    df = pd.read_csv(f'{path}/jarlids_annots.csv')
    images_dir = os.path.join(path)
    print(f"Images directory: {images_dir}")

def save_data():
    global images_dir
    global df
    if not os.path.exists(images_dir):
        print("Images directory does not exist./", images_dir)
    else:
        # Шаг 1: Определение диапазона номеров
        start_num = 1
        end_num = 200  # Измените это значение в зависимости от ваших данных

        # Шаг 2: Генерация имен файлов
        filenames = [f'p{number}.JPG' for number in range(start_num, end_num + 1)]
        print(f"Generated filenames: {filenames[:5]}...")  # Показываем первые 5 имен для примера

        # Шаг 3: Проверка существования файлов
        existing_files = []
        for filename in filenames:
            file_path = os.path.join(images_dir, filename)
            if os.path.exists(file_path):
                existing_files.append(filename)

        print(f"Number of existing files: {len(existing_files)}")

        # Шаг 4: Создание DataFrame
        df = pd.DataFrame({'filename': existing_files})

        # Добавление других необходимых столбцов, если они есть
        # Например, если у вас есть метки, вы можете добавить их здесь
        # Предположим, что метки основаны на номере файла
        df['label'] = df['filename'].apply(lambda x: 1 if int(x.replace('p', '').replace('.JPG', '')) % 2 == 0 else 0)
        # Преобразование меток в строки
        df['label'] = df['label'].astype(str)
        # Добавление путей к изображениям
        df['image_path'] = df['filename'].apply(lambda x: os.path.join(images_dir, x))

        print(df.head())

        # Дополнительно: Сохранение DataFrame в CSV
        df.to_csv(os.path.join(path, 'jarlids_annots_generated.csv'), index=False)
        print("Generated CSV file saved.")

def generator():

    # Параметры
    img_height = 64
    img_width = 64
    batch_size = 32

    # Создание генераторов данных
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2  # 20% данных для валидации
    )

    train_generator = datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_dataframe(
        df,
        x_col='image_path',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Проверка генераторов данных
    print(f"Number of training samples: {train_generator.samples}")
    print(f"Number of validation samples: {validation_generator.samples}")

    # Создание модели
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Обучение модели
    try:
        history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator
        )
    except Exception as e:
        print(f"An error occurred during model training: {e}")

    # Оценка модели
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Точность проверки: {accuracy * 100:.2f}%')

    # Визуализация процесса обучения
    if 'history' in locals():
        plt.plot(history.history['accuracy'], label='Точность')
        plt.plot(history.history['val_accuracy'], label='Валидационная точность')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("Model training failed, 'history' is not defined.")


# Запуск процесса
def main():
     download_data()
     load_data()
     save_data()
     generator()

if __name__ == "__main__":
    main()



