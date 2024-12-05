from ultralytics import YOLO

# Загрузка модели
model = YOLO("trainModel/exp/weights/best.pt")  # загрузка модели (более легкая версия YOLO для ускорения)

# Параметры тренировки
model.train(
    data='C:/Users/ACER/PycharmProjects/pythonProject3/DataSet2/data.yaml',  # Путь к файлу data.yaml
    epochs=5,                  # Уменьшено количество эпох для предварительного теста
    project='trainModel',       # Папка для сохранения результатов
    name='exp',                # Имя папки с результатами
    exist_ok=True,             # Перезаписывать папку, если она уже существует
    cos_lr=True,               # Косинусный график скорости обучения для улучшения результата
    patience=50,               # Раннее завершение при отсутствии улучшений
    save_period=1              # Сохранение результатов после каждой эпохи
)

print("Обучение завершено.")