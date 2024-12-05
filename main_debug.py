import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import easyocr
from pytesseract import pytesseract, image_to_string

def load_image_with_padding(image, stride=32):
    # Вычисляем размеры, кратные 32
    height, width, _ = image.shape
    new_height = (height + stride - 1) // stride * stride
    new_width = (width + stride - 1) // stride * stride

    # Добавляем отступы
    padded_image = np.full((new_height, new_width, 3), 114, dtype=np.uint8)  # Цвет заполнения (114 - серый)
    padded_image[:height, :width] = image  # Вставляем изображение в центр

    return padded_image


def resize_to_640x360(image):
    return cv2.resize(image, (900, 600), interpolation=cv2.INTER_LINEAR)


def draw_boxes_and_crop_objects(original_image, detections, model, output_dir):
    cropped_objects = []
    # Рисуем рамки и добавляем метки на оригинальном изображении
    for idx, box in enumerate(detections.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамки
        confidence = box.conf[0] * 100  # Уверенность в процентах
        class_id = int(box.cls[0])  # Идентификатор класса
        label = f"{model.names[class_id]}: {confidence:.2f}%"  # Название класса и уверенность

        # Рисуем прямоугольник вокруг найденного объекта
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # Добавляем текст с названием класса и уверенностью
        cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Вырезаем объект
        cropped_object = original_image[y1:y2, x1:x2]
        cropped_objects.append(cropped_object)

        # Сохраняем вырезанный объект
        output_path = os.path.join(output_dir, f"object_{idx}.jpg")
        cv2.imwrite(output_path, cropped_object)

        # Распознаем текст на вырезанном изображении
        recognized_text = recognize_text(cropped_object)
        print(f"Объект {idx}: {recognized_text}")

    return original_image, cropped_objects

def recognize_text(image):
   #"""Распознаёт текст на переданном изображении с использованием EasyOCR."""
    try:
       # Инициализация модели EasyOCR
        reader = easyocr.Reader(['ru', 'en'])  # Укажите языки, например 'ru' и 'en'

        # Распознавание текста на изображении
        results = reader.readtext(image, detail=0)  # detail=0 возвращает только текст

        # Объединение всех строк текста в одну переменную
        full_text = ' '.join(results)

        print(full_text)

        return full_text.strip()
    except Exception as e:
       return f"Ошибка распознавания текста: {str(e)}"

def run_detection(model_path, source, output_dir):
    # Загружаем модель YOLO
    model = YOLO(model_path)

    # Создаем папку для сохранения объектов
    os.makedirs(output_dir, exist_ok=True)

    # Проверяем, что источник — это изображение или видео
    if source.endswith(('.jpg', '.jpeg', '.png')):
        # Загружаем и изменяем размер изображения
        original_image = cv2.imread(source)
        resized_image = resize_to_640x360(original_image)
        padded_image = load_image_with_padding(resized_image)

        # Выполняем детекцию на изображении
        results = model(padded_image, conf=0.3)  # Порог уверенности 30%
        annotated_image, _ = draw_boxes_and_crop_objects(resized_image.copy(), results[0], model, output_dir)

        # Отображаем изображение с аннотациями
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)  # Ждем нажатия любой клавиши
        cv2.destroyAllWindows()  # Закрываем все окна

    elif source.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(source)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Изменяем размер кадра видео и добавляем отступы
            resized_frame = resize_to_640x360(frame)
            padded_frame = load_image_with_padding(resized_frame)

            # Выполняем детекцию на каждом кадре
            results = model(padded_frame, conf=0.3)
            annotated_frame, cropped_objects = draw_boxes_and_crop_objects(resized_frame.copy(), results[0], model, output_dir)

            # Сохраняем кадры с вырезанными объектами (опционально для видео)
            for idx, obj in enumerate(cropped_objects):
                frame_output_dir = os.path.join(output_dir, f"frame_{frame_count}")
                os.makedirs(frame_output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(frame_output_dir, f"object_{idx}.jpg"), obj)

            # Отображаем кадр с аннотациями
            cv2.imshow('Annotated Frame', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Закрываем при нажатии 'q'
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Формат файла не поддерживается. Используйте изображения или видео.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection Script with Object Cropping")
    parser.add_argument('model', type=str, default='trainModel/exp/weights/best.pt', help='Model path')
    parser.add_argument('source', type=str, default='DataSet/test/images/test.jpg', help='Dataset path')
    parser.add_argument('--output', type=str, default='cropped_objects', help='Directory to save cropped objects')

    args = parser.parse_args()

    run_detection(args.model, args.source, args.output)
