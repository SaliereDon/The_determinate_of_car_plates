import cv2
import easyocr

# Инициализация модели EasyOCR
reader = easyocr.Reader(['ru', 'en'])  # Укажите языки, например 'ru' и 'en'

        # Увеличиваем контрастность и удаляем шум
image = cv2.imread('cropped_objects/object_0.jpg', )
gray = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Распознавание текста на изображении
results = reader.readtext(gray, detail=0)  # detail=0 возвращает только текст

        # Объединение всех строк текста в одну переменную
full_text = ' '.join(results)

print(full_text)
