import cv2
import pytesseract
import numpy as np

# Функция для выравнивания текста с использованием Hough Transform
def align_text(image):
    # Конвертируем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем гауссово размытие для удаления шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применяем пороговую фильтрацию для выделения текста
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Применяем операцию открытия для уменьшения помех
    kernel = np.ones((1, 1), np.uint8)
    morph_img = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # Применяем преобразование Хафа для нахождения угла наклона текста
    lines = cv2.HoughLines(morph_img, 1, np.pi / 180, 200)
    cv2.imwrite('test_image1.jpg', morph_img)
    cv2.imwrite('test_image2.jpg', gray)
    cv2.imwrite('test_image3.jpg', blurred)
    cv2.imwrite('test_image4.jpg', thresholded)

    # Если линии найдены, вычисляем угол
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90  # угол наклона линии
            angles.append(angle)

        # Среднее значение угла наклона
        angle = np.mean(angles)

        # Поворот изображения для выравнивания
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        print('Детекция произошла')
        return aligned_image
    else:
        # Если линии не найдены, возвращаем исходное изображение
        return image

# Загружаем изображение
image_path = 'cropped_objects/object_0.jpg'
image = cv2.imread(image_path)

# Выравниваем текст на изображении
aligned_image = align_text(image)

# Конвертируем в оттенки серого для OCR
gray = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)

# Увеличиваем контрастность и удаляем шум
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Сохраняем промежуточный результат (опционально)
cv2.imwrite('aligned_image.jpg', aligned_image)
cv2.imwrite('preprocessed_image.jpg', gray)

# Распознаем текст
custom_config = r'--oem 3 --psm 6'  # OEM: движок Tesseract, PSM: режим сегментации
text = pytesseract.image_to_string(gray, config=custom_config, lang='rus')
print("Распознанный текст:")
print(text)
