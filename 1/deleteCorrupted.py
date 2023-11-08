from PIL import Image
import os

# Шлях до папки з датасетом
dataset_folder = '/home/unionim/tensorflow_datasets/cats_vs_dogs/4.0.0.'

# Функція для перевірки, чи зображення є пошкодженим
def is_corrupted_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return False
    except (IOError, SyntaxError):
        return True

# Перевірка і видалення пошкоджених зображень
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        file_path = os.path.join(root, file)
        if is_corrupted_image(file_path):
            os.remove(file_path)
            print(f"Видалено пошкоджене зображення: {file_path}")

print("Готово")
