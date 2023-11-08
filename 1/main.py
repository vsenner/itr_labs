import tensorflow as tf
import tensorflow_datasets as tfds
from ffn import model_ffn
from cnn import model_cnn
from preprocess import preprocess_image

# Шлях до датасету, який вже було завантажено
dataset_path = "cats_vs_dogs"

# Завантаження даних та попередня обробка в main.py
(train_data, test_data), info = tfds.load(
    dataset_path,  # Використовуйте шлях до датасету
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

train_data = train_data.map(preprocess_image)
test_data = test_data.map(preprocess_image)

BATCH_SIZE = 32

train_data = train_data.shuffle(1000).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

epochs = 10

# Навчання FFN моделі
ffn_history = model_ffn.fit(train_data, epochs=epochs, validation_data=test_data)

# Навчання CNN моделі
cnn_history = model_cnn.fit(train_data, epochs=epochs, validation_data=test_data)

