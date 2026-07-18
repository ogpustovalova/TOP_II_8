"""Эталонное решение КИМ 4.1 — Сверточные сети и перенос обучения."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 4.1 — эталон")
nb.add(solution_header("КИМ 4.1. Сверточные сети и перенос обучения", "kim-04-cnn.ipynb"))

# === Часть А. Свёртка с нуля ===
nb.add(md("---\n## Часть А. Свёртка с нуля"))
nb.add(md("### 0. Импорт"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline"""))

nb.add(md("### 1. Реализация conv2d"))
nb.add(sol("""def conv2d(image, kernel, stride=1, padding=0):
    # image: (H, W), kernel: (K, K)
    if padding > 0:
        image = np.pad(image, padding, mode='constant')
    H, W = image.shape
    K, _ = kernel.shape
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            region = image[i*stride:i*stride+K, j*stride:j*stride+K]
            out[i, j] = np.sum(region * kernel)
    return out

# Проверка
img = np.random.rand(5, 5)
kern = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # вертикальный контур
print('Вход 5x5, ядро 3x3, stride=1, padding=0 ->', conv2d(img, kern).shape, '(должно быть 3x3)')"""))

nb.add(md("### 2. Влияние параметров"))
nb.add(sol("""from tensorflow import keras
(x_train, _), _ = keras.datasets.cifar10.load_data()
sample = x_train[0, :, :, 0].astype('float32') / 255.0  # один канал

print('Размер входа:', sample.shape)
print('kernel=3, stride=1, pad=0:', conv2d(sample, np.ones((3,3))).shape, '(30x30)')
print('kernel=5, stride=1, pad=0:', conv2d(sample, np.ones((5,5))).shape, '(28x28)')
print('kernel=3, stride=2, pad=0:', conv2d(sample, np.ones((3,3)), stride=2).shape, '(15x15)')
print('kernel=3, stride=1, pad=1:', conv2d(sample, np.ones((3,3)), padding=1).shape, '(32x32 — same)')"""))

# === Часть Б. CNN на CIFAR-10 ===
nb.add(md("---\n## Часть Б. Собственная CNN на CIFAR-10"))
nb.add(md("### 3. Загрузка и подготовка"))
nb.add(sol("""import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
print(x_train.shape, y_train.shape)"""))

nb.add(md("### 4. Архитектура CNN в стиле VGG"))
nb.add(sol("""def build_cnn():
    return keras.Sequential([
        layers.Input((32, 32, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])
model = build_cnn()
model.summary()"""))

nb.add(md("### 5. Обучение"))
nb.add(sol("""model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=20,
                    validation_split=0.2, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\\nТестовая точность собственной CNN: {test_acc:.4f}')  # ~0.70-0.75"""))

nb.add(md("### 6. Кривые обучения"))
nb.add(sol("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['loss'], label='train'); ax[0].plot(history.history['val_loss'], label='val')
ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(history.history['accuracy'], label='train'); ax[1].plot(history.history['val_accuracy'], label='val')
ax[1].set_title('Accuracy'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

# === Часть В. Transfer learning ===
nb.add(md("---\n## Часть В. Transfer learning на Intel Image Classification"))
nb.add(md("""### 7. Загрузка предобученной ResNet50

Ниже показан шаблон для каталога `intel_image_classification`. Если датасет
скачан и распакован в `data/intel/`, расскомментируйте блок загрузки."""))
nb.add(sol("""# Подготовка: загрузим ResNet50 (предобученную на ImageNet)
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # ЭТАП 1: заморозка backbone
print('Число слоёв в backbone:', len(base_model.layers))
print('Обучаемых параметров после заморозки:',
      sum(np.prod(w.shape) for w in base_model.trainable_weights))"""))

nb.add(md("### 8. Этап 1 — feature extraction (заморозка)"))
nb.add(sol("""from tensorflow.keras.utils import image_dataset_from_directory

# Раскомментировать, если датасет доступен:
# train_ds = image_dataset_from_directory('data/intel/seg_train/seg_train',
#                                         image_size=(224,224), batch_size=64,
#                                         validation_split=0.2, subset='training', seed=42)
# val_ds = image_dataset_from_directory('data/intel/seg_train/seg_train',
#                                       image_size=(224,224), batch_size=64,
#                                       validation_split=0.2, subset='validation', seed=42)

# Модель с новой головой (заменяем классификатор на 6 классов Intel)
inputs = keras.Input((224, 224, 3))
x = keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(6, activation='softmax')(x)
model_tl = keras.Model(inputs, outputs)

model_tl.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_tl.summary(show_trainable=True)"""))

nb.add(md("### 9. Этап 2 — fine-tuning с низким lr"))
nb.add(sol("""# Разморозка
base_model.trainable = True
model_tl.compile(optimizer=keras.optimizers.Adam(1e-5),  # НИЗКИЙ lr!
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
model_tl.summary(show_trainable=True)

# Дообучаем 2-3 эпохи (без датасета здесь не запустится)
# history_ft = model_tl.fit(train_ds, validation_data=val_ds, epochs=2)"""))

nb.add(md("""**Ответ (почему lr=1e-5 при fine-tuning):** backbone уже обучен на
большом датасете (ImageNet) и содержит хорошие признаки. Если обновлять его веса
с большой скоростью (`lr=0.01`), мы быстро разрушим эти признаки (catastrophic
forgetting), и качество упадёт. Низкий lr (1e-5 vs 1e-3 на этапе feature
extraction) вносит малые, аккуратные поправки в backbone, дообучая его под
новый домен, не разрушая общих признаков."""))

nb.add(md("""### 10. Сравнение

| Модель | Точность | Время/эпоха | Обучаемых параметров |
|---|---|---|---|
| Собственная CNN на CIFAR-10 | ~0.70–0.75 | ~30 с | ~1.2 М |
| ResNet50 feature extraction (Intel) | ~0.85 | ~60 с | ~0.4 М (только голова) |
| ResNet50 + fine-tuning (Intel) | ~0.90 | ~90 с | ~24 М (весь backbone) |

Transfer learning даёт быстрый старт (точность выше) за счёт предобученных
признаков. Fine-tuning добавляет ещё 3–5 п.п. ценой разморозки 24 М параметров."""))

path = "M4-cnn/attachments/kim-04-cnn-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
