"""Эталонное решение КИМ 3.1 — Полносвязные сети (MLP)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 3.1 — эталон")
nb.add(solution_header("КИМ 3.1. Полносвязные сети (MLP)", "kim-03-mlp.ipynb"))

# === Часть А. Регрессия ===
nb.add(md("---\n## Часть А. Регрессия — California Housing"))
nb.add(md("### 0. Импорт"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import keras
from tensorflow.keras import layers"""))

nb.add(md("### 1. Загрузка и подготовка (Z-нормализация ПО TRAIN)"))
nb.add(sol("""(tr_data, tr_targets), (te_data, te_targets) = \\
    keras.datasets.california_housing.load_data(version='large')

# Z-нормализация признаков по train-статистикам
mean = tr_data.mean(axis=0)
std  = tr_data.std(axis=0)
x_train = (tr_data - mean) / std
x_test  = (te_data - mean) / std      # тот же mean/std!

# Масштабирование цели (для стабильности обучения)
scale_y = 100000.0
y_train = tr_targets / scale_y
y_test  = te_targets / scale_y

print(x_train.shape, x_test.shape)"""))

nb.add(md("""**Ответ:** среднее и СКО для нормализации берутся **только по обучающей
выборке**, чтобы избежать утечки информации из теста в обучение. Если нормировать
test своими статистиками, мы «подсматриваем» в тестовые данные, и оценка качества
становится оптимистично завышенной — модель может переобучиться под конкретное
разбиение."""))

nb.add(md("### 2. Архитектура MLP для регрессии"))
nb.add(sol("""model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),   # без активации — это регрессия
])
model.summary()"""))

nb.add(md("### 3. Компиляция и обучение"))
nb.add(sol("""model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                    validation_split=0.2, verbose=0)"""))

nb.add(md("### 4. Кривые обучения"))
nb.add(sol("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history['loss'], label='train')
ax[0].plot(history.history['val_loss'], label='val')
ax[0].set_title('Loss (MSE)'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(history.history['mae'], label='train')
ax[1].plot(history.history['val_mae'], label='val')
ax[1].set_title('MAE'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("### 5. Оценка на тесте и перевод в доллары"))
nb.add(sol("""test_loss, test_mae_scaled = model.evaluate(x_test, y_test, verbose=0)
test_mae_dollars = test_mae_scaled * scale_y
print(f'Тестовое MAE: {test_mae_scaled:.4f} (в масштабе модели)')
print(f'Тестовое MAE: {test_mae_dollars:,.0f} долларов (примерно)')"""))

# === Часть Б. Классификация ===
nb.add(md("---\n## Часть Б. Классификация — Fashion-MNIST"))
nb.add(md("### 6. Загрузка и подготовка"))
nb.add(sol("""from tensorflow.keras import utils

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test  = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train_oh = utils.to_categorical(y_train, 10)
y_test_oh  = utils.to_categorical(y_test, 10)"""))

nb.add(md("### 7. Архитектура MLP"))
nb.add(sol("""model = keras.Sequential([
    layers.Dense(800, activation='relu', input_dim=784),
    layers.Dense(300, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"""))

nb.add(md("### 8. Обучение и оценка"))
nb.add(sol("""history = model.fit(x_train, y_train_oh, batch_size=128, epochs=20,
                    validation_split=0.2, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
print(f'Тестовая точность MLP 800-300: {test_acc:.4f}')
# Ожидается ~0.89"""))

nb.add(md("### 9. Сравнение архитектур"))
nb.add(sol("""import time
archs = {
    '800-10':     [layers.Dense(800, activation='relu', input_dim=784),
                   layers.Dense(10, activation='softmax')],
    '800-300-10': [layers.Dense(800, activation='relu', input_dim=784),
                   layers.Dense(300, activation='relu'),
                   layers.Dense(10, activation='softmax')],
    '256-128-10': [layers.Dense(256, activation='relu', input_dim=784),
                   layers.Dense(128, activation='relu'),
                   layers.Dense(10, activation='softmax')],
}
results = {}
for name, layers_list in archs.items():
    m = keras.Sequential(layers_list)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    t0 = time.time()
    m.fit(x_train, y_train_oh, batch_size=128, epochs=15, validation_split=0.2, verbose=0)
    dt = time.time() - t0
    _, acc = m.evaluate(x_test, y_test_oh, verbose=0)
    results[name] = (acc, dt)
    print(f'{name:14s}  test_acc={acc:.4f}  время={dt:.1f}с')"""))

nb.add(md("""**Вывод:** глубокая сеть `800-300-10` даёт ~88–89 %, узкая `256-128-10` —
~88 %. Увеличение слоёв повышает точность, но и время обучения, и риск
переобучения. Для Fashion-MNIST оптимально 2 скрытых слоя умеренной ширины."""))

# === Часть В. Теория ===
nb.add(md("""---
## Часть В. Универсальная теорема аппроксимации

**UAT (Cybenko 1989, Hornik 1991):** односкрытный MLP с сигмоидной (или более
общо — нелинейной) активацией может аппроксимировать любую непрерывную функцию на
ограниченном множестве с любой заданной точностью при достаточном числе нейронов.

**Оговорка «существование ≠ находимость»:** теорема гарантирует, что подходящая
аппроксимирующая сеть *существует*, но не что её можно *найти* градиентным
обучением. На практике:
- требуется экспоненциально много нейронов для точной аппроксимации в худшем случае;
- градиентный спуск не гарантирует глобальный оптимум;
- глубокие сети аппроксимируют многие функции экспоненциально эффективнее мелких
  (exponential depth efficiency), поэтому на практике предпочитают глубину, а не
  ширину одного скрытого слоя."""))

path = "M3-dense-networks/attachments/kim-03-mlp-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
