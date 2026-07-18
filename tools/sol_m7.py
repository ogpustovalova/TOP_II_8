"""Эталонное решение КИМ 7.1 — Оптимизаторы и управление lr."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 7.1 — эталон")
nb.add(solution_header("КИМ 7.1. Оптимизаторы и управление скоростью обучения", "kim-07-optimizers.ipynb"))

# === Часть А. Сравнение оптимизаторов ===
nb.add(md("---\n## Часть А. Сравнение оптимизаторов"))
nb.add(md("### 0. Импорт"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import time"""))

nb.add(md("### 1. Загрузка Fashion-MNIST"))
nb.add(sol("""(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 784).astype('float32') / 255.0
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh  = keras.utils.to_categorical(y_test, 10)"""))

nb.add(md("### 2. Фиксация архитектуры"))
nb.add(sol("""def build_model():
    return keras.Sequential([
        layers.Dense(256, activation='relu', input_dim=784),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax'),
    ])"""))

nb.add(md("### 3. Обучение с разными оптимизаторами"))
nb.add(sol("""opts = {
    'SGD':       optimizers.SGD(),
    'SGD+momentum': optimizers.SGD(momentum=0.9),
    'Adam':      optimizers.Adam(),
    'AdamW':     optimizers.AdamW(),
    'RMSProp':   optimizers.RMSprop(),
}

EPOCHS = 20
BATCH = 128
histories = {}
for name, opt in opts.items():
    tf.random.set_seed(42)  # одинаковая инициализация
    m = build_model()
    m.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    t0 = time.time()
    h = m.fit(x_train, y_train_oh, batch_size=BATCH, epochs=EPOCHS,
              validation_split=0.2, verbose=0)
    dt = time.time() - t0
    test_acc = m.evaluate(x_test, y_test_oh, verbose=0)[1]
    histories[name] = h.history
    print(f'{name:14s}  test_acc={test_acc:.4f}  время={dt:.1f}с')"""))

nb.add(md("### 4. Сравнительные графики"))
nb.add(sol("""fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for name, h in histories.items():
    ax[0].plot(h['loss'], label=name)
    ax[1].plot(h['val_accuracy'], label=name)
ax[0].set_title('Train loss'); ax[0].set_xlabel('эпоха'); ax[0].legend(); ax[0].grid(True)
ax[1].set_title('Val accuracy'); ax[1].set_xlabel('эпоха'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("""**Ответ (SGD vs Adam/RMSProp):**
- **SGD** обновляет веса по $w \\leftarrow w - \\eta \\nabla L$, одинаково для всех
  параметров. Медленный, чувствителен к масштабу градиентов.
- **Момент** (momentum) добавляет инерцию: $v \\leftarrow \\beta v + \\nabla L$,
  $w \\leftarrow w - \\eta v$ — ускоряет в согласованном направлении, сглаживает
  колебания.
- **Adam/RMSProp** — адаптивные: делят шаг на корень из скользящей дисперсии
  градиентов, поэтому каждый параметр имеет свой эффективный lr. Adam = момент +
  адаптивность. Сходится быстрее SGD на практике, потому что автоматически
  подбирает масштаб шага под каждый параметр.

**Недостатки Adam:** иногда хуже обобщает (из-за адаптивности может застревать в
острых минимумах с плохой генерализацией), поэтому для задач с большой
генерализацией часто предпочитают SGD с моментом."""))

# === Часть Б. Управление lr ===
nb.add(md("---\n## Часть Б. Управление скоростью обучения"))
nb.add(md("### 5. Влияние lr"))
nb.add(sol("""lrs = [1e-2, 1e-3, 1e-4]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for lr in lrs:
    tf.random.set_seed(42)
    m = build_model()
    m.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr), metrics=['accuracy'])
    h = m.fit(x_train, y_train_oh, batch_size=BATCH, epochs=15,
              validation_split=0.2, verbose=0)
    ax[0].plot(h.history['loss'], label=f'lr={lr}')
    ax[1].plot(h.history['val_accuracy'], label=f'lr={lr}')
ax[0].set_title('Train loss (влияние lr)'); ax[0].legend(); ax[0].grid(True)
ax[1].set_title('Val accuracy (влияние lr)'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()
# lr=1e-2: расходимость/нестабильность; lr=1e-4: слишком медленно; lr=1e-3: оптимум"""))

nb.add(md("### 6. lr-планировщики: ReduceLROnPlateau и CosineDecay"))
nb.add(sol("""# (А) ReduceLROnPlateau
tf.random.set_seed(42)
m1 = build_model()
m1.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(),
           metrics=['accuracy'])
cb_plateau = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=3, min_lr=1e-6)
h1 = m1.fit(x_train, y_train_oh, batch_size=BATCH, epochs=30,
            validation_split=0.2, callbacks=[cb_plateau], verbose=0)

# (Б) CosineDecay
tf.random.set_seed(42)
decay = optimizers.schedules.CosineDecay(1e-3, decay_steps=30 * (len(x_train)*0.8 // BATCH))
m2 = build_model()
m2.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(decay),
           metrics=['accuracy'])
h2 = m2.fit(x_train, y_train_oh, batch_size=BATCH, epochs=30,
            validation_split=0.2, verbose=0)

# (В) Const lr для сравнения
tf.random.set_seed(42)
m3 = build_model()
m3.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-3),
           metrics=['accuracy'])
h3 = m3.fit(x_train, y_train_oh, batch_size=BATCH, epochs=30,
            validation_split=0.2, verbose=0)

plt.figure(figsize=(9, 5))
plt.plot(h3.history['val_loss'], label='const lr=1e-3')
plt.plot(h1.history['val_loss'], label='ReduceLROnPlateau')
plt.plot(h2.history['val_loss'], label='CosineDecay')
plt.xlabel('эпоха'); plt.ylabel('val_loss'); plt.title('Сравнение стратегий lr')
plt.legend(); plt.grid(True); plt.show()"""))

nb.add(md("### 7. EarlyStopping"))
nb.add(sol("""tf.random.set_seed(42)
m = build_model()
m.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(1e-3),
          metrics=['accuracy'])
es = callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                              restore_best_weights=True)
h = m.fit(x_train, y_train_oh, batch_size=BATCH, epochs=200,
          validation_split=0.2, callbacks=[es], verbose=0)
print(f'Остановилось на эпохе {len(h.epoch)+1} из 200 (экономия ~{200-len(h.epoch)-1} эпох)')
print(f'Лучшая val_accuracy: {max(h.history[\"val_accuracy\"]):.4f}')
print(f'Тестовая точность: {m.evaluate(x_test, y_test_oh, verbose=0)[1]:.4f}')"""))

nb.add(md("""**Ответ (ReduceLROnPlateau):** когда val_loss перестаёт улучшаться
(монитор `val_loss`, `patience` эпох подряд), lr уменьшается в `factor` раз
(например, в 2). Логика: на «плато» мы в окрестности минимума; большой lr мешает
сойтись точно, поэтому lr снижают, чтобы делать более мелкие шаги и уточнить
минимум. `EarlyStopping` останавливает обучение, когда метрика на val не улучшается
`patience` эпох подряд. `restore_best_weights=True` возвращает веса с лучшей
эпохи — иначе в конце остались бы веса с переобученной финальной эпохи."""))

# === Вывод ===
nb.add(md("""---
## 8. Вывод

Лучшие результаты дают **Adam** (быстрая сходимость) или **AdamW** (с правильной
декаплинг-регуляризацией), в сочетании с **ReduceLROnPlateau** или
**CosineDecay** для уточнения минимума и **EarlyStopping** для предотвращения
переобучения. SGD+momentum может дать лучшую генерализацию на больших задачах,
но требует больше эпох. Для модулей курса (Fashion-MNIST, CIFAR-10) — оптимально
**AdamW + ReduceLROnPlateau + EarlyStopping**."""))

path = "M7-optimizers/attachments/kim-07-optimizers-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
