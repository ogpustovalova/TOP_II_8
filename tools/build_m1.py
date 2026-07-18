"""Сборка ноутбука КИМ 1.1 — Нейрон и перцептрон (Модуль 1).

Студент:
- реализует нейрон на NumPy (прямой проход, функции активации);
- строит перцептрон на фреймворке (Keras или PyTorch);
- обучает на Fashion-MNIST и оценивает точность.

Формат: теория + ячейки `# < ENTER YOUR CODE HERE >`.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, placeholder

nb = Notebook("КИМ 1.1 — Нейрон и перцептрон")

# === Заголовок ===
nb.add(md("""# КИМ 1.1. Нейрон и перцептрон

**Модуль 1. Искусственные нейронные сети** · Курс «Основы нейронных сетей» · УрФУ

Проверяемые результаты (КРМ v3.0):
- **DL-1.1 (С):** задаёт скорость обучения, выбирает функцию потерь, применяет регуляризацию.
- **DL-1.2 (Б):** способен реализовывать многослойный персептрон.

Подробное описание КИМ, критерии и контрольные вопросы для защиты — в `kim-01-neuron-perceptron.md`."""))

# === Часть А. Нейрон на NumPy ===
nb.add(md("""---
## Часть А. Нейрон на чистом NumPy

Математическая модель нейрона:

$$y = f(w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b) = f\\bigl(W \\cdot x + b\\bigr)$$

где $W$ — вектор весов, $b$ — смещение, $f$ — функция активации."""))

nb.add(md("### 0. Импорт библиотек"))
nb.add(placeholder())

nb.add(md("""### 1. Прямой проход одного нейрона

Реализуйте функцию `neuron_forward(x, w, b, activation)`, возвращающую $y = f(w \\cdot x + b)$
как **матричное** умножение (используйте `np.dot`)."""))
nb.add(placeholder("neuron_forward"))


nb.add(md("""### 2. Функции активации

Реализуйте три функции активации:
- `sigmoid(z)` — $\\sigma(z) = 1/(1+e^{-z})$;
- `tanh(z)` — гиперболический тангенс;
- `relu(z)` — $\\max(0, z)$."""))
nb.add(placeholder("sigmoid, tanh, relu"))


nb.add(md("""### 3. Сравнение функций активации

На одном входе `x_test`, `w_test`, `b_test` посчитайте выход нейрона для трёх
активаций. Постройте графики трёх функций активации на отрезке $[-5, 5]$ с помощью
`matplotlib`. Сделайте вывод о насыщении `sigmoid`."""))
nb.add(placeholder("сравнение активаций"))


nb.add(md("""**Контрольный вопрос (ответьте текстом в ячейке markdown ниже):**
При каких $x$ насыщается `sigmoid`? Почему это problematicично для обучения через
градиент? Чем `relu` лучше в этом смысле?"""))
nb.add(md("> *Ваш ответ:* ..."))


# === Часть Б. Перцептрон на фреймворке ===
nb.add(md("""---
## Часть Б. Перцептрон на фреймворке (Fashion-MNIST)

Теперь построим простейший перцептрон на фреймворке (на выбор — TensorFlow/Keras
или PyTorch) и обучим его распознавать изображения одежды из набора Fashion-MNIST.

Fashion-MNIST: 60 000 обучающих и 10 000 тестовых изображений, каждое 28×28
градаций серого, 10 классов одежды."""))

nb.add(md("### 4. Загрузка Fashion-MNIST и подготовка данных"))
nb.add(placeholder("load_data + reshape(−1, 784) + /255 + one-hot"))

nb.add(md("""Преобразуйте изображения 28×28 в вектор длины 784, нормируйте пиксели
в диапазон $[0, 1]$, приведите метки к one-hot (для Keras — `utils.to_categorical`,
для PyTorch — можно оставить индексы, `CrossEntropyLoss` примет их)."""))

nb.add(md("### 5. Построение перцептрона"))
nb.add(placeholder())

nb.add(md("""Соберите простейший перцептрон: вход `784` → `Dense(800, relu)` →
`Dense(10, softmax)` (Keras), либо эквивалент `Linear(784,800)` → `ReLU` →
`Linear(800,10)` (PyTorch)."""))

nb.add(md("### 6. Компиляция / выбор loss и оптимизатора"))
nb.add(placeholder("compile(loss, optimizer, metrics)"))

nb.add(md("""Для Keras: `loss='categorical_crossentropy'`, `optimizer='SGD'` или
`'adam'`, `metrics=['accuracy']`.
Для PyTorch: `nn.CrossEntropyLoss()` (встроенный softmax) + `torch.optim.SGD/Adam`."""))

nb.add(md("### 7. Обучение"))
nb.add(placeholder("fit / training loop"))

nb.add(md("""Обучите сеть. Для Keras: `model.fit(x_train, y_train, batch_size=200,
epochs=10, validation_split=0.2)`. Для PyTorch — явный цикл по эпохам с
`zero_grad`/`forward`/`loss`/`backward`/`step`."""))

nb.add(md("### 8. Оценка точности на тестовой выборке"))
nb.add(placeholder("evaluate / test loop"))

nb.add(md("### 9. Визуализация предсказаний"))
nb.add(placeholder("predict + argmax + imshow для нескольких примеров"))

nb.add(md("""Покажите несколько тестовых изображений, предсказанные и истинные
классы. Совпадают ли они?"""))


# === Итог ===
nb.add(md("""---
## Итог

Перед сдачей убедитесь, что:
- [ ] ноутбук выполняется сверху вниз без ошибок;
- [ ] есть графики функций активации и предсказаний;
- [ ] тестовая точность перцептрона достигнута;
- [ ] вы готовы ответить на контрольные вопросы из `kim-01-neuron-perceptron.md`.

**Формат сдачи:** заполненный ноутбук + устная защита (5–7 минут).
"""))

# Сохранить
path = "M1-neuron-basics/attachments/kim-01-neuron-perceptron.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
