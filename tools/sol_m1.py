"""Эталонное решение КИМ 1.1 — Нейрон и перцептрон."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, placeholder, sol, solution_header

nb = Notebook("КИМ 1.1 — эталон")
nb.add(solution_header("КИМ 1.1. Нейрон и перцептрон", "kim-01-neuron-perceptron.ipynb"))

# === Часть А. Нейрон на NumPy ===
nb.add(md("---\n## Часть А. Нейрон на чистом NumPy"))
nb.add(md("### 0. Импорт библиотек"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(42)"""))

nb.add(md("### 1. Прямой проход одного нейрона"))
nb.add(sol("""def neuron_forward(x, w, b, activation):
    z = np.dot(w, x) + b
    return activation(z)"""))

nb.add(md("### 2. Функции активации"))
nb.add(sol("""def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)"""))

nb.add(md("### 3. Сравнение функций активации"))
nb.add(sol("""x_test = np.array([1.0, 2.0, -1.0, 0.5])
w_test = np.array([0.5, -0.5, 1.0, 0.25])
b_test = 0.1

print("sigmoid:", neuron_forward(x_test, w_test, b_test, sigmoid))
print("tanh:   ", neuron_forward(x_test, w_test, b_test, tanh))
print("relu:   ", neuron_forward(x_test, w_test, b_test, relu))

# Графики активаций
z = np.linspace(-5, 5, 200)
plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid(z), label='sigmoid')
plt.plot(z, tanh(z), label='tanh')
plt.plot(z, relu(z), label='relu')
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.xlabel('z'); plt.ylabel('f(z)'); plt.title('Функции активации')
plt.legend(); plt.grid(True); plt.show()"""))

nb.add(md("""**Ответ:** `sigmoid` насыщается (производная → 0) при $|z| \\gtrsim 4$, где
$\\sigma(z) \\to 0$ или $1$. Это проблематично, потому что при насыщении градиент
обнуляется и обучение через backprop замедляется или останавливается (проблема
затухающего градиента). `relu` не насыщается в положительной области
($f'(z) = 1$ при $z > 0$), поэтому градиент течёт свободно — основной повод
предпочесть `relu` в скрытых слоях глубоких сетей."""))

# === Часть Б. Перцептрон на Keras ===
nb.add(md("---\n## Часть Б. Перцептрон на Keras (Fashion-MNIST)"))
nb.add(md("### 4. Загрузка Fashion-MNIST и подготовка данных"))
nb.add(sol("""from tensorflow import keras
from tensorflow.keras import layers, utils

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 28x28 -> 784, нормировка в [0, 1], one-hot метки
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test  = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train_oh = utils.to_categorical(y_train, 10)
y_test_oh  = utils.to_categorical(y_test, 10)

class_names = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
               'Сандали', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботильоны']
print(x_train.shape, x_test.shape, y_train_oh.shape)"""))

nb.add(md("### 5. Построение перцептрона"))
nb.add(sol("""model = keras.Sequential([
    layers.Dense(800, input_dim=784, activation='relu', name='hidden'),
    layers.Dense(10, activation='softmax', name='output'),
])
model.summary()"""))

nb.add(md("### 6. Компиляция"))
nb.add(sol("""model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])"""))

nb.add(md("### 7. Обучение"))
nb.add(sol("""history = model.fit(x_train, y_train_oh,
                    batch_size=200, epochs=10,
                    validation_split=0.2, verbose=2)"""))

nb.add(md("### 8. Оценка точности на тесте"))
nb.add(sol("""test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
print(f'Тестовая точность: {test_acc:.4f}')
# Ожидается ~0.88"""))

nb.add(md("### 9. Визуализация предсказаний"))
nb.add(sol("""predictions = model.predict(x_test, verbose=0)

n = 8
plt.figure(figsize=(14, 3))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    pred = np.argmax(predictions[i])
    true = y_test[i]
    color = 'green' if pred == true else 'red'
    plt.title(f'{class_names[pred][:6]}\\n({class_names[true][:6]})', color=color, fontsize=9)
    plt.axis('off')
plt.suptitle('Зелёный — верно, красный — ошибка'); plt.tight_layout(); plt.show()"""))

nb.add(md("""---
**Параметры сети `Dense(800) → Dense(10)`:** 784·800 + 800 = 628 000 на скрытом
слое; 800·10 + 10 = 8 010 на выходном — всего ~636 000 обучаемых параметров."""))

path = "M1-neuron-basics/attachments/kim-01-neuron-perceptron-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
