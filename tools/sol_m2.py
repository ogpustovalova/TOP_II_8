"""Эталонное решение КИМ 2.1 — Backprop и обучение сети."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 2.1 — эталон")
nb.add(solution_header("КИМ 2.1. Backprop и обучение сети", "kim-02-backprop-training.ipynb"))

nb.add(md("---\n## Часть А. Backprop на чистом NumPy"))
nb.add(md("### 0. Импорт и подмножество Fashion-MNIST"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import keras

np.random.seed(42)

(x_train_full, y_train_full), _ = keras.datasets.fashion_mnist.load_data()
# Берём 10 000 примеров для скорости
N = 10000
idx = np.random.choice(60000, N, replace=False)
X = x_train_full[idx].reshape(N, 784).astype('float32') / 255.0

# One-hot
y = np.eye(10)[y_train_full[idx]]

# Разделение train/val
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
print(X_train.shape, y_train.shape)"""))

nb.add(md("### 1. Инициализация параметров (He для ReLU)"))
nb.add(sol("""def init_params():
    # He initialization: W ~ N(0, sqrt(2/fan_in))
    W1 = np.random.randn(784, 64) * np.sqrt(2.0 / 784)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 10) * np.sqrt(2.0 / 64)
    b2 = np.zeros(10)
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()"""))

nb.add(md("### 2. Прямой проход"))
nb.add(sol("""def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # стабилизация
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def forward(X):
    z1 = X @ W1 + b1        # (B, 64)
    a1 = relu(z1)           # (B, 64)
    z2 = a1 @ W2 + b2       # (B, 10)
    y_hat = softmax(z2)     # (B, 10)
    return z1, a1, z2, y_hat"""))

nb.add(md("### 3. Функция потерь — кросс-энтропия"))
nb.add(sol("""def loss(y_hat, y):
    eps = 1e-12
    B = y.shape[0]
    return -np.sum(y * np.log(y_hat + eps)) / B"""))

nb.add(md("""### 4. Обратный проход (backprop)

Для softmax + cross-entropy совместно: $\\partial L / \\partial z_2 = \\hat{y} - y$.
Дальше — правило цепи по графу."""))
nb.add(sol("""def backward(X, y, z1, a1, z2, y_hat):
    B = X.shape[0]
    dz2 = (y_hat - y) / B                     # (B, 10)
    dW2 = a1.T @ dz2                          # (64, 10)
    db2 = np.sum(dz2, axis=0)                 # (10,)
    da1 = dz2 @ W2.T                          # (B, 64)
    dz1 = da1 * (z1 > 0)                      # производная ReLU
    dW1 = X.T @ dz1                           # (784, 64)
    db1 = np.sum(dz1, axis=0)                 # (64,)
    return dW1, db1, dW2, db2"""))

nb.add(md("### 5. Цикл обучения с мини-батчами"))
nb.add(sol("""def iterate_minibatches(X, y, batch_size, shuffle=True):
    if shuffle:
        idx = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        sel = idx[i:i + batch_size] if shuffle else slice(i, i + batch_size)
        yield X[sel], y[sel]

lr = 0.1
epochs = 50
batch_size = 64
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    for xb, yb in iterate_minibatches(X_train, y_train, batch_size):
        z1, a1, z2, y_hat = forward(xb)
        dW1, db1, dW2, db2 = backward(xb, yb, z1, a1, z2, y_hat)
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    # Метрики после эпохи
    _, _, _, yh_tr = forward(X_train)
    _, _, _, yh_va = forward(X_val)
    train_losses.append(loss(yh_tr, y_train))
    val_losses.append(loss(yh_va, y_val))
    train_accs.append(np.mean(np.argmax(yh_tr, axis=1) == np.argmax(y_train, axis=1)))
    val_accs.append(np.mean(np.argmax(yh_va, axis=1) == np.argmax(y_val, axis=1)))

    if (epoch + 1) % 10 == 0:
        print(f"Эпоха {epoch+1}: train_loss={train_losses[-1]:.4f}, "
              f"val_loss={val_losses[-1]:.4f}, val_acc={val_accs[-1]:.4f}")"""))

nb.add(md("### 6. Графики loss и accuracy"))
nb.add(sol("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(train_losses, label='train'); ax[0].plot(val_losses, label='val')
ax[0].set_title('Loss'); ax[0].set_xlabel('эпоха'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(train_accs, label='train'); ax[1].plot(val_accs, label='val')
ax[1].set_title('Accuracy'); ax[1].set_xlabel('эпоха'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("""**Ответ (правило цепи):** backprop основан на правиле цепи
$\\frac{\\partial L}{\\partial W_1} = \\frac{\\partial L}{\\partial \\hat{y}}
\\cdot \\frac{\\partial \\hat{y}}{\\partial z_2} \\cdot \\frac{\\partial z_2}{\\partial a_1}
\\cdot \\frac{\\partial a_1}{\\partial z_1} \\cdot \\frac{\\partial z_1}{\\partial W_1}$.
Для softmax+cross-entropy первые два сомножителя сворачиваются в $\\hat{y} - y$
(производная $z_2$). Дальше: $\\partial z_2 / \\partial a_1 = W_2^T$,
$\\partial a_1 / \\partial z_1 = \\mathbb{1}[z_1 > 0]$ (ReLU),
$\\partial z_1 / \\partial W_1 = X^T$. Итого: $\\partial L / \\partial W_1 =
X^T \\cdot ((\\hat{y} - y) W_2^T) \\odot \\mathbb{1}[z_1 > 0]$ — что и реализовано."""))

# === Часть Б. Цикл обучения на Keras ===
nb.add(md("---\n## Часть Б. Цикл обучения на Keras"))
nb.add(md("### 7. Та же сеть через Keras Sequential + fit"))
nb.add(sol("""from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

tf.random.set_seed(42)

model = keras.Sequential([
    layers.Dense(64, input_dim=784, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=20,
                    validation_split=0.2, verbose=0)
print(f"Финальная val_acc (Keras): {history.history['val_accuracy'][-1]:.4f}")"""))

nb.add(md("### 8. Сравнение GD и SGD (разные batch_size)"))
nb.add(sol("""batch_sizes = [10, 50, 200, 500]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for bs in batch_sizes:
    tf.random.set_seed(42)
    m = keras.Sequential([layers.Dense(64, activation='relu', input_dim=784),
                          layers.Dense(10, activation='softmax')])
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    h = m.fit(X_train, y_train, batch_size=bs, epochs=15,
              validation_split=0.2, verbose=0)
    ax[0].plot(h.history['loss'], label=f'bs={bs}')
    ax[1].plot(h.history['val_accuracy'], label=f'bs={bs}')
ax[0].set_title('Train loss'); ax[1].set_title('Val accuracy')
for a in ax: a.set_xlabel('эпоха'); a.legend(); a.grid(True)
plt.tight_layout(); plt.show()
# Вывод: малый batch_size (10, 50) даёт более быструю сходимость по эпохе,
# но с большим шумом; большой batch (500) — медленнее, но более гладко."""))

# === Часть В. Регуляризация ===
nb.add(md("---\n## Часть В. Переобучение и регуляризация\n### 9. Обнаружение переобучения (много эпох)"))
nb.add(sol("""tf.random.set_seed(42)
m_overfit = keras.Sequential([
    layers.Dense(512, activation='relu', input_dim=784),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
m_overfit.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
h = m_overfit.fit(X_train, y_train, batch_size=64, epochs=100,
                  validation_split=0.2, verbose=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h.history['loss'], label='train'); ax[0].plot(h.history['val_loss'], label='val')
ax[0].set_title('Loss: переобучение (val растёт)'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(h.history['accuracy'], label='train'); ax[1].plot(h.history['val_accuracy'], label='val')
ax[1].set_title('Accuracy: разрыв train/val'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("### 10. Регуляризация: Dropout + EarlyStopping"))
nb.add(sol("""from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
m_reg = keras.Sequential([
    layers.Dense(512, activation='relu', input_dim=784),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax'),
])
m_reg.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
h_reg = m_reg.fit(X_train, y_train, batch_size=64, epochs=100,
                  validation_split=0.2, callbacks=[es], verbose=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h.history['val_loss'], label='без регул.')
ax[0].plot(h_reg.history['val_loss'], label='Dropout+EarlyStopping')
ax[0].set_title('Val loss'); ax[0].legend(); ax[0].grid(True)
ax[1].plot(h.history['val_accuracy'], label='без регул.')
ax[1].plot(h_reg.history['val_accuracy'], label='Dropout+EarlyStopping')
ax[1].set_title('Val accuracy'); ax[1].legend(); ax[1].grid(True)
plt.tight_layout(); plt.show()"""))

nb.add(md("""**Вывод:** `Dropout(0.5)` + `EarlyStopping` эффективно подавляют
переобучение: val_loss перестаёт расти, расхождение train/val уменьшается,
а `EarlyStopping` экономит эпохи обучения, восстанавливая лучшие веса."""))

path = "M2-training/attachments/kim-02-backprop-training-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
