"""Эталонное решение КИМ 6.1 — Автокодировщики AE и VAE."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 6.1 — эталон")
nb.add(solution_header("КИМ 6.1. Автокодировщики AE и VAE", "kim-06-autoencoders.ipynb"))

# === Часть А. AE ===
nb.add(md("---\n## Часть А. AE и снижение размерности"))
nb.add(md("### 0. Импорт"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(42)
np.random.seed(42)"""))

nb.add(md("### 1. Загрузка MNIST"))
nb.add(sol("""(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)
print(x_train.shape, x_test.shape)"""))

nb.add(md("### 2. Архитектура AE (латентная размерность 32)"))
nb.add(sol("""latent_dim = 32

encoder = keras.Sequential([
    layers.Input((784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(latent_dim, activation='relu', name='latent'),
], name='encoder')

decoder = keras.Sequential([
    layers.Input((latent_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid'),
], name='decoder')

ae_input = layers.Input((784,))
z = encoder(ae_input)
x_hat = decoder(z)
ae = keras.Model(ae_input, x_hat, name='autoencoder')
ae.compile(optimizer='adam', loss='mse')
ae.summary()"""))

nb.add(md("### 3. Обучение"))
nb.add(sol("""history = ae.fit(x_train, x_train, epochs=30, batch_size=256,
                    validation_split=0.1, verbose=2)"""))

nb.add(md("### 4. Визуализация восстановлений"))
nb.add(sol("""recon = ae.predict(x_test, verbose=0)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Оригинал
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray'); plt.axis('off')
    # Восстановление
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon[i].reshape(28, 28), cmap='gray'); plt.axis('off')
plt.suptitle('Верхний ряд: оригиналы. Нижний ряд: восстановления AE.'); plt.tight_layout(); plt.show()"""))

nb.add(md("### 5. Латентное пространство 2D vs PCA"))
nb.add(sol("""# Переобучим AE с latent_dim=2 для визуализации
enc2 = keras.Sequential([layers.Input((784,)), layers.Dense(64, activation='relu'),
                         layers.Dense(2, name='latent2')])
dec2 = keras.Sequential([layers.Input((2,)), layers.Dense(64, activation='relu'),
                         layers.Dense(784, activation='sigmoid')])
ae2 = keras.Model(enc2.input, dec2(enc2.output))
ae2.compile(optimizer='adam', loss='mse')
ae2.fit(x_train, x_train, epochs=20, batch_size=256, verbose=0)

z2 = enc2.predict(x_test, verbose=0)

# PCA для сравнения
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca2 = pca.fit_transform(x_test)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
scatter1 = ax[0].scatter(z2[:, 0], z2[:, 1], c=y_test, cmap='tab10', s=5, alpha=0.6)
ax[0].set_title('AE latent (2D) — нелинейная проекция'); ax[0].set_xlabel('z1'); ax[0].set_ylabel('z2')
scatter2 = ax[1].scatter(pca2[:, 0], pca2[:, 1], c=y_test, cmap='tab10', s=5, alpha=0.6)
ax[1].set_title('PCA (2D) — линейная проекция'); ax[1].set_xlabel('PC1'); ax[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=ax[1], label='цифра')
plt.tight_layout(); plt.show()
# AE обычно даёт чуть более чёткие кластеры, чем PCA, за счёт нелинейности."""))

# === Часть Б. Аномалии ===
nb.add(md("---\n## Часть Б. Обнаружение аномалий"))
nb.add(md("### 6. Обучение AE на классе «0»"))
nb.add(sol("""x_zero = x_train[y_train if (y_train:=keras.datasets.mnist.load_data()[0][1]) is not None else 0] if False else None
# Чисто и без хитростей:
(_, y_train_full), _ = keras.datasets.mnist.load_data()
x_zero = x_train[y_train_full == 0]
print('Размер «нормального» датасета (только цифра 0):', x_zero.shape)

ae_zero = keras.Sequential([
    layers.Input((784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid'),
])
ae_zero.compile(optimizer='adam', loss='mse')
ae_zero.fit(x_zero, x_zero, epochs=30, batch_size=128, verbose=0)"""))

nb.add(md("### 7. Гистограмма ошибок и порог"))
nb.add(sol("""from sklearn.metrics import precision_recall_fscore_support

# Ошибка восстановления для всех тестовых примеров
recon_test = ae_zero.predict(x_test, verbose=0)
errors = np.mean((x_test - recon_test) ** 2, axis=1)   # MSE по пикселям
is_normal = (y_test == 0)                               # True = «нормальный»

# Гистограмма
plt.figure(figsize=(9, 5))
plt.hist(errors[is_normal], bins=50, alpha=0.6, label='норма (цифра 0)')
plt.hist(errors[~is_normal], bins=50, alpha=0.6, label='аномалия (≠0)')
plt.xlabel('Ошибка восстановления (MSE)'); plt.ylabel('количество')
plt.legend(); plt.yscale('log'); plt.title('Распределение ошибок восстановления')
plt.show()

# Порог (например, 95-й перцентиль нормы)
thr = np.quantile(errors[is_normal], 0.95)
pred_anomaly = errors > thr
p, r, f1, _ = precision_recall_fscore_support(~is_normal, pred_anomaly,
                                              average='binary', zero_division=0)
print(f'Порог = {thr:.4f}, Precision = {p:.3f}, Recall = {r:.3f}, F1 = {f1:.3f}')"""))

# === Часть В. VAE ===
nb.add(md("---\n## Часть В. VAE и генерация"))
nb.add(md("### 8-9. Энкодер VAE (μ, log σ²) и слой репараметризации"))
nb.add(sol("""latent_dim = 2

class Sampling(layers.Layer):
    '''Слой репараметризации: z = mu + exp(0.5*log_var) * epsilon'''
    def call(self, inputs):
        mu, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps

# Энкодер
enc_input = layers.Input((784,), name='x')
h = layers.Dense(256, activation='relu')(enc_input)
mu = layers.Dense(latent_dim, name='mu')(h)
log_var = layers.Dense(latent_dim, name='log_var')(h)
z = Sampling(name='z')([mu, log_var])
encoder_vae = keras.Model(enc_input, [mu, log_var, z], name='encoder_vae')
encoder_vae.summary()"""))

nb.add(md("### 10-11. Loss VAE и сборка модели"))
nb.add(sol("""# Декодер
dec_input = layers.Input((latent_dim,), name='z')
h = layers.Dense(256, activation='relu')(dec_input)
out = layers.Dense(784, activation='sigmoid')(h)
decoder_vae = keras.Model(dec_input, out, name='decoder_vae')

# Полный VAE с кастомным loss
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kw):
        super().__init__(**kw)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name='loss')
        self.recon_tracker = keras.metrics.Mean(name='recon')
        self.kl_tracker = keras.metrics.Mean(name='kl')

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x)
            x_recon = self.decoder(z)
            # Реконструкция (BCE per sample, summed over pixels)
            recon = tf.reduce_sum(
                keras.losses.binary_crossentropy(x, x_recon)) * 784 / tf.cast(tf.shape(x)[0], tf.float32)
            recon = tf.reduce_mean(recon)
            # KL-дивергенция: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
            loss = recon + kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon)
        self.kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

vae = VAE(encoder_vae, decoder_vae, name='vae')
vae.compile(optimizer='adam')
vae.fit(x_train, epochs=20, batch_size=256, verbose=2)"""))

nb.add(md("### 12. Генерация новых изображений"))
nb.add(sol("""# Сетка точек в латентном пространстве
n = 15
grid_x = np.linspace(-2, 2, n)
grid_y = np.linspace(-2, 2, n)
figure = np.zeros((28 * n, 28 * n))
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder_vae.predict(z_sample, verbose=0)
        figure[i*28:(i+1)*28, j*28:(j+1)*28] = x_decoded[0].reshape(28, 28)

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.title('Сгенерированные изображения (равномерная сетка в латентном пространстве VAE)')
plt.axis('off'); plt.show()"""))

nb.add(md("""**Ответ (репараметризация):** слой `Sampling` берёт $\\mu$ и $\\log \\sigma^2$
из энкодера и возвращает $z = \\mu + \\sigma \\odot \\varepsilon$, где $\\varepsilon
\\sim \\mathcal{N}(0, I)$. Случайная величина $\\varepsilon$ нужна, чтобы градиент
мог протечь через стохастическую операцию сэмплинга (reparameterization trick):
без неё backprop не работает, потому что $z$ случайный. Мы сэмплируем шум из
фиксированного распределения и дифференцируемое преобразование переносит градиент
к $\\mu$ и $\\sigma$.

**Loss VAE (два слагаемых):**
1. **Реконструкция** — близость $\\hat{x}$ к $x$ (BCE/MSE). Заставляет декодер
   восстанавливать вход.
2. **KL-дивергенция** — $-\\tfrac{1}{2} \\sum_i (1 + \\log\\sigma_i^2 - \\mu_i^2 - \\sigma_i^2)$
   между латентным распределением $\\mathcal{N}(\\mu, \\sigma^2)$ и априорным $\\mathcal{N}(0, I)$.
   Заставляет латентное пространство быть гладким и близким к стандартному
   нормальному, что делает возможной генерацию новых объектов."""))

path = "M6-autoencoders/attachments/kim-06-autoencoders-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
