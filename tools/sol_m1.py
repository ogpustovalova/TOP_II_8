"""Эталонное решение КИМ 1.1 — Нейрон и перцептрон (PyTorch)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, sol, solution_header

nb = Notebook("КИМ 1.1 — эталон (PyTorch)")
nb.add(solution_header("КИМ 1.1. Нейрон и перцептрон", "kim-01-neuron-perceptron.ipynb"))

nb.add(md("""В этом эталоне:
- **Часть А** — реализация нейрона на чистом NumPy (педагогический стержень: увидеть
  матричную природу прямого прохода).
- **Часть Б** — перцептрон на **PyTorch** (в задании допускается Keras или PyTorch;
  здесь приводится PyTorch-версия)."""))

# === Часть А. Нейрон на NumPy ===
nb.add(md("---\n## Часть А. Нейрон на чистом NumPy"))
nb.add(md("### 0. Импорт библиотек"))
nb.add(sol("""import numpy as np
import matplotlib.pyplot as plt
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
    return np.maximum(0.0, z)"""))

nb.add(md("### 3. Сравнение функций активации"))
nb.add(sol("""x_test = np.array([1.0, 2.0, -1.0, 0.5])
w_test = np.array([0.5, -0.5, 1.0, 0.25])
b_test = 0.1

print('sigmoid:', neuron_forward(x_test, w_test, b_test, sigmoid))
print('tanh:   ', neuron_forward(x_test, w_test, b_test, tanh))
print('relu:   ', neuron_forward(x_test, w_test, b_test, relu))

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
предпочесть `relu` в скрытых слоях."""))

# === Часть Б. Перцептрон на PyTorch ===
nb.add(md("---\n## Часть Б. Перцептрон на PyTorch (Fashion-MNIST)"))
nb.add(md("### 4. Импорт PyTorch и выбор устройства"))
nb.add(sol("""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print('Устройство:', device)
torch.manual_seed(42)"""))

nb.add(md("### 5. Загрузка Fashion-MNIST через torchvision"))
nb.add(sol("""transform = transforms.Compose([
    transforms.ToTensor(),                # в [0, 1], форма (1, 28, 28)
    transforms.Lambda(lambda x: x.view(-1))  # вытянуть в (784,)
])

train_data = datasets.FashionMNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=500, shuffle=False)

class_names = ['Футболка', 'Брюки', 'Свитер', 'Платье', 'Пальто',
               'Сандали', 'Рубашка', 'Кроссовки', 'Сумка', 'Ботильоны']
print('Train:', len(train_data), 'Test:', len(test_data))"""))

nb.add(md("### 6. Архитектура перцептрона"))
nb.add(sol("""class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 800),
            nn.ReLU(),
            nn.Linear(800, 10),
            # Softmax не нужен — он встроен в CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)

model = Perceptron().to(device)
print(model)
print('Параметров:', sum(p.numel() for p in model.parameters()))"""))

nb.add(md("""**Замечание:** в PyTorch `nn.CrossEntropyLoss` уже включает softmax, поэтому
`Softmax` в конце модели ставить **не нужно** — иначе функция потерь применит его
дважды. Это отличие от Keras, где softmax должен быть в модели явно."""))

nb.add(md("### 7. Loss и оптимизатор"))
nb.add(sol("""criterion = nn.CrossEntropyLoss()   # принимает raw logits + индексы классов
optimizer = optim.Adam(model.parameters(), lr=1e-3)"""))

nb.add(md("### 8. Цикл обучения"))
nb.add(sol("""def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)              # прямой проход
        loss = criterion(logits, y)    # вычисление потерь
        loss.backward()                # backprop (автоматический!)
        optimizer.step()               # обновление весов
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total

for epoch in range(10):
    loss, acc = train(model, train_loader, criterion, optimizer, device)
    print(f'Эпоха {epoch+1:2d}: loss={loss:.4f}  acc={acc:.4f}')"""))

nb.add(md("### 9. Оценка на тесте"))
nb.add(sol("""def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

test_acc = evaluate(model, test_loader, device)
print(f'\\nТестовая точность: {test_acc:.4f}')  # ожидается ~0.88"""))

nb.add(md("### 10. Визуализация предсказаний"))
nb.add(sol("""# Возьмём 8 тестовых изображений
model.eval()
x_batch, y_batch = next(iter(test_loader))
with torch.no_grad():
    logits = model(x_batch.to(device)).cpu()
    preds = logits.argmax(1)

n = 8
plt.figure(figsize=(14, 3))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.imshow(x_batch[i].view(28, 28), cmap='gray')
    pred, true = preds[i].item(), y_batch[i].item()
    color = 'green' if pred == true else 'red'
    plt.title(f'{class_names[pred][:6]}\\n({class_names[true][:6]})', color=color, fontsize=9)
    plt.axis('off')
plt.suptitle('Зелёный — верно, красный — ошибка'); plt.tight_layout(); plt.show()"""))

nb.add(md("""---
**Параметры сети `Linear(784, 800) → Linear(800, 10)`:** 784·800 + 800 = 628 000 на
скрытом слое; 800·10 + 10 = 8 010 на выходном — всего ~636 000 обучаемых параметров.

**Ключевые этапы цикла обучения PyTorch (для защиты):**
1. `optimizer.zero_grad()` — обнулить накопленные градиенты (PyTorch их суммирует!).
2. `logits = model(x)` — прямой проход.
3. `loss = criterion(logits, y)` — вычислить функцию потерь.
4. `loss.backward()` — **обратное распространение ошибки** (автодифференцирование).
5. `optimizer.step()` — обновить веса по вычисленным градиентам."""))

path = "M1-neuron-basics/attachments/kim-01-neuron-perceptron-solution.ipynb"
nb.save(path, preserve_outputs=True)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
