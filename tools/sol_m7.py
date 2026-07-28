"""Эталонное решение КИМ 7.1 на PyTorch: оптимизаторы и управление lr."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, sol, solution_header


nb = Notebook("КИМ 7.1 — эталон PyTorch")
nb.add(solution_header(
    "КИМ 7.1. Оптимизаторы и управление скоростью обучения",
    "kim-07-optimizers.ipynb",
))

nb.add(md("""В решении используется **PyTorch**. Все эксперименты проводят на одном
разбиении Fashion-MNIST; внутри одного seed совпадают начальные веса, порядок
батчей и источник случайности Dropout. Это важно: иначе случайные различия можно
ошибочно принять за различия оптимизаторов.

M7 подтверждает вклад в составной **DL-1.1 С КРМ**: выбор initial learning rate
для данной задачи, датасета и архитектуры и понимание градиентного спуска. Loss,
`batch_size`, регуляризация и Dropout фиксированы как контрольные параметры и
проверяются шлюзом M2. Поэтому M7 не является самостоятельным полным
подтверждением: полный DL-1.1 С требует `шлюз M2 = 1` и `шлюз M7 = 1`.

Для protocol-fair сравнения каждый из пяти optimizer настраивается на своей
трёхточечной сетке initial `lr`, и только затем сравниваются пять validation-
победителей. В основной сетке `weight_decay=0`: AdamW включён как алгоритм, но
эффект decoupled weight decay этим экспериментом не проверяется.

Число эпох и train-подвыборка ограничены, а на трёх seed повторяются только два
финалиста, не вся сетка. Test dataset и loader не создаются до окончательной
фиксации конфигурации."""))

# === Часть А. Общий протокол и сравнение оптимизаторов ===
nb.add(md("---\n## Часть А. Сравнение оптимизаторов"))
nb.add(md("### 0. Импорт, выбор устройства и воспроизводимость"))
nb.add(sol("""import copy
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

%matplotlib inline

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Детерминированность важнее небольшой прибавки скорости в учебном сравнении.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
print(f"Устройство: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")"""))

nb.add(md("""`torch.manual_seed` фиксирует генератор CPU, а
`torch.cuda.manual_seed_all` — генераторы CUDA. Детерминированные алгоритмы
снижают разброс повторных запусков. Полная побитовая воспроизводимость между
разными версиями PyTorch, CUDA и разным оборудованием всё же не гарантируется."""))

nb.add(md("### 1. Загрузка и фиксированное разбиение Fashion-MNIST"))
nb.add(sol("""# ToTensor переводит uint8-пиксели в float32 и масштабирует их в [0, 1].
# Затем используем известные среднее и стандартное отклонение Fashion-MNIST.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])

data_root = Path("data")
full_train = datasets.FashionMNIST(
    root=data_root, train=True, download=True, transform=transform
)

# Подвыборка сохраняет обязательную сетку 5x3 при разумном времени выполнения.
split_generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, unused_dataset = random_split(
    full_train, [20_000, 5_000, 35_000], generator=split_generator
)
del unused_dataset

BATCH_SIZE = 512
NUM_WORKERS = min(2, os.cpu_count() or 1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def make_loader(dataset, *, shuffle, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )

# Для каждого запуска train loader создаётся заново с тем же seed. Поэтому
# оптимизаторы видят одинаковый порядок объектов. Validation не перемешивается.
val_loader = make_loader(val_dataset, shuffle=False)
print(f"train={len(train_dataset)}, validation={len(val_dataset)}")
print("Test dataset/loader будут созданы только после финального выбора.")"""))

nb.add(md("""Метки остаются целыми индексами классов от 0 до 9: one-hot encoding
для `nn.CrossEntropyLoss` не нужен."""))

nb.add(md("### 2. Общая MLP и одинаковые начальные веса внутри seed"))
nb.add(sol("""class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.layers(x)

def build_model():
    return FashionMLP()

def fresh_model(seed=SEED):
    # Один seed задаёт и инициализацию, и последующие маски Dropout.
    set_seed(seed)
    return build_model().to(DEVICE)

model = fresh_model()
print(model)
del model"""))

nb.add(md("""Последний слой возвращает **логиты**, поэтому после него нет
`Softmax`. `nn.CrossEntropyLoss` численно устойчиво применяет `LogSoftmax` внутри
себя. Явный `Softmax` перед этой функцией потерь был бы ошибкой."""))

nb.add(md("### 3. Переиспользуемые циклы обучения и оценки"))
nb.add(sol("""criterion = nn.CrossEntropyLoss()

def train_one_epoch(model, loader, optimizer):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=PIN_MEMORY)
        targets = targets.to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += batch_size

    return loss_sum / count, correct / count

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()  # отключает Dropout
    loss_sum = 0.0
    correct = 0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=PIN_MEMORY)
        targets = targets.to(DEVICE, non_blocking=PIN_MEMORY)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        count += batch_size

    return loss_sum / count, correct / count

def fit(
    model,
    optimizer,
    epochs,
    *,
    train_loader,
    scheduler=None,
    scheduler_on_plateau=False,
    verbose=False,
):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": [],
    }
    best_val_acc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if (
            val_acc > best_val_acc
            or (val_acc == best_val_acc and val_loss < best_val_loss)
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            if scheduler_on_plateau:
                # ReduceLROnPlateau принимает измеренную validation loss.
                scheduler.step(val_loss)
            else:
                # CosineAnnealingLR делает шаг один раз после каждой эпохи.
                scheduler.step()

        if verbose:
            print(
                f"epoch {epoch:02d}/{epochs}: "
                f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, lr={current_lr:.2e}"
            )

    history["best_epoch"] = best_epoch
    history["best_state"] = best_state
    return history

CONVERGENCE_TARGET = 0.84

def summarize_history(history):
    best_index = max(
        range(len(history["val_acc"])),
        key=lambda index: (
            history["val_acc"][index],
            -history["val_loss"][index],
        ),
    )
    convergence_epoch = next(
        (
            epoch
            for epoch, value in enumerate(history["val_acc"], start=1)
            if value >= CONVERGENCE_TARGET
        ),
        None,
    )
    return {
        "val_metric": history["val_acc"][best_index],
        "val_loss": history["val_loss"][best_index],
        "convergence": convergence_epoch,
        "best_epoch": best_index + 1,
    }

def selection_key(row):
    # Convergence не участвует: недостигнутый порог нельзя заменять эпохой.
    return (row["val_metric"], -row["val_loss"])

def print_validation_metrics(rows):
    print(f'{"эксперимент":<34} {"best val_acc":>12} {"val_loss":>10} '
          f'{"convergence":>14} {"сек":>8}')
    print("-" * 84)
    for row in rows:
        convergence = (
            str(row["convergence"])
            if row["convergence"] is not None
            else "not reached"
        )
        print(
            f'{row["name"]:<34} {row["val_metric"]:>12.4f} '
            f'{row["val_loss"]:>10.4f} {convergence:>14} '
            f'{row["seconds"]:>8.1f}'
        )"""))

nb.add(md("""Loss суммируется с весом размера батча, а затем делится на число
примеров. Так последний неполный батч не получает непропорционально большой вес.
`model.eval()` обязателен для корректной оценки сети с Dropout."""))

nb.add(md(r"""### 3.1. Правило выбора, зафиксированное до экспериментов

До получения результатов и до любого обращения к test фиксируем критерий:

1. максимальная **лучшая validation accuracy** за одинаковый бюджет эпох;
2. при точном равенстве — меньший validation loss в лучшую по accuracy эпоху.

Порог 0.84 используется только как отдельный измеритель сходимости. Если порог
не достигнут, значение остаётся `None` и выводится как `not reached`; фиктивная
эпоха не создаётся и в tie-break не участвует. Test не участвует в выборе.

Базовый градиентный шаг имеет вид
$\theta_{t+1}=\theta_t-lr\,\nabla_\theta L(\theta_t)$. Градиент указывает
направление наиболее быстрого роста loss, поэтому знак минус задаёт спуск, а
`lr` масштабирует перемещение. Слишком большой `lr` может перескакивать через
хорошую область, вызывать колебания или расходимость; слишком малый даёт
устойчивые, но медленные шаги и может не успеть сойтись за доступный бюджет.
Momentum и адаптивные методы меняют направление или покоординатный масштаб шага,
но необходимость выбрать разумный начальный `lr` остаётся."""))

nb.add(md("### 4. Fair tuning: пять optimizer × три initial lr"))
nb.add(sol("""# У всех сеток одинаковый размер, но диапазоны подходят масштабу
# конкретного алгоритма. Значения фиксируются до просмотра результатов.
optimizer_lr_grids = {
    "SGD": [1e-2, 5e-2, 1e-1],
    "SGD + momentum": [1e-2, 5e-2, 1e-1],
    "Adam": [1e-4, 1e-3, 1e-2],
    "AdamW": [1e-4, 1e-3, 1e-2],
    "RMSProp": [1e-4, 1e-3, 1e-2],
}
grid_sizes = {len(values) for values in optimizer_lr_grids.values()}
assert len(grid_sizes) == 1 and min(grid_sizes) >= 3

WEIGHT_DECAY = 0.0

def make_optimizer(name, params, lr):
    common = {"lr": lr, "weight_decay": WEIGHT_DECAY}
    if name == "SGD":
        return torch.optim.SGD(params, **common)
    if name == "SGD + momentum":
        return torch.optim.SGD(params, momentum=0.9, **common)
    if name == "Adam":
        return torch.optim.Adam(params, **common)
    if name == "AdamW":
        return torch.optim.AdamW(params, **common)
    if name == "RMSProp":
        return torch.optim.RMSprop(params, **common)
    raise ValueError(f"Неизвестный optimizer: {name}")

TUNING_EPOCHS = 6
tuning_histories = {}
tuning_results = []

for optimizer_name, learning_rates in optimizer_lr_grids.items():
    for lr in learning_rates:
        model = fresh_model(SEED)
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
        train_loader = make_loader(train_dataset, shuffle=True, seed=SEED)

        started = time.perf_counter()
        history = fit(
            model, optimizer, TUNING_EPOCHS, train_loader=train_loader
        )
        seconds = time.perf_counter() - started

        tuning_histories[(optimizer_name, lr)] = history
        tuning_results.append({
            "name": f"{optimizer_name}, lr={lr:g}",
            "optimizer": optimizer_name,
            "initial_lr": lr,
            "scheduler": "constant",
            "seconds": seconds,
            **summarize_history(history),
        })

print("Полная таблица fair-сетки 5x3:")
print_validation_metrics(tuning_results)

tuned_optimizer_results = []
tuned_histories = {}
for optimizer_name in optimizer_lr_grids:
    candidates = [
        row for row in tuning_results if row["optimizer"] == optimizer_name
    ]
    best = max(candidates, key=selection_key)
    tuned_optimizer_results.append(best)
    tuned_histories[optimizer_name] = tuning_histories[
        (optimizer_name, best["initial_lr"])
    ]

print("\\nПять optimizer после отдельного validation-выбора lr:")
print_validation_metrics(tuned_optimizer_results)

selected_optimizer_result = max(tuned_optimizer_results, key=selection_key)
selected_optimizer_name = selected_optimizer_result["optimizer"]
selected_initial_lr = selected_optimizer_result["initial_lr"]
print(
    "Лучшая tuned-пара по validation:",
    selected_optimizer_name,
    f"lr={selected_initial_lr:g}",
)

train_batches = (len(train_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
print(
    f"Tuning budget: {len(tuning_results)} runs x {TUNING_EPOCHS} epochs "
    f"x {train_batches} train batches = "
    f"{len(tuning_results) * TUNING_EPOCHS * train_batches} optimizer steps"
)"""))

nb.add(sol("""fig, axes = plt.subplots(5, 2, figsize=(14, 19), sharex=True)
for row_index, (optimizer_name, learning_rates) in enumerate(
    optimizer_lr_grids.items()
):
    for lr in learning_rates:
        history = tuning_histories[(optimizer_name, lr)]
        epochs = range(1, TUNING_EPOCHS + 1)
        axes[row_index, 0].plot(
            epochs, history["train_loss"], marker="o", label=f"lr={lr:g}"
        )
        axes[row_index, 1].plot(
            epochs, history["val_acc"], marker="o", label=f"lr={lr:g}"
        )
    axes[row_index, 0].set_ylabel(optimizer_name)
    axes[row_index, 0].legend()
    axes[row_index, 1].legend()

axes[0, 0].set_title("Train loss: lr grid")
axes[0, 1].set_title("Validation accuracy: lr grid")
for axis in axes.flat:
    axis.set_xlabel("эпоха")
    axis.grid(alpha=0.3)
plt.tight_layout()
plt.show()

for optimizer_name, learning_rates in optimizer_lr_grids.items():
    print(f"\\n{optimizer_name}:")
    boundary_lrs = [("малый", min(learning_rates)), ("большой", max(learning_rates))]
    for label, lr in boundary_lrs:
        history = tuning_histories[(optimizer_name, lr)]
        loss_increases = sum(
            current > previous
            for previous, current in zip(
                history["train_loss"], history["train_loss"][1:]
            )
        )
        val_gain = history["val_acc"][-1] - history["val_acc"][0]
        print(
            f"  {label} lr={lr:g}: best val_acc={max(history['val_acc']):.4f}, "
            f"прирост val_acc={val_gain:+.4f}, "
            f"рост train loss={loss_increases} раз"
        )"""))

nb.add(sol("""fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for optimizer_name, history in tuned_histories.items():
    selected_lr = next(
        row["initial_lr"]
        for row in tuned_optimizer_results
        if row["optimizer"] == optimizer_name
    )
    label = f"{optimizer_name}, lr={selected_lr:g}"
    epochs = range(1, TUNING_EPOCHS + 1)
    axes[0].plot(epochs, history["train_loss"], label=label)
    axes[1].plot(epochs, history["val_loss"], label=label)
    axes[2].plot(epochs, history["val_acc"], label=label)

axes[0].set_title("Train loss: tuned optimizer")
axes[1].set_title("Validation loss: tuned optimizer")
axes[2].set_title("Validation accuracy: tuned optimizer")
for axis in axes:
    axis.set_xlabel("эпоха")
    axis.grid(alpha=0.3)
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()"""))

nb.add(md("""Сравнение protocol-fair: каждый алгоритм получил три запуска и
сначала выбрал свой initial `lr` по одному validation-правилу. Поэтому optimizer
не выбирался до tuning `lr`. Малый шаг обычно даёт более плавное, но медленное
изменение; большой может ускорить старт, но рост loss между эпохами и ухудшение
validation указывают на перескоки. Конкретный вывод делается по напечатанным
значениям и кривым каждого алгоритма.

Обычный SGD использует общий масштаб шага, momentum сглаживает направление,
RMSProp адаптирует покоординатный масштаб, Adam сочетает адаптацию с моментом.
`AdamW` при `weight_decay=0` включён как алгоритмический baseline, но его шаг в
этом режиме практически совпадает с Adam. Поэтому из основной сетки нельзя
делать причинный вывод о пользе decoupled weight decay; механизм отделения decay
от градиента обсуждается отдельно, а регуляризация подтверждается M2.

Вывод относится к Fashion-MNIST, данной MLP, нормализации и `batch_size=512`.
Test dataset всё ещё не создан."""))

# === Часть Б. Learning rate и scheduler ===
nb.add(md("---\n## Часть Б. Управление скоростью обучения"))
nb.add(md("### 5. ReduceLROnPlateau и CosineAnnealingLR"))
nb.add(sol("""SCHEDULER_EPOCHS = 12
scheduler_histories = {}
scheduler_results = []

def make_scheduler(strategy, optimizer, epochs):
    if strategy == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            threshold=5e-3,
            threshold_mode="abs",
            min_lr=selected_initial_lr / 100,
        )
        return scheduler, True
    if strategy == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=selected_initial_lr / 100
        )
        return scheduler, False
    if strategy == "constant":
        return None, False
    raise ValueError(f"Неизвестная стратегия: {strategy}")

def run_configuration(strategy, seed, epochs=SCHEDULER_EPOCHS):
    model = fresh_model(seed)
    optimizer = make_optimizer(
        selected_optimizer_name, model.parameters(), selected_initial_lr
    )
    scheduler, scheduler_on_plateau = make_scheduler(strategy, optimizer, epochs)

    train_loader = make_loader(train_dataset, shuffle=True, seed=seed)
    started = time.perf_counter()
    history = fit(
        model,
        optimizer,
        epochs,
        train_loader=train_loader,
        scheduler=scheduler,
        scheduler_on_plateau=scheduler_on_plateau,
    )
    seconds = time.perf_counter() - started
    return history, seconds

for strategy in ["constant", "ReduceLROnPlateau", "CosineAnnealingLR"]:
    history, seconds = run_configuration(strategy, SEED)
    scheduler_histories[strategy] = history
    row = {
        "name": f"{strategy}, lr={selected_initial_lr:g}",
        "optimizer": selected_optimizer_name,
        "initial_lr": selected_initial_lr,
        "scheduler": strategy,
        "seconds": seconds,
        **summarize_history(history),
    }
    scheduler_results.append(row)

print_validation_metrics(scheduler_results)

plateau_lrs = scheduler_histories["ReduceLROnPlateau"]["lr"]
if min(plateau_lrs) == plateau_lrs[0]:
    print("ReduceLROnPlateau не активировался за бюджет эпох; это часть результата.")

ranked_scheduler_results = sorted(
    scheduler_results, key=selection_key, reverse=True
)
preliminary_strategy = ranked_scheduler_results[0]["scheduler"]
nearest_alternative = ranked_scheduler_results[1]["scheduler"]
print("Предварительный победитель:", preliminary_strategy)
print("Ближайшая альтернатива по validation:", nearest_alternative)"""))

nb.add(sol("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for strategy, history in scheduler_histories.items():
    epochs = range(1, SCHEDULER_EPOCHS + 1)
    axes[0].plot(epochs, history["val_loss"], marker="o", label=strategy)
    axes[1].plot(epochs, history["lr"], marker="o", label=strategy)

axes[0].set_title("Validation loss")
axes[0].set_xlabel("эпоха")
axes[0].set_ylabel("loss")
axes[1].set_title("Learning rate в начале эпохи")
axes[1].set_xlabel("эпоха")
axes[1].set_ylabel("lr")
axes[1].set_yscale("log")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.legend()
plt.tight_layout()
plt.show()"""))

nb.add(md("""`ReduceLROnPlateau` не знает номер «правильной» эпохи: после каждой
эпохи ему явно передаётся измеренный `val_loss` вызовом
`scheduler.step(val_loss)`. Если улучшения нет `patience` эпох, lr умножается на
`factor`. Косинусный scheduler, напротив, заранее задаёт плавную траекторию от
начального lr к `eta_min` и вызывается как `scheduler.step()` без метрики.

Уменьшение шага около минимума позволяет точнее настраивать веса и меньше
перескакивать через хорошие области. Сравнение `constant` использует тот же
выбранный initial `lr` и тот же бюджет. Scheduler **не заменяет выбор initial
lr**: он лишь задаёт дальнейшую траекторию, а старт из слишком большого или
слишком малого значения остаётся проблемой. Улучшение не предполагается заранее
и пока проверяется только по validation; test всё ещё закрыт."""))

# === Часть В. Ручная ранняя остановка ===
nb.add(md("---\n## Часть В. Ручная ранняя остановка"))
nb.add(md("""### 6. Early stopping с `patience=15` и восстановлением весов

`monitor='val_accuracy'` выбран до запуска, потому что основное правило выбора
модели максимизирует именно validation accuracy. `patience=15` допускает
кратковременные плато и шум из-за mini-batch обучения и Dropout на дискретной
validation-метрике. Цена такого patience — до 15 лишних эпох после последнего
улучшения.

Для наглядного plateau берётся 4 000 train-объектов, validation остаётся 5 000.
Отсутствие остановки до лимита 60 эпох является допустимым результатом: callback
не обязан срабатывать. В обоих случаях лучшие веса и метрики должны корректно
восстановиться."""))
nb.add(sol("""EARLY_STOP_MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 15
EARLY_STOP_TRAIN_SIZE = 4_000

early_train_dataset = Subset(train_dataset, range(EARLY_STOP_TRAIN_SIZE))

model = fresh_model()
optimizer = make_optimizer(
    selected_optimizer_name, model.parameters(), selected_initial_lr
)
train_loader = make_loader(early_train_dataset, shuffle=True, seed=SEED)

early_history = {
    "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
}
best_val_acc = -float("inf")
best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
best_weights = copy.deepcopy(model.state_dict())
stopped_early = False

for epoch in range(1, EARLY_STOP_MAX_EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    early_history["train_loss"].append(train_loss)
    early_history["train_acc"].append(train_acc)
    early_history["val_loss"].append(val_loss)
    early_history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0
        # state_dict содержит ссылки на текущие тензоры, поэтому нужна deep copy.
        best_weights = copy.deepcopy(model.state_dict())
        marker = " <- лучшие веса сохранены"
    else:
        epochs_without_improvement += 1
        marker = ""

    print(
        f"epoch {epoch:02d}: train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.4f}{marker}"
    )

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        stopped_early = True
        print(f"Ранняя остановка после эпохи {epoch}")
        break
else:
    print("Early stopping корректно не сработал до лимита эпох")

epochs_run = len(early_history["val_acc"])
last_val_acc = early_history["val_acc"][-1]
model.load_state_dict(best_weights)
restored_val_loss, restored_val_acc = evaluate(model, val_loader)
assert abs(restored_val_acc - best_val_acc) < 1e-12
assert abs(restored_val_loss - best_val_loss) < 1e-12

saved_epochs = EARLY_STOP_MAX_EPOCHS - epochs_run
print("Статус:", "сработал" if stopped_early else "не сработал")
print(f"Выполнено эпох: {epochs_run} из {EARLY_STOP_MAX_EPOCHS}")
print(f"Сэкономлено эпох: {saved_epochs}")
print(f"Лучшая эпоха: {best_epoch}")
print(f"До восстановления: val_accuracy={last_val_acc:.4f}")
print(f"Validation: loss={restored_val_loss:.4f}, accuracy={restored_val_acc:.4f}")"""))

nb.add(sol("""epochs = range(1, epochs_run + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(epochs, early_history["train_loss"], label="train loss")
axes[0].plot(epochs, early_history["val_loss"], label="validation loss")
axes[1].plot(epochs, early_history["train_acc"], label="train accuracy")
axes[1].plot(epochs, early_history["val_acc"], label="validation accuracy")
axes[1].axvline(best_epoch, color="tab:red", linestyle="--", label="best epoch")
for axis in axes:
    axis.set_xlabel("эпоха")
    axis.grid(alpha=0.3)
    axis.legend()
axes[0].set_title("Loss")
axes[1].set_title("Early stopping по validation accuracy")
plt.tight_layout()
plt.show()"""))

nb.add(md("""**Почему нужен `copy.deepcopy`.** Значения в обычном
`model.state_dict()` связаны с тензорами модели и продолжили бы изменяться при
следующих шагах оптимизатора. Глубокая копия фиксирует снимок лучшей эпохи.
Early stopping с `patience=15` завершает обучение только если накопилась нужная
серия эпох без улучшения. Если этого не произошло, обучение корректно достигает
лимита и экономит ноль эпох. В обоих случаях `load_state_dict(best_weights)`
возвращает лучший снимок, а два `assert` проверяют восстановленные accuracy и
loss. Поэтому отсутствие срабатывания не считается ошибкой реализации."""))

# === Часть Г. Устойчивость и окончательное решение ===
nb.add(md("""---
## Часть Г. Устойчивость и окончательное решение

### 7. Выбранная конфигурация и ближайшая альтернатива на трёх seed

После одноseedового validation-сравнения повторяем только предварительного
победителя и второй по тому же критерию вариант. Seed 42 уже посчитан выше, к
нему добавляются два независимых запуска. В каждом seed обе стратегии получают
одинаковые инициализацию, порядок батчей и источник случайности Dropout."""))

nb.add(md("""**Правило финального выбора, зафиксированное до test и до расчёта
seed-агрегатов:**

1. оставить конфигурации не хуже чем на `0.002` от лучшего validation mean;
2. среди них выбрать минимальный sample std;
3. при равном std выбрать большую долю seed, достигших convergence-порога;
4. затем выбрать меньшую среднюю эпоху только среди достигших порога;
5. оставшееся равенство разрешить по validation mean и имени стратегии.

`Not reached` хранится как `None`: оно влияет на долю успеха, но не превращается
в фиктивную эпоху."""))

nb.add(sol("""FINAL_MEAN_TOLERANCE = 0.002
STABILITY_SEEDS = [17, SEED, 73]
assert len(set(STABILITY_SEEDS)) >= 3
finalist_strategies = [preliminary_strategy, nearest_alternative]
stability_runs = {strategy: [] for strategy in finalist_strategies}

for strategy in finalist_strategies:
    for seed in STABILITY_SEEDS:
        if seed == SEED:
            # Повторно используем идентичный запуск из scheduler-сравнения.
            history = scheduler_histories[strategy]
        else:
            history, _ = run_configuration(strategy, seed)

        summary = summarize_history(history)
        stability_runs[strategy].append({"seed": seed, **summary})
        convergence_text = (
            str(summary["convergence"])
            if summary["convergence"] is not None
            else "not reached"
        )
        print(
            f"{strategy:<22} seed={seed:>2}: "
            f"best val_acc={summary['val_metric']:.4f}, "
            f"convergence={convergence_text}"
        )

stability_results = []
for strategy, runs in stability_runs.items():
    values = np.array([run["val_metric"] for run in runs])
    reached_epochs = [
        run["convergence"]
        for run in runs
        if run["convergence"] is not None
    ]
    convergence_success_count = len(reached_epochs)
    convergence_success_rate = convergence_success_count / len(runs)
    convergence_epoch_mean = (
        float(np.mean(reached_epochs)) if reached_epochs else None
    )
    stability_results.append({
        "optimizer": selected_optimizer_name,
        "initial_lr": selected_initial_lr,
        "scheduler": strategy,
        "val_mean": values.mean(),
        "val_std": values.std(ddof=1),
        "convergence_success_count": convergence_success_count,
        "convergence_success_rate": convergence_success_rate,
        "convergence_epoch_mean": convergence_epoch_mean,
        "seeds": len(runs),
    })

print("\\nУстойчивость по validation accuracy (sample std, ddof=1):")
for row in stability_results:
    convergence_mean = (
        f"{row['convergence_epoch_mean']:.1f}"
        if row["convergence_epoch_mean"] is not None
        else "not reached"
    )
    print(
        f"{row['scheduler']:<22}: {row['val_mean']:.4f} "
        f"+/- {row['val_std']:.4f}, "
        f"reached={row['convergence_success_count']}/{row['seeds']}, "
        f"success rate={row['convergence_success_rate']:.0%}, "
        f"mean reached epoch={convergence_mean}"
    )"""))

nb.add(md("""Среднее показывает ожидаемую validation-метрику, а sample standard
deviation (`ddof=1`) — чувствительность результата к трём исследованным seed.
Доля достижения порога считается отдельно; средняя эпоха существует только по
успешным seed. Это локальная проверка устойчивости двух финалистов, а не доказательство
устойчивости всей сетки или всех возможных источников недетерминизма."""))

nb.add(md("""### 8. Decision table, финальный выбор и однократная test-оценка

Теперь применяется зафиксированное выше правило с допуском `0.002`. Только после
получения итогового имени конфигурации создаются test dataset и loader."""))

nb.add(sol("""best_val_mean = max(row["val_mean"] for row in stability_results)
eligible_results = [
    row
    for row in stability_results
    if best_val_mean - row["val_mean"] <= FINAL_MEAN_TOLERANCE
]
print(
    "Кандидаты в пределах допуска 0.002:",
    [row["scheduler"] for row in eligible_results],
)

def robust_selection_key(row):
    # Статус и эпоха образуют отдельный ключ, без суррогатной эпохи для None.
    convergence_order = (
        (1,)
        if row["convergence_epoch_mean"] is None
        else (0, row["convergence_epoch_mean"])
    )
    return (
        row["val_std"],
        -row["convergence_success_rate"],
        convergence_order,
        -row["val_mean"],
        row["scheduler"],
    )

final_result = min(eligible_results, key=robust_selection_key)
final_strategy = final_result["scheduler"]

print(
    f'{"optimizer":<18} {"initial lr":>10} {"scheduler":<22} '
    f'{"validation metric":<22} {"convergence":<16} {"seed stability":<20}'
)
print("-" * 114)
for row in stability_results:
    validation_metric = f'{row["val_mean"]:.4f} +/- {row["val_std"]:.4f}'
    if row["convergence_epoch_mean"] is None:
        convergence = f'not reached (0/{row["seeds"]}, 0%)'
    else:
        convergence = (
            f'{row["convergence_epoch_mean"]:.1f} ep '
            f'({row["convergence_success_count"]}/{row["seeds"]}, '
            f'{row["convergence_success_rate"]:.0%})'
        )
    seed_stability = f'std={row["val_std"]:.4f}, n={row["seeds"]}'
    print(
        f'{row["optimizer"]:<18} {row["initial_lr"]:>10g} '
        f'{row["scheduler"]:<22} {validation_metric:<22} '
        f'{convergence:<16} {seed_stability:<20}'
    )

print("\\nИтоговая конфигурация зафиксирована до test:")
print({
    "optimizer": selected_optimizer_name,
    "initial_lr": selected_initial_lr,
    "scheduler": final_strategy,
})

# Единственное создание и обращение к test во всём цикле выбора.
final_model = fresh_model(SEED)
final_model.load_state_dict(scheduler_histories[final_strategy]["best_state"])
test_dataset = datasets.FashionMNIST(
    root=data_root, train=False, download=True, transform=transform
)
test_loader = make_loader(test_dataset, shuffle=False, seed=SEED)
final_test_loss, final_test_acc = evaluate(final_model, test_loader)
print(f"Final test: loss={final_test_loss:.4f}, accuracy={final_test_acc:.4f}")"""))

nb.add(md("""**Рекомендация.** Для классификации Fashion-MNIST данной MLP следует
использовать напечатанные `optimizer`, `initial lr` и `scheduler`: решение
основано на validation mean/std трёх seed, а test сообщает итоговую оценку и не
меняет конфигурацию. При близких средних предпочтительнее меньший разброс и более
быстрая сходимость.

Рекомендация не переносится автоматически на CIFAR-10, CNN, другой
`batch_size`, нормализацию или регуляризацию: масштаб градиентов и геометрия
задачи меняются, поэтому initial `lr` нужно выбирать заново. Scheduler может
уточнять шаг по ходу обучения, но не исправляет произвольно плохой initial `lr`."""))

# === Вывод ===
nb.add(md("""---
## 9. Вывод

- Каждый из пяти optimizer получил отдельную трёхточечную сетку подходящих `lr`;
  сравнивались только пять пар с validation-выбранными initial `lr`.
- При `weight_decay=0` сетка изолирует регуляризацию и не доказывает преимущество
  decoupled weight decay AdamW; этот механизм обсуждается отдельно.
- `ReduceLROnPlateau` реагирует на фактическую validation loss, а
  `CosineAnnealingLR` меняет lr по заранее заданной траектории; оба стартуют с
  уже выбранного initial `lr`.
- На трёх seed повторены только два финалиста; `not reached`, доля достижения и
  средняя эпоха достигших не смешиваются. Финальное правило использует допуск
  0.002 по mean, затем std и convergence.
- Ручной early stopping должен хранить независимую копию лучших весов и
  восстанавливать её независимо от того, сработала ли остановка.

Конкретная рекомендация ограничена Fashion-MNIST и данной MLP. Test просмотрен
один раз после решения. Выполнение условий M7 даёт `шлюз M7 = 1`, но полный
DL-1.1 С КРМ фиксируется только при `шлюз M2 = 1` и `шлюз M7 = 1`; шлюз M2
проверяет loss, `batch_size`, регуляризацию и Dropout."""))

path = "M7-optimizers/attachments/kim-07-optimizers-solution.ipynb"
nb.save(path, preserve_outputs=True)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
