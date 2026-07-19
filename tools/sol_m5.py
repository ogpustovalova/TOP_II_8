"""Эталонное решение КИМ 5.1 на PyTorch: последовательности и временные ряды."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, sol, solution_header


nb = Notebook("КИМ 5.1 — эталон на PyTorch")
nb.add(solution_header(
    "КИМ 5.1. Рекуррентные сети и 1D-CNN (PyTorch)",
    "kim-05-sequences.ipynb",
))
nb.add(md("""В решении нет зависимости от Keras: предобработка текста, модели,
циклы обучения и датасет временных окон реализованы явно. По умолчанию часть с
текстом использует небольшой синтетический набор, потому что `banks.csv` не
входит в репозиторий. Все напечатанные метрики получаются при текущем запуске.

Полный запуск рассчитан примерно на 15 минут или меньше. Для Jena Climate
используется репрезентативная хронологическая подвыборка целевых моментов, но
физический смысл каждого окна (5 суток истории и прогноз на 24 часа вперёд)
сохраняется."""))

# === Общая настройка ===
nb.add(md("---\n## Общая настройка"))
nb.add(sol("""import random
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import pymorphy3
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Эти настройки делают CUDA-запуски воспроизводимее. На CPU они безвредны.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch:", torch.__version__)
print("Устройство:", DEVICE)"""))

# === Часть А. Предобработка текстов ===
nb.add(md("---\n## Часть А. Предобработка русских текстов"))
nb.add(md("""### 1. Токенизация, стоп-слова и лемматизация

Сначала текст приводится к нижнему регистру и разбивается именно
`nltk.word_tokenize`. Режим `preserve_line=True` не требует внешней модели
`punkt`. Стоп-слова берутся из корпуса NLTK; если доступ к загрузчику NLTK
ограничен, используется явно обозначенный локальный список. Пунктуацию и числа
отбрасываем через `isalpha`, затем выполняем лемматизацию с помощью
`pymorphy3.MorphAnalyzer`."""))
nb.add(sol("""morph = pymorphy3.MorphAnalyzer()

try:
    ru_stopwords = set(stopwords.words("russian"))
    stopword_source = "корпус NLTK"
except LookupError:
    # Исполняемый fallback для изолированной среды без доступа к nltk.download.
    ru_stopwords = {
        "а", "без", "более", "бы", "был", "была", "были", "было", "быть",
        "в", "вам", "вас", "весь", "во", "вот", "все", "всего", "вы", "где",
        "да", "даже", "для", "до", "его", "ее", "если", "есть", "еще", "же",
        "за", "здесь", "и", "из", "или", "им", "их", "к", "как", "когда",
        "кто", "ли", "либо", "мне", "может", "мы", "на", "над", "надо", "наш",
        "не", "него", "нее", "нет", "ни", "них", "но", "ну", "о", "об", "однако",
        "он", "она", "они", "оно", "от", "очень", "по", "под", "при", "про",
        "раз", "с", "сам", "себе", "себя", "так", "также", "такой", "там", "те",
        "тем", "то", "того", "тоже", "только", "том", "ты", "у", "уже", "хотя",
        "чего", "чем", "что", "чтобы", "эта", "эти", "это", "я",
    }
    stopword_source = "локальный fallback"

print("Источник стоп-слов:", stopword_source)


def preprocess(text):
    tokens = word_tokenize(
        str(text).lower(), language="russian", preserve_line=True,
    )
    lemmas = []
    for token in tokens:
        if not token.isalpha() or token in ru_stopwords:
            continue
        lemma = morph.parse(token)[0].normal_form
        if lemma not in ru_stopwords:
            lemmas.append(lemma)
    return lemmas


example = "Банк быстро решил мою проблему, а сотрудники были очень вежливы!"
print("Исходный текст:", example)
print("После preprocess:", preprocess(example))"""))

nb.add(md("""### 2. Данные и построение словаря

`banks.csv` является внешним датасетом и намеренно не загружается по умолчанию.
Для работы с ним нужно вручную положить файл рядом с ноутбуком, проверить имена
столбцов `text`/`label` и переключить `USE_BANKS_CSV = True`. В обычном
исполняемом пути ниже используется явно обозначенный синтетический набор: он
достаточен для проверки препроцессинга, размерностей и полного цикла обучения,
но не для вывода о качестве на реальных отзывах."""))
nb.add(sol("""USE_BANKS_CSV = False  # ОПЦИОНАЛЬНО: включить только при наличии внешнего файла
BANKS_PATH = Path("banks.csv")

if USE_BANKS_CSV:
    if not BANKS_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {BANKS_PATH}. Скопируйте внешний banks.csv или выключите загрузку."
        )
    banks_df = pd.read_csv(BANKS_PATH)
    required_columns = {"text", "label"}
    if not required_columns.issubset(banks_df.columns):
        raise ValueError("В banks.csv ожидаются столбцы text и label")
    texts = banks_df["text"].fillna("").astype(str).tolist()
    labels = banks_df["label"].astype("int64").to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Для бинарной классификации label должен содержать только 0 и 1")
    dataset_kind = "реальный banks.csv"
else:
    positive_phrases = [
        "банк быстро одобрил заявку и вежливо ответил",
        "сотрудник помог решить вопрос без очереди",
        "удобное приложение работает быстро и надёжно",
        "перевод пришёл мгновенно комиссия оказалась низкой",
        "поддержка подробно объяснила условия и помогла",
        "карта доставлена вовремя обслуживание отличное",
        "приятный офис и внимательный доброжелательный персонал",
        "вклад открыли быстро условия оказались выгодными",
        "оператор компетентно исправил ошибку в тот же день",
        "доволен банком все операции проходят без проблем",
        "кредит оформили честно и понятно без скрытых платежей",
        "спасибо поддержке за быстрый и полезный ответ",
    ]
    negative_phrases = [
        "банк долго рассматривает заявку и не отвечает",
        "сотрудник отказался помочь пришлось стоять в очереди",
        "неудобное приложение постоянно зависает и ломается",
        "перевод задержали и списали высокую комиссию",
        "поддержка не объяснила условия и закрыла обращение",
        "карту доставили поздно обслуживание ужасное",
        "грязный офис и грубый невнимательный персонал",
        "вклад закрывали долго условия оказались невыгодными",
        "оператор допустил ошибку и неделю её не исправлял",
        "разочарован банком операции постоянно отклоняются",
        "в кредите обнаружились непонятные скрытые платежи",
        "поддержка прислала бесполезный ответ после долгого ожидания",
    ]
    contexts = [
        "Отзыв клиента: {}.",
        "Недавно обратился в отделение. {}.",
        "Мой опыт работы с банком: {}.",
        "После обращения могу сказать следующее: {}.",
    ]
    texts = [template.format(phrase) for phrase in positive_phrases for template in contexts]
    labels = np.ones(len(texts), dtype=np.int64)
    negative_texts = [
        template.format(phrase) for phrase in negative_phrases for template in contexts
    ]
    texts.extend(negative_texts)
    labels = np.concatenate([labels, np.zeros(len(negative_texts), dtype=np.int64)])
    dataset_kind = "ДЕМО: синтетические русские отзывы"

print(dataset_kind)
print("Число текстов:", len(texts), "баланс классов:", np.bincount(labels))"""))

nb.add(sol("""MAX_WORDS = 10_000
MAXLEN = 200


def stratified_split_indices(y, test_fraction=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    train_indices, test_indices = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        test_size = max(1, int(round(len(indices) * test_fraction)))
        test_indices.extend(indices[:test_size])
        train_indices.extend(indices[test_size:])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.asarray(train_indices), np.asarray(test_indices)


def build_vocab(tokenized_texts, max_words=MAX_WORDS):
    frequencies = Counter(token for tokens in tokenized_texts for token in tokens)
    word_to_index = {"<PAD>": 0, "<OOV>": 1}
    for word, _ in frequencies.most_common(max_words - 2):
        word_to_index[word] = len(word_to_index)
    return word_to_index


def text_to_sequence(tokens, word_to_index, maxlen=MAXLEN):
    sequence = [word_to_index.get(token, 1) for token in tokens]
    # Предпаддинг: последнее состояние LSTM соответствует последнему слову, а не PAD.
    sequence = [0] * max(0, maxlen - len(sequence)) + sequence
    return sequence[-maxlen:]


train_idx, test_idx = stratified_split_indices(labels)
train_tokens = [preprocess(texts[i]) for i in train_idx]
test_tokens = [preprocess(texts[i]) for i in test_idx]

# Словарь строится только по train, поэтому test не передаёт информацию в обучение.
word_to_index = build_vocab(train_tokens)
X_train = torch.tensor(
    [text_to_sequence(tokens, word_to_index) for tokens in train_tokens],
    dtype=torch.long,
)
X_test = torch.tensor(
    [text_to_sequence(tokens, word_to_index) for tokens in test_tokens],
    dtype=torch.long,
)
y_train = torch.tensor(labels[train_idx], dtype=torch.float32)
y_test = torch.tensor(labels[test_idx], dtype=torch.float32)

assert word_to_index["<PAD>"] == 0 and word_to_index["<OOV>"] == 1
assert X_train.shape[1] == MAXLEN and int(X_train.max()) < MAX_WORDS
print("Размер словаря:", len(word_to_index), "из максимальных", MAX_WORDS)
print("Train/test:", X_train.shape, X_test.shape)
print("Проверка PAD/OOV:", text_to_sequence(["совсемновоеслово"], word_to_index, 4))"""))

# === Часть Б. Классификация текста ===
nb.add(md("---\n## Часть Б. Классификация текста: LSTM и 1D-CNN"))
nb.add(md("""Обе модели возвращают **логиты**, поэтому используется численно
устойчивая `BCEWithLogitsLoss` (сигмоида уже включена в loss). `padding_idx=0`
фиксирует нулевой эмбеддинг для паддинга. В CNN `Conv1d` ожидает каналы перед
длиной последовательности, поэтому после `Embedding` оси переставляются."""))
nb.add(sol("""text_train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)
text_test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=MAX_WORDS, embedding_dim=64, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)             # (batch, length, 64)
        _, (hidden, _) = self.lstm(embedded)
        return self.output(hidden[-1]).squeeze(1)         # (batch,)


class CNN1DClassifier(nn.Module):
    def __init__(self, vocab_size=MAX_WORDS, embedding_dim=64, channels=250):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, channels, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.output = nn.Linear(channels, 1)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).transpose(1, 2)  # (batch, 64, length)
        features = torch.relu(self.conv(embedded))
        pooled = self.pool(features).squeeze(2)
        return self.output(pooled).squeeze(1)                  # (batch,)


lstm_classifier = LSTMClassifier().to(DEVICE)
cnn_classifier = CNN1DClassifier().to(DEVICE)
sample_tokens, _ = next(iter(text_train_loader))
with torch.no_grad():
    print("Вход:", tuple(sample_tokens.shape))
    print("Выход LSTM:", tuple(lstm_classifier(sample_tokens.to(DEVICE)).shape))
    print("Выход CNN:", tuple(cnn_classifier(sample_tokens.to(DEVICE)).shape))"""))

nb.add(md("### Обучение и измеренное сравнение"))
nb.add(sol("""@torch.inference_mode()
def evaluate_classifier(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss, total_correct, total_items = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        total_correct += ((logits >= 0) == targets.bool()).sum().item()
        total_items += targets.numel()
    return total_loss / total_items, total_correct / total_items


def train_classifier(model, train_loader, epochs=8, learning_rate=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.numel()
            seen += targets.numel()
        if epoch in {1, epochs}:
            print(f"  эпоха {epoch:02d}: train loss = {running_loss / seen:.4f}")
    elapsed = time.perf_counter() - started
    test_loss, test_accuracy = evaluate_classifier(model, text_test_loader)
    return {"test_loss": test_loss, "test_accuracy": test_accuracy, "seconds": elapsed}


print("LSTM")
lstm_text_result = train_classifier(lstm_classifier, text_train_loader)
print("1D-CNN")
cnn_text_result = train_classifier(cnn_classifier, text_train_loader)

metric_note = (
    "РЕАЛЬНАЯ МЕТРИКА banks.csv"
    if USE_BANKS_CSV
    else "ДЕМОНСТРАЦИОННАЯ МЕТРИКА НА СИНТЕТИЧЕСКИХ ТЕКСТАХ"
)
print("\\n" + metric_note)
text_results = pd.DataFrame([
    {"модель": "LSTM", **lstm_text_result},
    {"модель": "1D-CNN", **cnn_text_result},
])
print(text_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
if not USE_BANKS_CSV:
    print("Эти accuracy нельзя выдавать за качество на banks.csv.")"""))

nb.add(md("""`Conv1d(kernel_size=5)` ищет локальные признаки, похожие на
5-граммы, а `AdaptiveMaxPool1d(1)` выбирает максимальный отклик каждого фильтра
по всему тексту. CNN хорошо параллелится и обычно быстрее. LSTM читает слова
последовательно и хранит состояние, поэтому лучше приспособлена к зависимостям,
разнесённым дальше размера ядра, но обучение обходится дороже. Корректный вывод о
точности нужно делать только после запуска обеих моделей на одном реальном
разбиении `banks.csv`; таблица выше для синтетики проверяет реализацию, а не
заменяет такой эксперимент."""))

# === Часть В. Jena Climate ===
nb.add(md("---\n## Часть В. Прогнозирование Jena Climate"))
nb.add(md("""### Загрузка без Keras

Архив скачивается напрямую по опубликованному адресу и распаковывается стандартным
`zipfile`. Повторный запуск использует локальный CSV из каталога `data/jena_climate`."""))
nb.add(sol("""JENA_URL = (
    "https://storage.googleapis.com/tensorflow/"
    "tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)
JENA_DIR = Path("data/jena_climate")
JENA_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = JENA_DIR / "jena_climate_2009_2016.csv.zip"
CSV_PATH = JENA_DIR / "jena_climate_2009_2016.csv"

if not CSV_PATH.exists():
    if not ZIP_PATH.exists():
        print("Скачивание Jena Climate...")
        urllib.request.urlretrieve(JENA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_members) != 1:
            raise RuntimeError(f"В архиве ожидался один CSV, найдено: {csv_members}")
        with archive.open(csv_members[0]) as source, CSV_PATH.open("wb") as target:
            target.write(source.read())

jena_df = pd.read_csv(CSV_PATH)
feature_names = jena_df.columns[1:].tolist()  # Date Time не подаём в модель
if "T (degC)" not in feature_names:
    raise ValueError("В Jena Climate не найден целевой столбец T (degC)")
temperature_index = feature_names.index("T (degC)")
raw_data = jena_df[feature_names].to_numpy(dtype=np.float32)
print("Размер таблицы:", jena_df.shape)
print("Признаков:", raw_data.shape[1], "цель:", feature_names[temperature_index])"""))

nb.add(md("""### Хронологическое разбиение и train-only нормализация

Первые 50% наблюдений образуют train, следующие 25% — validation, последние
25% — test. Среднее и стандартное отклонение вычисляются **только на train**.
Цель тоже нормализована, поэтому основная метрика ниже — normalized MAE. Для
интерпретации дополнительно печатается MAE в градусах Цельсия."""))
nb.add(sol("""n_rows = len(raw_data)
train_end = int(0.50 * n_rows)
val_end = int(0.75 * n_rows)

train_mean = raw_data[:train_end].mean(axis=0)
train_std = raw_data[:train_end].std(axis=0)
train_std = np.where(train_std < 1e-7, 1.0, train_std)
normalized_data = ((raw_data - train_mean) / train_std).astype(np.float32)

print("Границы по времени:", {"train": train_end, "val": val_end, "test": n_rows})
print(
    "Train mean/std нормализованной температуры:",
    f"{normalized_data[:train_end, temperature_index].mean():.4f}",
    f"{normalized_data[:train_end, temperature_index].std():.4f}",
)"""))

nb.add(md("""### Скользящие окна `Dataset` / `DataLoader`

Исходные измерения сделаны каждые 10 минут. Берём 120 точек с шагом 6, то есть
одно измерение в час примерно за 5 суток. Последняя входная точка находится за
24 часа (144 исходных шага) до цели. Такое индексирование принципиально: цель не
может оказаться внутри входного окна.

Для ограничения времени цели рассматриваются раз в час, а затем равномерно
прореживаются до 30 000 train и 6 000 validation/test примеров. Порядок границ
остаётся хронологическим; перемешивается только train `DataLoader`."""))
nb.add(sol("""SEQUENCE_LENGTH = 120
SAMPLING_RATE = 6
FORECAST_HORIZON = 24 * 6
TARGET_STRIDE = 6
FIRST_TARGET = FORECAST_HORIZON + (SEQUENCE_LENGTH - 1) * SAMPLING_RATE


def select_target_indices(start, end, max_samples):
    first = max(start, FIRST_TARGET)
    candidates = np.arange(first, end, TARGET_STRIDE, dtype=np.int64)
    if len(candidates) > max_samples:
        positions = np.linspace(0, len(candidates) - 1, max_samples, dtype=np.int64)
        candidates = candidates[positions]
    return candidates


class JenaWindowDataset(Dataset):
    def __init__(self, values, target_indices, target_column):
        self.values = values
        self.target_indices = target_indices
        self.target_column = target_column

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, item):
        target_index = int(self.target_indices[item])
        last_input = target_index - FORECAST_HORIZON
        first_input = last_input - (SEQUENCE_LENGTH - 1) * SAMPLING_RATE
        window = self.values[first_input:last_input + 1:SAMPLING_RATE]
        target = self.values[target_index, self.target_column]
        if window.shape[0] != SEQUENCE_LENGTH:
            raise RuntimeError(f"Некорректная длина окна: {window.shape}")
        return torch.from_numpy(window), torch.tensor(target, dtype=torch.float32)


train_targets = select_target_indices(FIRST_TARGET, train_end, 30_000)
val_targets = select_target_indices(train_end, val_end, 6_000)
test_targets = select_target_indices(val_end, n_rows, 6_000)

train_jena = JenaWindowDataset(normalized_data, train_targets, temperature_index)
val_jena = JenaWindowDataset(normalized_data, val_targets, temperature_index)
test_jena = JenaWindowDataset(normalized_data, test_targets, temperature_index)

BATCH_SIZE = 256
loader_options = {
    "batch_size": BATCH_SIZE,
    "num_workers": 0,
    "pin_memory": DEVICE.type == "cuda",
}
train_jena_loader = DataLoader(
    train_jena,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    **loader_options,
)
val_jena_loader = DataLoader(val_jena, shuffle=False, **loader_options)
test_jena_loader = DataLoader(test_jena, shuffle=False, **loader_options)

window_example, target_example = train_jena[0]
assert window_example.shape == (SEQUENCE_LENGTH, raw_data.shape[1])
assert train_targets.max() < train_end <= val_targets.min()
assert val_targets.max() < val_end <= test_targets.min()
print("Число окон train/val/test:", len(train_jena), len(val_jena), len(test_jena))
print("Форма одного окна/цели:", tuple(window_example.shape), tuple(target_example.shape))"""))

nb.add(md("""### Наивный бейзлайн

Для прогноза на 24 часа вперёд бейзлайн повторяет последнюю доступную температуру,
то есть температуру ровно за сутки до целевого момента. Это честный ориентир:
модель полезна только тогда, когда её MAE меньше."""))
nb.add(sol("""@torch.inference_mode()
def evaluate_naive(loader):
    absolute_error, count = 0.0, 0
    for samples, targets in loader:
        predictions = samples[:, -1, temperature_index]
        absolute_error += torch.abs(predictions - targets).sum().item()
        count += targets.numel()
    return absolute_error / count


temperature_scale = float(train_std[temperature_index])
naive_val_mae = evaluate_naive(val_jena_loader)
naive_test_mae = evaluate_naive(test_jena_loader)
print(f"Naive val  normalized MAE: {naive_val_mae:.4f} ({naive_val_mae * temperature_scale:.3f} °C)")
print(f"Naive test normalized MAE: {naive_test_mae:.4f} ({naive_test_mae * temperature_scale:.3f} °C)")"""))

nb.add(md("""### LSTM(16) и stacked GRU

В Keras `recurrent_dropout` маскирует рекуррентные связи внутри шага
`h[t-1] -> h[t]`. У встроенных `torch.nn.LSTM/GRU` такого аргумента нет:
параметр `dropout` накладывает dropout на **выходы между рекуррентными слоями**,
кроме последнего. Поэтому PyTorch-идиоматичный вариант ниже — двухслойная GRU с
`dropout=0.25`. Точное воспроизведение `recurrent_dropout` потребовало бы ручного
цикла на `GRUCell`, было бы существенно медленнее и здесь не подменяется обычным
dropout."""))
nb.add(sol("""class TemperatureLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=16, batch_first=True)
        self.output = nn.Linear(16, 1)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        return self.output(hidden[-1]).squeeze(1)


class StackedTemperatureGRU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size=32,
            num_layers=2,
            dropout=0.25,  # dropout выходов первого слоя перед вторым
            batch_first=True,
        )
        self.output = nn.Linear(32, 1)

    def forward(self, inputs):
        _, hidden = self.gru(inputs)
        return self.output(hidden[-1]).squeeze(1)


@torch.inference_mode()
def evaluate_forecaster(model, loader):
    model.eval()
    absolute_error, count = 0.0, 0
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        absolute_error += torch.abs(model(inputs) - targets).sum().item()
        count += targets.numel()
    return absolute_error / count


def train_forecaster(model, train_loader, val_loader, epochs=8, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_mae = float("inf")
    best_state = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        squared_error, count = 0.0, 0
        for inputs, targets in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            squared_error += loss.item() * targets.numel()
            count += targets.numel()

        val_mae = evaluate_forecaster(model, val_loader)
        print(
            f"  эпоха {epoch:02d}: train MSE={squared_error / count:.4f}, "
            f"val normalized MAE={val_mae:.4f}"
        )
        if val_mae < best_mae - 1e-4:
            best_mae = val_mae
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("  ранняя остановка")
                break

    model.load_state_dict(best_state)
    return time.perf_counter() - started


input_size = raw_data.shape[1]
lstm_forecaster = TemperatureLSTM(input_size).to(DEVICE)
gru_forecaster = StackedTemperatureGRU(input_size).to(DEVICE)

shape_batch, _ = next(iter(train_jena_loader))
with torch.no_grad():
    print("Вход временного ряда:", tuple(shape_batch.shape))
    print("Выход LSTM:", tuple(lstm_forecaster(shape_batch.to(DEVICE)).shape))
    print("Выход GRU:", tuple(gru_forecaster(shape_batch.to(DEVICE)).shape))"""))

nb.add(md("### Реальное обучение и итоговые MAE"))
nb.add(sol("""print("Обучение LSTM(16)")
lstm_seconds = train_forecaster(lstm_forecaster, train_jena_loader, val_jena_loader)
print("Обучение stacked GRU(32 x 2, dropout=0.25)")
gru_seconds = train_forecaster(gru_forecaster, train_jena_loader, val_jena_loader)

lstm_val_mae = evaluate_forecaster(lstm_forecaster, val_jena_loader)
lstm_test_mae = evaluate_forecaster(lstm_forecaster, test_jena_loader)
gru_val_mae = evaluate_forecaster(gru_forecaster, val_jena_loader)
gru_test_mae = evaluate_forecaster(gru_forecaster, test_jena_loader)

jena_results = pd.DataFrame([
    {
        "модель": "Naive last temperature",
        "val normalized MAE": naive_val_mae,
        "test normalized MAE": naive_test_mae,
        "test MAE, °C": naive_test_mae * temperature_scale,
        "train seconds": 0.0,
    },
    {
        "модель": "LSTM(16)",
        "val normalized MAE": lstm_val_mae,
        "test normalized MAE": lstm_test_mae,
        "test MAE, °C": lstm_test_mae * temperature_scale,
        "train seconds": lstm_seconds,
    },
    {
        "модель": "Stacked GRU(32x2)",
        "val normalized MAE": gru_val_mae,
        "test normalized MAE": gru_test_mae,
        "test MAE, °C": gru_test_mae * temperature_scale,
        "train seconds": gru_seconds,
    },
])
print(jena_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

for name, score in [("LSTM", lstm_test_mae), ("GRU", gru_test_mae)]:
    comparison = "лучше" if score < naive_test_mae else "не лучше"
    print(f"{name}: {comparison} test-бейзлайна на {abs(naive_test_mae - score):.4f}")"""))

nb.add(md("""Все значения в итоговой таблице измеряются, а не подставляются.
Обычный ориентир исправного запуска для рекуррентной модели — normalized MAE
порядка 0.25–0.29, но точное число зависит от устройства, версии PyTorch,
подвыборки и ранней остановки. Если модель не превзошла наивный прогноз, сначала
следует увеличить `epochs` или лимит train-окон, а не объявлять ожидаемое число
полученным результатом.

В этой задаче выход — одно непрерывное значение без активации. MSE используется
для обучения, MAE — для понятной оценки. В бинарной классификации выше выход тоже
один, но это логит класса, а функция потерь — бинарная кросс-энтропия.
Наивный бейзлайн проверяет, извлекает ли сеть из истории больше информации, чем
простое сезонное правило «температура как сутки назад»."""))

path = "M5-sequences/attachments/kim-05-sequences-solution.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
