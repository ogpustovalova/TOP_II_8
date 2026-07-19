"""Сборка ноутбука КИМ 4.1 — Сверточные сети и перенос обучения (Модуль 4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from nb_builder import Notebook, md, code, placeholder

nb = Notebook("КИМ 4.1 — Сверточные сети и перенос обучения")

nb.add(md("""# КИМ 4.1. Сверточные сети и перенос обучения

**Модуль 4. Сверточные нейронные сети** · Курс «Основы нейронных сетей» · УрФУ

Проверяемые результаты (КРМ v3.0):
- **DL-1.3 (Б):** применяет VGG-подобную архитектуру и предобученную ResNet.
- **DL-1.4 (Б):** исследует гиперпараметры свёртки и пулинга, разрабатывает CNN.
- **DL-1.12 (Б):** применяет перенос обучения, fine-tuning и малую выборку.
- **DL-3.1 (Б):** применяет ResNet и YOLO к изображениям и видео, использует
  OpenCV, сохраняет и повторно загружает модель.

Подробное описание, критерии и контрольные вопросы — в `kim-01-cnn.md`."""))

# === Часть А. Свёртка с нуля ===
nb.add(md("""---
## Часть А. Свёртка с нуля

Операция двумерной свёртки:

$$(I \\star K)[i,j] = \\sum_m \\sum_n I[i+m, j+n] \\cdot K[m, n]$$

Размер выхода при шаге $S$ и дополнении $P$:

$$H_{out} = \\left\\lfloor \\frac{H_{in} - K + 2P}{S} \\right\\rfloor + 1$$"""))

nb.add(md("### 0. Импорт библиотек"))
nb.add(placeholder())

nb.add(md("""### 1. Реализация свёртки на NumPy

Реализуйте `conv2d(image, kernel, stride=1, padding=0)` для одноканального
изображения. Проверьте на маленьком примере (матрица 5×5, ядро 3×3)."""))
nb.add(placeholder("conv2d"))

nb.add(md("""### 2. Влияние параметров

Покажите на одном изображении, как меняется размер выхода при разных `kernel_size`
(3, 5), `stride` (1, 2) и `padding` (0, 'same'). Сравните с формулой размера выхода."""))
nb.add(placeholder("эксперимент с параметрами"))

# === Часть Б. Собственная CNN на CIFAR-10 ===
nb.add(md("""---
## Часть Б. Собственная CNN на CIFAR-10

CIFAR-10: 60 000 RGB-изображений 32×32, 10 классов."""))

nb.add(md("### 3. Загрузка CIFAR-10 и подготовка"))
nb.add(placeholder("load + /255 + one-hot"))

nb.add(md("""### 4. Архитектура CNN в стиле VGG

Соберите CNN: каскады `Conv2D → ReLU → MaxPooling → Dropout`, затем `Flatten →
Dense → Dense(10, softmax)`."""))
nb.add(placeholder("Sequential CNN"))

nb.add(md("### 5. Обучение и оценка"))
nb.add(placeholder("compile(sparse_categorical_crossentropy, Adam) + fit + evaluate"))

nb.add(md("""Обучите с `validation_split=0.2`, `batch_size=128`, `epochs=20–30`.
Достигните тестовой точности не ниже 65 %."""))

nb.add(md("### 6. Кривые обучения"))
nb.add(placeholder("графики accuracy/loss для train и val"))

# === Часть В. Transfer learning ===
nb.add(md("""---
## Часть В. Предобученная ResNet и перенос обучения

### 7. Загрузка предобученной модели

Загрузите `ResNet50(weights='imagenet', include_top=False)` (Keras) или
`torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` (PyTorch)."""))
nb.add(placeholder("load pretrained ResNet"))

nb.add(md("""### 8. Этап 1 — Feature extraction (заморозка backbone)

Заморозьте backbone: `base_model.trainable = False`. Добавьте новый классификатор
(`GlobalAveragePooling2D → Dense → Dense(N, softmax)`). Обучите **только голову**."""))
nb.add(placeholder("freeze + new head + fit"))

nb.add(md("""### 9. Этап 2 — Fine-tuning

Разморозьте backbone, перекомпилируйте с **низкой** скоростью обучения
(`Adam(learning_rate=1e-5)`), дообучите. Сравните точность до и после."""))
nb.add(placeholder("unfreeze + compile(1e-5) + fit"))

nb.add(md("""**Контрольный вопрос:** почему при fine-tuning уменьшают скорость
обучения? Что произойдёт при `lr=0.01`?"""))
nb.add(md("> *Ваш ответ:* ..."))

nb.add(md("""### 10. Сравнение

Сведите в таблицу: собственная CNN vs. transfer learning (заморозка) vs.
transfer learning + fine-tuning. Сравните точность, время обучения и число
обучаемых параметров."""))
nb.add(placeholder("таблица сравнения"))

# === Часть Г. Малая выборка ===
nb.add(md("""---
## Часть Г. Transfer learning на малой выборке

### 11. Сбалансированная few-shot-выборка

Выберите по 20 изображений каждого класса CIFAR-10 только из train. На одном и
том же разбиении обучите новую голову замороженной ResNet, затем выполните
fine-tuning с малым lr. Сравните оба режима и результат на полной выборке.
Зафиксируйте seed и выведите таблицу `режим — число объектов — val/test accuracy
— время`."""))
nb.add(placeholder("20 изображений на класс + frozen/fine-tuning + таблица"))

# === Часть Д. CV-видеопайплайн ===
nb.add(md("""---
## Часть Д. YOLO и видеопайплайн OpenCV

### 12. Инференс, сохранение и повторная загрузка

Используйте готовую YOLO и открытое тестовое изображение с несколькими объектами.
Через OpenCV соберите или прочитайте короткое видео, выполните покадровый
инференс и сохраните размеченный файл `detections.mp4`. Сохраните модель,
повторно загрузите её и проверьте совпадение классов и confidence на одном
контрольном кадре. В отчёт включите пути к файлам и краткое описание цикла
`VideoCapture/VideoWriter`."""))
nb.add(placeholder("YOLO + OpenCV video + save/load consistency check"))

nb.add(md("""---
## Итог

Перед сдачей убедитесь, что:
- [ ] свёртка реализована, формула размера выхода проверена;
- [ ] собственная CNN обучена на CIFAR-10 (точность ≥ 65 %);
- [ ] выполнен двухэтапный transfer learning (заморозка + fine-tuning);
- [ ] проведён few-shot-эксперимент на 20 изображениях каждого класса;
- [ ] сохранены YOLO-модель и размеченное видео, проверена повторная загрузка;
- [ ] вы готовы объяснить residual connection и роль `lr=1e-5`.

**Формат сдачи:** заполненный ноутбук + устная защита."""))

path = "M4-cnn/attachments/kim-04-cnn.ipynb"
nb.save(path)
print(f"Сохранён: {path}  ({nb.cell_count()} ячеек)")
