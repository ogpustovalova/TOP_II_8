# Датасеты

Учебные и проектные наборы данных, используемые в КИМ. Лицензия библиотеки-
загрузчика не считается лицензией набора данных. Если первоисточник не публикует
отдельную лицензию, карточка прямо запрещает включать копию данных в репозиторий.

## Активные наборы

| Название | Аннотация | Связанные КИМ | Доступ | Лицензия / условия | Дата проверки |
|---|---|---|---|---|---|
| Fashion-MNIST | 70 000 изображений одежды 28×28 grayscale, 10 классов; базовая классификация изображений | [М1](../../M1-neuron-basics/kim-01-neuron-perceptron.md), [М2](../../M2-training/kim-01-backprop-training.md), [М3](../../M3-dense-networks/kim-01-mlp.md), [М6](../../M6-autoencoders/kim-01-autoencoders.md), [М7](../../M7-optimizers/kim-01-optimizers.md) | [репозиторий и данные](https://github.com/zalandoresearch/fashion-mnist) | [MIT](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE): сохранять copyright и текст лицензии | 2026-07-19 |
| California Housing | 20 640 записей, 8 числовых признаков, цель — медианная стоимость недвижимости; регрессия | [М3](../../M3-dense-networks/kim-01-mlp.md) | [`keras.datasets.california_housing`](https://keras.io/api/datasets/california_housing/) | StatLib и Keras не указывают отдельную лицензию данных: загружать из источника, цитировать Pace и Barry (1997), не включать копию в репозиторий | 2026-07-22 |
| CIFAR-10 | 60 000 изображений 32×32 RGB, 10 классов; стандартный CNN-бенчмарк | [М4](../../M4-cnn/kim-01-cnn.md), [М7](../../M7-optimizers/kim-01-optimizers.md) | [официальная страница](https://www.cs.toronto.edu/~kriz/cifar.html) через Keras/torchvision | Отдельная лицензия данных не опубликована; автор требует цитировать технический отчёт. Не включать архив в репозиторий | 2026-07-19 |
| Russian Reviews Classification | 75 000 русскоязычных отзывов, сплиты train/validation/test, метки negative/neutral/positive; протокол М5 использует только фиксированную подвыборку из train | [М5](../../M5-sequences/kim-01-sequences.md) | [`train.jsonl`, revision `0f67d914...`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification/resolve/0f67d914f396ce22917dc6463ec619799b3b08d2/train.jsonl) | В метаданных указана [Apache-2.0](https://huggingface.co/datasets/ai-forever/ru-reviews-classification/blob/0f67d914f396ce22917dc6463ec619799b3b08d2/README.md), но происхождение отзывов и основание лицензирования не описаны; не включать копию в репозиторий до подтверждения прав | 2026-07-22 |
| Jena Climate 2009–2016 | ~420 000 измерений метеостанции Йены, 14 признаков; прогноз температуры | [М5](../../M5-sequences/kim-01-sequences.md) | [Max Planck Institute for Biogeochemistry](https://www.bgc-jena.mpg.de/wetter/weather_data.html), учебная копия Keras | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): требуется указание источника и изменений | 2026-07-19 |
| MNIST | 70 000 рукописных цифр 28×28; базовый датасет для AE/VAE | [М6](../../M6-autoencoders/kim-01-autoencoders.md) | [`keras.datasets.mnist`](https://keras.io/api/datasets/mnist/) / torchvision | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/): атрибуция и та же лицензия для производных наборов | 2026-07-19 |

## Исключённые источники

Эти наборы встречались в исходных материалах, но не используются в текущих
обязательных КИМ до подтверждения происхождения и прав.

| Название | Проверенный источник | Причина исключения | Дата проверки |
|---|---|---|---|
| Intel Scene Classification | [архивная страница Analytics Vidhya](https://www.analyticsvidhya.com/datahack/contest/practice-problem-intel-scene-classification-challe/) | Соревнование закрыто, для скачивания нужна регистрация, отдельная лицензия данных не опубликована | 2026-07-19 |
| Animals, 10 классов | локальная ссылка из исходного [`dlpython_course`](https://github.com/sozykin/dlpython_course) | Нет воспроизводимого первоисточника и лицензии; копирование и использование запрещены до подтверждения прав | 2026-07-19 |
| Отзывы на банки (`banks.csv`) | ссылка Dropbox из исходного [`dlpython_course`](https://github.com/sozykin/dlpython_course) | Нет устойчивого первоисточника и лицензии; в М5 заменён на Russian Reviews Classification | 2026-07-19 |

## Протокол Russian Reviews в М5

Закреплённый `train.jsonl` имеет SHA-256
`0b97698a0c6871437d17e07c973018af9b8c9230ec9048d85cb875cc2c2470ea`. В М5
нейтральные отзывы исключаются, затем при `seed=42` выбираются по 500 объектов
negative/positive и локально делятся стратифицированно на train/validation/test
640/160/200. Эти части целиком получены из опубликованного сплита `train`;
локальный test не является официальным test набора, и его метрики нельзя
публиковать как метрики official test.

Закрепление revision и хэша обеспечивает техническую воспроизводимость, но не
устраняет юридическую неопределённость: карточка набора содержит только
лицензионную метку без описания источника текстов.

## Требования к добавлению

Укажите первоисточник, лицензию именно данных, состав признаков, целевую
переменную, объём, ограничения и возможные смещения. Для проектной работы
добавьте рекомендуемое разбиение train/val/test. Если лицензия не опубликована,
не называйте набор открытым и не загружайте его копию в репозиторий.
