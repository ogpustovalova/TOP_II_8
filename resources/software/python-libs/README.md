# Python-библиотеки и программные средства

| Название | Аннотация | Связанные КИМ | Доступ | Лицензия / условия | Дата проверки |
|---|---|---|---|---|---|
| Python 3 | Базовый язык курса | все | https://www.python.org/ | PSF (открытая) | 2026-07-18 |
| NumPy | Векторные/матричные операции, основа для ручной реализации нейрона и backprop | [М1](../../../M1-neuron-basics/kim-01-neuron-perceptron.md), [М2](../../../M2-training/kim-01-backprop-training.md) | https://numpy.org/ | BSD (открытая) | 2026-07-18 |
| PyTorch | Основной фреймворк глубокого обучения курса (рекомендованный) | все модули | https://pytorch.org/ | BSD (открытая) | 2026-07-18 |
| TensorFlow / Keras | Альтернативный фреймворк курса; высокоуровневый API Keras | все модули | https://www.tensorflow.org/, https://keras.io/ | Apache 2.0 (открытая) | 2026-07-18 |
| scikit-learn | Классическое ML, метрики, подготовка данных (пререквизит) | пререквизит, [М3](../../../M3-dense-networks/kim-01-mlp.md) | https://scikit-learn.org/ | BSD (открытая) | 2026-07-18 |
| Matplotlib | Визуализация (кривые обучения, латентное пространство) | все | https://matplotlib.org/ | PSF (открытая) | 2026-07-18 |
| torchvision | Предобученные модели и датасеты CV для PyTorch | [М4](../../../M4-cnn/kim-01-cnn.md) | https://pytorch.org/vision/ | BSD (открытая) | 2026-07-18 |
| pymorphy3 | Морфологический анализ русского языка (лемматизация) | [М5](../../../M5-sequences/kim-01-sequences.md) | https://pypi.org/project/pymorphy3/ | MIT (открытая) | 2026-07-18 |
| nltk | Токенизация, стоп-слова | [М5](../../../M5-sequences/kim-01-sequences.md) | https://www.nltk.org/ | Apache 2.0 (открытая) | 2026-07-18 |

## Требования к добавлению

Для каждой библиотеки укажите версию, зафиксированную в `requirements.txt`
проектной работы. При обновлении курса проверяйте совместимость версий
PyTorch/TensorFlow с примерами.
