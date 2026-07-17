# Датасеты

Учебные и проектные наборы данных, используемые в КИМ. Для каждого — источник,
лицензия, состав, объём и рекомендуемое использование.

| Название | Аннотация | Связанные КИМ | Доступ | Лицензия / условия | Дата проверки |
|---|---|---|---|---|---|
| Fashion-MNIST | 70 000 изображений одежды 28×28 grayscale, 10 классов; «hello world» нейросетей | [М1](../../M1-neuron-basics/kim-01-neuron-perceptron.md), [М2](../../M2-training/kim-01-backprop-training.md), [М3](../../M3-dense-networks/kim-01-mlp.md), [М6](../../M6-autoencoders/kim-01-autoencoders.md), [М7](../../M7-optimizers/kim-01-optimizers.md) | https://github.com/zalandoresearch/fashion-mnist | MIT (открытый) | 2026-07-18 |
| California Housing | 20 640 записей, 8 числовых признаков, цель — медианная стоимость недвижимости; классическая регрессия | [М3](../../M3-dense-networks/kim-01-mlp.md) | `tf.keras.datasets.california_housing` | открытый (Statlib) | 2026-07-18 |
| CIFAR-10 | 60 000 изображений 32×32 RGB, 10 классов; стандарт CNN-бенчмарк | [М4](../../M4-cnn/kim-01-cnn.md), [М7](../../M7-optimizers/kim-01-optimizers.md) | `tf.keras.datasets.cifar10` / `torchvision.datasets.CIFAR10` | MIT (открытый) | 2026-07-18 |
| Intel Image Classification | ~25 000 изображений сцен 150×150, 6 классов (здания, лес, ледник, горы, море, улица) | [М4](../../M4-cnn/kim-01-cnn.md) (transfer learning) | https://www.kaggle.com/puneet6068/intel-image-classification | Kaggle (нужна регистрация) | 2026-07-18 |
| Animals | изображения животных, 10 классов | [М4](../../M4-cnn/kim-01-cnn.md) (transfer learning, PyTorch) | материалы курса `dlpython_course/04_transfer_learning/pytorch/animals.ipynb` | — | 2026-07-18 |
| Отзывы на банки (banks.csv) | Тексты отзывов с бинарной меткой тональности; классификация текстов | [М5](../../M5-sequences/kim-01-sequences.md) | материалы курса `dlpython_course/05_text_processing/` (ссылка на Dropbox внутри) | — | 2026-07-18 |
| Jena Climate 2009–2016 | ~420 000 измерений метеостанции Йены, 14 признаков; прогноз температуры | [М5](../../M5-sequences/kim-01-sequences.md) | `jena_climate_2009_2016.csv` (Keras S3) | открытый (CC) | 2026-07-18 |
| MNIST | 70 000 рукописных цифр 28×28; базовый датасет для AE/VAE | [М6](../../M6-autoencoders/kim-01-autoencoders.md) | `tf.keras.datasets.mnist` / `torchvision.datasets.MNIST` | открытый | 2026-07-18 |

## Требования к добавлению

Укажите источник, лицензию, состав признаков, целевую переменную, объём,
ограничения и возможные смещения. Для проектной работы — рекомендуемое
разбиение train/val/test.
