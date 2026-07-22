# Модуль 2. Обучение искусственных нейронных сетей

Раздел РПД №2. Механика обучения нейронных сетей: функции потерь, градиентный
спуск, алгоритм обратного распространения ошибки (backprop), обнаружение
переобучения и методы регуляризации (включая слой Dropout). Модуль закладывает
интуицию того, как градиенты реально текут через вычислительный граф.

## КИУ (КРМ v3.0)

| Компетенция | Индикатор | Уровень |
|---|---|:---:|
| DL-1. Архитектуры глубоких НС | DL-1.1 — математические основы НС | Б |

## КИМ модуля

| КИМ | Аннотация | Дескриптор | Формат |
|---|---|---|---|
| [КИМ 2.1. Backprop и обучение сети](kim-01-backprop-training.md) | Backprop на NumPy, выбор активации, связности и инициализации, градиентный спуск, переобучение и регуляризация | Реализует backprop, экспериментально выбирает конфигурацию обучения и применяет регуляризацию | Ноутбук + защита |

## Связанные ресурсы

- [Python-библиотеки](../resources/software/python-libs/README.md) — NumPy, PyTorch, TensorFlow/Keras
- [Датасеты](../resources/datasets/README.md) — Fashion-MNIST
- Исходные материалы курса: [`dlpython_course/01_introduction/fashion_mnist_prevent_overfitting.ipynb`](https://github.com/sozykin/dlpython_course),
  [`dlpython_course/01_introduction/pytorch/fashion_mnist_dense.ipynb`](https://github.com/sozykin/dlpython_course)
