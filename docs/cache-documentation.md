# Документация кэшей и датасетов

Все относительные пути и команды ниже указаны от корня репозитория. При запуске
через `nbconvert` рабочим каталогом notebook является `M*/attachments/`, поэтому
используемый в коде путь `./data` соответствует `M*/attachments/data/`. Эти
каталоги, скачанные веса и сгенерированные артефакты игнорируются Git.

Команды очистки удаляют только повторно загружаемые кэши. Перед их выполнением
следует завершить notebooks, использующие соответствующие файлы.

## Fashion-MNIST

| Поле | Значение |
|---|---|
| Локальный путь | `M1-neuron-basics/attachments/data/FashionMNIST/`, аналогичные каталоги в M2, M3 и M7 |
| Источник | [`torchvision.datasets.FashionMNIST`](https://docs.pytorch.org/vision/0.20/generated/torchvision.datasets.FashionMNIST.html) |
| Заявленная лицензия | [MIT](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE) |
| Revision/версия | `torchvision 0.20.1`; версия набора источником отдельно не обозначена |
| SHA-256 | Не закреплён; целостность архивов проверяет `torchvision` |
| Способ повторной загрузки | Запустить соответствующий notebook либо передать нужный `M*/attachments/data` как `root` в `FashionMNIST(..., download=True)` |
| Команда проверки | `for d in M1-neuron-basics M2-training M3-dense-networks M7-optimizers; do test -d "$d/attachments/data/FashionMNIST" || exit 1; done` |
| Безопасная команда очистки | `rm -rf M1-neuron-basics/attachments/data/FashionMNIST M2-training/attachments/data/FashionMNIST M3-dense-networks/attachments/data/FashionMNIST M7-optimizers/attachments/data/FashionMNIST` |

## MNIST

| Поле | Значение |
|---|---|
| Локальный путь | `M6-autoencoders/attachments/data/MNIST/` |
| Источник | [`torchvision.datasets.MNIST`](https://docs.pytorch.org/vision/0.20/generated/torchvision.datasets.MNIST.html) |
| Заявленная лицензия | [CC BY-SA 3.0](https://yann.lecun.com/exdb/mnist/) |
| Revision/версия | `torchvision 0.20.1`; версия набора источником отдельно не обозначена |
| SHA-256 | Не закреплён; целостность архивов проверяет `torchvision` |
| Способ повторной загрузки | `python -c "from torchvision import datasets; datasets.MNIST(root='M6-autoencoders/attachments/data', download=True)"` |
| Команда проверки | `test -d M6-autoencoders/attachments/data/MNIST` |
| Безопасная команда очистки | `rm -rf M6-autoencoders/attachments/data/MNIST` |

## California Housing

| Поле | Значение |
|---|---|
| Локальный путь | `<sklearn-data-home>/cal_housing_py3.pkz`; каталог выводит `python -c "from sklearn.datasets import get_data_home; print(get_data_home())"` |
| Источник | [`sklearn.datasets.fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) |
| Заявленная лицензия | StatLib и scikit-learn не указывают отдельную лицензию данных; требуется цитировать Pace и Barry (1997) и не публиковать копию |
| Revision/версия | `scikit-learn 1.9.0` |
| SHA-256 | `740b911884b5405599c461cf5c9a620aad36c63a0d32494567de2ebee803a42f` для проверенной копии |
| Способ повторной загрузки | `python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"` |
| Команда проверки | `python -c "from sklearn.datasets import fetch_california_housing; assert fetch_california_housing().data.shape == (20640, 8)"` |
| Безопасная команда очистки | `rm -f "$(python -c 'from sklearn.datasets import get_data_home; print(get_data_home())')/cal_housing_py3.pkz"` |

## CIFAR-10

| Поле | Значение |
|---|---|
| Локальный путь | `M4-cnn/attachments/data/cifar-10-batches-py/` и `M4-cnn/attachments/data/cifar-10-python.tar.gz` |
| Источник | [`torchvision.datasets.CIFAR10`](https://docs.pytorch.org/vision/0.20/generated/torchvision.datasets.CIFAR10.html) |
| Заявленная лицензия | Отдельная лицензия данных не опубликована; автор требует цитировать технический отчёт |
| Revision/версия | `torchvision 0.20.1`; версия набора источником отдельно не обозначена |
| SHA-256 | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` для архива |
| Способ повторной загрузки | `python -c "from torchvision import datasets; datasets.CIFAR10(root='M4-cnn/attachments/data', download=True)"` |
| Команда проверки | `sha256sum M4-cnn/attachments/data/cifar-10-python.tar.gz` |
| Безопасная команда очистки | `rm -rf M4-cnn/attachments/data/cifar-10-batches-py M4-cnn/attachments/data/cifar-10-python.tar.gz` |

## Russian Reviews Classification

| Поле | Значение |
|---|---|
| Локальный путь | `M5-sequences/attachments/data/ru_reviews/train.jsonl` |
| Источник | [`ai-forever/ru-reviews-classification`](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) |
| Заявленная лицензия | Apache-2.0 в метаданных Hugging Face; происхождение отзывов не описано |
| Revision/версия | `0f67d914f396ce22917dc6463ec619799b3b08d2` |
| SHA-256 | `0b97698a0c6871437d17e07c973018af9b8c9230ec9048d85cb875cc2c2470ea` |
| Способ повторной загрузки | `ensure_cached_file` с атомарной заменой и проверкой SHA-256; см. `tools/sol_m5.py:239-279` |
| Команда проверки | `sha256sum M5-sequences/attachments/data/ru_reviews/train.jsonl` |
| Безопасная команда очистки | `rm -rf M5-sequences/attachments/data/ru_reviews` |

## Jena Climate 2009-2016

| Поле | Значение |
|---|---|
| Локальный путь | `M5-sequences/attachments/data/jena_climate/` |
| Источник | [`jena_climate_2009_2016.csv.zip`](https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip) |
| Заявленная лицензия | [CC BY 4.0](https://www.bgc-jena.mpg.de/wetter/) (Max Planck Institute for Biogeochemistry) |
| Revision/версия | Учебная копия Keras; отдельная revision не опубликована |
| SHA-256 | `63d757501e92284a7de7cdbef0337f03b24e13ead7ac2b5b8c86a18d8e38ba5b` для архива |
| Способ повторной загрузки | `urllib.request.urlretrieve` и `zipfile.ZipFile`; см. `tools/sol_m5.py:641-660` |
| Команда проверки | `test "$(wc -l < M5-sequences/attachments/data/jena_climate/jena_climate_2009_2016.csv)" -gt 420000` |
| Безопасная команда очистки | `rm -rf M5-sequences/attachments/data/jena_climate` |

## Веса ResNet18

| Поле | Значение |
|---|---|
| Локальный путь | `<torch-hub>/checkpoints/resnet18-f37072fd.pth`; каталог выводит `python -c "import torch; print(torch.hub.get_dir())"` |
| Источник | [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) через `ResNet18_Weights.DEFAULT` |
| Заявленная лицензия | Код torchvision - BSD-3-Clause; для весов, обученных на ImageNet-1K, отдельная лицензия не заявлена |
| Revision/версия | `ResNet18_Weights.IMAGENET1K_V1`, `torchvision 0.20.1` |
| SHA-256 | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| Способ повторной загрузки | `python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"` |
| Команда проверки | `sha256sum "$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints/resnet18-f37072fd.pth"` |
| Безопасная команда очистки | `rm -f "$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints/resnet18-f37072fd.pth"` |

## YOLO и CV-видеопайплайн M4

| Поле | Значение |
|---|---|
| Локальный путь | `M4-cnn/attachments/yolo11n.pt` и `M4-cnn/attachments/data/cv_pipeline/` |
| Источник | `YOLO('yolo11n.pt')` из `ultralytics 8.4.104`; тестовое изображение [`bus.jpg`](https://ultralytics.com/images/bus.jpg) |
| Заявленная лицензия | Ultralytics - AGPL-3.0; для тестового изображения отдельная лицензия не опубликована |
| Revision/версия | YOLO11 nano, `ultralytics 8.4.104` |
| SHA-256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` для `yolo11n.pt`; производные видео и сохранённая модель зависят от запуска |
| Способ повторной загрузки | Запустить M4: веса получает `YOLO`, изображение - `urllib.request.urlretrieve`, видео создаётся из 30 кадров |
| Команда проверки | `sha256sum M4-cnn/attachments/yolo11n.pt && test -s M4-cnn/attachments/data/cv_pipeline/detections.mp4` |
| Безопасная команда очистки | `rm -rf M4-cnn/attachments/yolo11n.pt M4-cnn/attachments/data/cv_pipeline` |

## Конфигурационный кэш Ultralytics

| Поле | Значение |
|---|---|
| Локальный путь | Каталог выводит `python -c "from ultralytics.utils import USER_CONFIG_DIR; print(USER_CONFIG_DIR)"` |
| Источник | Автоматически создаётся пакетом [`ultralytics`](https://pypi.org/project/ultralytics/) |
| Заявленная лицензия | AGPL-3.0 |
| Revision/версия | Формат settings `0.0.6`, `ultralytics 8.4.104` |
| SHA-256 | Не фиксируется: содержит локальные пути и идентификатор установки; каталог не входит в репозиторий |
| Способ повторной загрузки | Удалённая конфигурация создаётся при следующем `import ultralytics` |
| Команда проверки | `python -c "from ultralytics.utils import SETTINGS; print(SETTINGS.file)"` |
| Безопасная команда очистки | `rm -rf "$(python -c 'from ultralytics.utils import USER_CONFIG_DIR; print(USER_CONFIG_DIR)')"` |

## banks.csv (не используется обязательными КИМ)

| Поле | Значение |
|---|---|
| Локальный путь | Не создаётся штатными notebooks |
| Источник | Исходный notebook курса dlpython; карточка сохранена в `resources/datasets/README.md` |
| Заявленная лицензия | MIT по метаданным Kaggle `nikee7/t1-sentiment-classification-russian-reviews` |
| Revision/версия | Не зафиксирована |
| SHA-256 | `aaef2536652b8590e878962e08e7bd21bd0f17bc2f5b9196542e7e414540dc5c` для проверенной копии из 13 999 записей |
| Способ повторной загрузки | Использовать URL из исходного `text_classification.ipynb` и сверить SHA-256 |
| Команда проверки | `sha256sum banks.csv` |
| Безопасная команда очистки | `rm -f banks.csv` |

В обязательном M5 вместо `banks.csv` используется закреплённая версия Russian
Reviews Classification.
