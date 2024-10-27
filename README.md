# Методы ИИ для решения задач устойчивого развития. Анализ последствий наводнений
### :hammer_and_wrench: Languages and Tools :

<div align="center">
  <img src="https://raw.githubusercontent.com/devicons/devicon/1119b9f84c0290e0f0b38982099a2bd027a48bf1/icons/python/python-original.svg" height="40" width="40">
  <img src="https://groups.io/img/org.1/mainlogo.png" height="40" width="auto">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png" height="40">
</div>

## :hammer: Pipeline
Создание виртуального окружения:
```
pythom -m venv venv
\venv\Scripts\Activate.ps1
```
Установка необходимых пакетов:

```
pip install -r ./requirements.txt
```

Описание файлов:

В папке Work with data расположены нотбуки с первичным анализом данным, который включает в себя построение гистрограмм, деление изображения на мелкие части.
В корневой папке есть слеудующие ноутбуки: train_5bits.ipynb - пайплайн для обучения модели; model_inference.ipynb - для импорта модели, предсказания, сабмита с киллер-фичей.
Для корректной работы кода, необходима скачать и поместить в корневую папку проекта модель по ссылке https://drive.google.com/drive/folders/10whLgzLCokluP5qeUHQZEzK5lF5xlHZv?usp=sharing.
## :moyai: Описание приложения

Мы представляем наше решение, которое <b>сегментирует затопленные территории и определяет пострадавшие объекты инфраструктуры </b> на спутниковом мультиспектральном снимке. Решение реализовано на языке программирования Python, с использованием библиотеки PyTorch и модели нейросети DeepLab V3. Киллер-фичей нашего решения является расчет приблизительных убытков в ходе паводков и обращение к открытому api кадастровых данных, благодаря чему наше решение является масштабируемым и расширяемым для любых территорий.

С демонстрацией решения можно ознакомиться [здесь](https://drive.google.com/drive/folders/1HrIBMxQ6ovTuIr6Xnafh75d2edptDGsr?usp=sharing).
