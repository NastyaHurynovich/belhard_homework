## Перед запуском тренировки распаковать архив с данными
так должны быть представлены данные для обучения модели
data/
├── facades/
      ├── train/
      ├── val/
      └── test/

## Команда для запуска тренировки
python pix2pix.py

## Команда для обработки эскиза
python main.py --input sketch.jpg --output result.png
