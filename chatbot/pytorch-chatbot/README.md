# Chatbot  
Банковский Telegram-бот на Python с LSTM-моделью для обработки естественного языка.

Основные функции:
✅ Финансовые операции:

Проверка баланса (/balance)

Перевод средств (/transfer)

Блокировка карты (/block_card)

История транзакций (/transactions)

Поддержка (/support)

✅ Технологии:

AI: LSTM-модель (PyTorch) для NLP (анализ интентов из intents.json)

Бот: aiogram (асинхронный Telegram API)

Логирование: logging (файлы bot.log + консоль)

Безопасность: FSM (Finite State Machine) для многошаговых операций (перевод, поддержка)

Логирование:
📝 Чат: вопросы/ответы пользователей, команды, ошибки.
🤖 Модель: теги запросов, выбранные ответы.
⚙ Обучение модели: loss/accuracy, время выполнения, лучшие эпохи.


## Installation

### Create an environment
Whatever you prefer (e.g. `conda` or `venv`)
```console
mkdir myproject
$ cd myproject
$ python3 -m venv venv
```

### Activate it
Mac / Linux:
```console
. venv/bin/activate
```
Windows:
```console
venv\Scripts\activate
```
### Install PyTorch and dependencies

For Installation of PyTorch see [official website](https://pytorch.org/).

You also need `nltk`:
 ```console
pip install nltk
 ```

If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python bot.py
```
## Customize
Have a look at [intents.json](intents.json). You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the chat bot. You have to re-run the training whenever this file is modified.
```console
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```
