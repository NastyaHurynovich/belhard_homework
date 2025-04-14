import logging
import os
import random
import json
import torch
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, Text
from aiogram.dispatcher.filters.state import State, StatesGroup
from dotenv import load_dotenv

from nltk_utils import tokenize, stem
from model import LSTMModel


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.environ.get("TOKEN")

if not TOKEN:
    logger.error("Токен не найден в переменных окружения")
    raise Exception("Токен не найден")

# Инициализация бота
storage = MemoryStorage()
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=storage)

# Загрузка модели и данных
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)


FILE = "data.pth"
data = torch.load(FILE, map_location=device)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

ignore_words = ['?', '.', '!']

model = LSTMModel(input_size, hidden_size, output_size, num_layers=2).to(device)
model.load_state_dict(model_state)
model.eval()


class UserState(StatesGroup):
    transaction = State()
    support = State()


def get_response(msg: str) -> str:
    """Получение ответа от модели на основе сообщения пользователя"""
    try:
        logger.info(f"Получен запрос: '{msg}'")

        words = tokenize(msg)
        words = [stem(w) for w in words if w not in ignore_words]
        bag = [1 if word in words else 0 for word in all_words]
        bag = torch.tensor(bag, dtype=torch.float32).to(device)

        output = model(bag.unsqueeze(0))
        _, predicted = torch.max(output, 1)
        tag = tags[predicted.item()]

        logger.info(f"Модель определила тег: '{tag}'")

        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                logger.info(f"Выбран ответ: '{response}'")
                return response

        logger.warning(f"Не найден подходящий ответ для тега: '{tag}'")
        return "Извините, я не понял ваш запрос. Пожалуйста, попробуйте сформулировать иначе."

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса '{msg}': {str(e)}")
        return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."


@dp.message_handler(Command(commands=['start', 'help']))
async def send_welcome(message: types.Message):
    """Обработка команд start и help"""
    welcome_text = (
        "🏦 Добро пожаловать в банковский чат-бот!\n\n"
        "Я могу помочь вам с:\n"
        "• Проверкой баланса - /balance\n"
        "• Переводом средств - /transfer\n"
        "• Блокировкой карты - /block_card\n"
        "• Историей транзакций - /transactions\n"
        "• Связью с поддержкой - /support\n\n"
        "Просто напишите ваш запрос или выберите команду из меню."
    )
    await message.reply(welcome_text)


@dp.message_handler(commands=['balance'])
async def check_balance(message: types.Message):
    """Проверка баланса"""
    response = get_response("баланс")
    await message.reply(response)


@dp.message_handler(Command("transfer"))
async def transfer_money(message: types.Message):
    """Начало процесса перевода"""
    await UserState.transaction.set()
    await message.reply(
        "💸 Введите сумму и номер счета для перевода в формате:\n"
        "<сумма> <номер счета>\n\n"
        "Например: 1000 40702810500000012345\n\n"
        "Для отмены введите /cancel"
    )


@dp.message_handler(state=UserState.transaction)
async def process_transfer(message: types.Message, state: FSMContext):
    """Обработка перевода"""
    try:
        amount, account = message.text.split()
        # Здесь должна быть логика обработки перевода
        await message.reply(f"✅ Перевод на сумму {amount} руб. на счет {account} выполнен успешно.")
    except ValueError:
        await message.reply("❌ Неверный формат. Пожалуйста, введите сумму и номер счета через пробел.")
    finally:
        await state.finish()


@dp.message_handler(Command("block_card"))
async def block_card(message: types.Message):
    """Блокировка карты"""
    response = get_response("блокировка карты")
    await message.reply(response)


@dp.message_handler(Command("transactions"))
async def show_transactions(message: types.Message):
    """Показать историю транзакций"""
    response = get_response("история транзакций")
    await message.reply(response)


@dp.message_handler(Command("support"))
async def contact_support(message: types.Message):
    """Обращение в поддержку"""
    await UserState.support.set()
    await message.reply(
        "🛟 Опишите вашу проблему, и мы постараемся помочь!\n\n"
        "Для отмены введите /cancel"
    )


@dp.message_handler(state=UserState.support)
async def process_support(message: types.Message, state: FSMContext):
    """Обработка обращения в поддержку"""
    await message.reply("✅ Ваш запрос передан в поддержку. Ожидайте ответа в ближайшее время.")
    await state.finish()


@dp.message_handler(Command("cancel"), state="*")
@dp.message_handler(Text(equals="отмена", ignore_case=True), state="*")
async def cancel_handler(message: types.Message, state: FSMContext):
    """Отмена текущего действия"""
    current_state = await state.get_state()
    if current_state is None:
        return

    await state.finish()
    await message.reply("❌ Действие отменено.", reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler()
async def handle_message(message: types.Message):
    """Обработка произвольных сообщений"""
    response = get_response(message.text)
    await message.reply(response)


if __name__ == "__main__":
    logger.info("Запуск банковского чат-бота...")
    try:
        executor.start_polling(dp, skip_updates=True)
    except Exception as e:
        logger.critical(f"Ошибка при работе бота: {str(e)}")
        raise
    finally:
        logger.info("Бот остановлен")