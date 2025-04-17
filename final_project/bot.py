import logging
import os
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from main import PoetTransformer, PoetDataset

from dotenv import load_dotenv

# логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot1.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.environ.get("TOKEN")
CHECKPOINT_PATH = "check/checkpoints/checkpoint_epoch_2.pt"
JSON_PATH = "classic_poems.json"


bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Глобальные переменные
model = None
dataset = None


class UserState:
    def __init__(self):
        self.temperature = 1.0
        self.max_len = 100
        self.author_id = 0


user_states = {}


def load_model():
    """Загрузка модели"""
    global model, dataset

    checkpoint = torch.load(CHECKPOINT_PATH)
    dataset = PoetDataset(JSON_PATH, max_length=32)

    model = PoetTransformer(
        vocab_size=checkpoint['vocab_size'],
        num_authors=checkpoint['num_authors'],
        d_model=128,
        nhead=4,
        num_layers=4
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    model.dataset = dataset
    model.load_state_dict(checkpoint['model_state'])
    model.eval()


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    """Обработка команды /start с отображением настроек по умолчанию"""
    user_id = message.from_user.id
    user_states[user_id] = UserState()  # Инициализация с настройками по умолчанию

    # Получаем текущие настройки
    state = user_states[user_id]

    text = (
        "👋 Добро пожаловать в генератор стихов!\n\n"
        "Отправьте мне начало стихотворения, и я продолжу его в стиле выбранного автора.\n\n"
        "⚙️ Текущие настройки по умолчанию:\n"
        f"• Температура: {state.temperature}\n"
        f"• Длина текста: {state.max_len} символов\n"
        f"• Автор: {dataset.authors[state.author_id]}\n\n"
        "Используйте команды:\n"
        "/settings - изменить параметры\n"
        "/help - справка"
    )
    await message.answer(text)


@dp.message(Command('help'))
async def cmd_help(message: types.Message):
    """Обработка команды /help"""
    help_text = (
        "📖 Помощь по боту:\n\n"
        "Просто отправьте мне начало стихотворения, и я продолжу его!\n\n"
        "Команды:\n"
        "/settings - настройки\n"
        "/help - эта справка\n\n"
        "Пример: \"Луна над рекой\""
    )
    await message.answer(help_text)


@dp.message(Command('settings'))
async def cmd_settings(message: types.Message):
    """Меню настроек с подробным описанием"""
    user_id = message.from_user.id
    if user_id not in user_states:
        user_states[user_id] = UserState()  # Инициализация с настройками по умолчанию

    state = user_states[user_id]

    # Создаем клавиатуру
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(
            text=f"✏️ Температура ({state.temperature})",
            callback_data="set_temp"
        ),
        types.InlineKeyboardButton(
            text=f"👨‍🎨 Автор ({dataset.authors[state.author_id]})",
            callback_data="set_author"
        ),
        types.InlineKeyboardButton(
            text=f"📏 Длина ({state.max_len} симв.)",
            callback_data="set_len"
        )
    )
    builder.adjust(1)  # По одной кнопке в ряд

    # Описание параметров
    settings_info = (
        "⚙️ Настройки генерации:\n\n"
        "1. Температура - влияет на креативность:\n"
        "   • 0.1-0.5 - более предсказуемые тексты\n"
        "   • 0.5-1.0 - баланс креативности и логики\n"
        "   • 1.0-2.0 - более неожиданные результаты\n\n"
        "2. Длина текста - количество генерируемых символов\n\n"
        "3. Автор - стиль генерации под конкретного поэта"
    )

    await message.answer(
        text=settings_info,
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda c: c.data == 'set_temp')
async def set_temperature(callback_query: types.CallbackQuery):
    """Выбор температуры"""
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(
            text="0.5 (консервативно)",
            callback_data="temp_0.5"
        ),
        types.InlineKeyboardButton(
            text="1.0 (стандарт)",
            callback_data="temp_1.0"
        ),
        types.InlineKeyboardButton(
            text="1.5 (креативно)",
            callback_data="temp_1.5"
        )
    )
    builder.adjust(2)

    await callback_query.message.edit_text(
        text="Выберите температуру генерации:",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda c: c.data.startswith('temp_'))
async def process_temperature(callback_query: types.CallbackQuery):
    """Обработка выбора температуры"""
    user_id = callback_query.from_user.id
    temp = float(callback_query.data.split('_')[1])
    user_states[user_id].temperature = temp

    await callback_query.message.edit_text(
        text=f"Температура установлена: {temp}"
    )


@dp.callback_query(lambda c: c.data == 'set_author')
async def set_author(callback_query: types.CallbackQuery):
    """Выбор автора"""
    builder = InlineKeyboardBuilder()
    for i, author in enumerate(dataset.authors):
        builder.add(types.InlineKeyboardButton(
            text=author,
            callback_data=f"author_{i}"
        ))
    builder.adjust(2)

    await callback_query.message.edit_text(
        text="Выберите автора:",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda c: c.data.startswith('author_'))
async def process_author(callback_query: types.CallbackQuery):
    """Обработка выбора автора"""
    user_id = callback_query.from_user.id
    author_id = int(callback_query.data.split('_')[1])
    user_states[user_id].author_id = author_id

    await callback_query.message.edit_text(
        text=f"Автор изменен на: {dataset.authors[author_id]}"
    )


@dp.callback_query(lambda c: c.data == 'set_len')
async def set_length(callback_query: types.CallbackQuery):
    """Выбор длины текста"""
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(
            text="50 символов",
            callback_data="len_50"
        ),
        types.InlineKeyboardButton(
            text="100 символов",
            callback_data="len_100"
        ),
        types.InlineKeyboardButton(
            text="150 символов",
            callback_data="len_150"
        )
    )
    builder.adjust(2)

    await callback_query.message.edit_text(
        text="Выберите длину текста:",
        reply_markup=builder.as_markup()
    )


@dp.callback_query(lambda c: c.data.startswith('len_'))
async def process_length(callback_query: types.CallbackQuery):
    """Обработка выбора длины"""
    user_id = callback_query.from_user.id
    length = int(callback_query.data.split('_')[1])
    user_states[user_id].max_len = length

    await callback_query.message.edit_text(
        text=f"Длина текста установлена: {length}"
    )


@dp.message()
async def generate_poem(message: types.Message):
    """Генерация стихотворения"""
    user_id = message.from_user.id
    if user_id not in user_states:
        user_states[user_id] = UserState()

    state = user_states[user_id]

    try:
        msg = await message.answer("🖋️ Генерирую стихотворение...")


        input_text = message.text.lower().strip()
        if not input_text:
            await message.answer("Пожалуйста, введите начало стихотворения.")
            return

        try:
            generated_text = model.generate(
                start_text=input_text,
                author_id=state.author_id,
                max_len=state.max_len,
                temperature=state.temperature
            )
        except Exception as e:
            logger.error(f"Ошибка генерации: {str(e)}")
            await msg.edit_text("⚠️ Произошла ошибка при генерации стихотворения. Попробуйте еще раз.")
            return

        result = (
            f"🎭 Автор: {dataset.authors[state.author_id]}\n"
            f"🌡️ Температура: {state.temperature}\n"
            f"📏 Длина: {state.max_len}\n\n"
            f"📜 Результат:\n\n{generated_text}"
        )

        await msg.edit_text(result)

    except Exception as e:
        logger.error(f"Ошибка в обработке сообщения: {str(e)}")
        await message.answer("⚠️ Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")



async def on_startup():
    """Действия при запуске бота"""
    load_model()
    logger.info("Бот запущен")


if __name__ == '__main__':
    import asyncio
    from aiogram import Dispatcher


    async def main():
        await on_startup()
        await dp.start_polling(bot, skip_updates=True)


    asyncio.run(main())