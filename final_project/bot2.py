import logging
import os
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.enums import ParseMode
from main import PoetTransformer, PoetDataset
from dotenv import load_dotenv

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
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_4.pt"
JSON_PATH = "classic_poems.json"

bot = Bot(
    token=TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML))
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

model = None
vocab = None
authors = None
author_to_id = None
word2idx = None
idx2word = None


class UserState:
    def __init__(self):
        self.temperature = 1.0
        self.max_len = 30
        self.author_id = 0


user_states = {}


def load_model():
    """Загрузка модели и словарей"""
    global model, vocab, authors, author_to_id, word2idx, idx2word

    try:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found: {CHECKPOINT_PATH}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Загружаем словари из чекпоинта
        word2idx = checkpoint['word2idx']
        idx2word = checkpoint['idx2word']
        vocab_size = checkpoint['vocab_size']
        num_authors = checkpoint['num_authors']

        # Создаем dataset для получения информации об авторах
        dataset = PoetDataset(JSON_PATH, max_length=32)
        authors = dataset.authors
        author_to_id = dataset.author_to_id

        # Модель
        model = PoetTransformer(
            vocab_size=vocab_size,
            num_authors=num_authors,
            d_model=256,
            nhead=8,
            num_layers=4
        ).to(device)

        # Загрузка весов
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        logger.info("Модель и словари успешно загружены")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}", exc_info=True)
        return False


def generate_text(start_text, author_id=None, max_len=100, temperature=1.0):
    """Функция для генерации текста с использованием модели"""
    try:
        logger.info(f"Начало генерации текста. Параметры: "
                    f"start_text='{start_text[:30]}...', "
                    f"author_id={author_id}, "
                    f"max_len={max_len}, "
                    f"temperature={temperature}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.debug(f"Используемое устройство для генерации: {device}")

        # Токенизация входного текста
        tokens = word_tokenize(start_text.lower())
        logger.debug(f"Токенизированный входной текст: {tokens}")

        input_ids = [word2idx.get(token, word2idx['<SOS>']) for token in tokens if token in word2idx]
        logger.debug(f"Конвертированные input_ids: {input_ids}")

        if not input_ids:
            input_ids = [word2idx['<SOS>']]
            logger.debug("Пустой input_ids, используется токен <SOS>")

        if len(input_ids) > 50:
            original_len = len(input_ids)
            input_ids = input_ids[-50:]
            logger.debug(f"Обрезаны input_ids с {original_len} до 50 токенов")

        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        logger.debug(f"Тензор input_ids: {input_ids.shape}")

        if author_id is not None:
            author_tensor = torch.tensor([author_id], dtype=torch.long, device=device)
            logger.debug(f"Авторский тензор: {author_tensor}")
        else:
            author_tensor = None
            logger.debug("Автор не указан, используется None")

        output_ids = []
        logger.info(f"Начало генерации последовательности (max_len={max_len})")

        for step in range(max_len):
            with torch.no_grad():
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
                mask = mask.to(device)

                if step % 10 == 0 or step == max_len - 1:
                    logger.debug(f"Шаг {step}: seq_len={seq_len}, "
                                 f"input_ids shape={input_ids.shape}")

                logits = model(input_ids, author_tensor, mask=mask)[0, -1, :] / temperature

                if step % 10 == 0:
                    top5 = torch.topk(logits, 5)
                    top5_words = [(idx2word[idx.item()], prob.item())
                                  for idx, prob in zip(top5.indices, top5.values)]
                    logger.debug(f"Топ-5 вероятных слов на шаге {step}: {top5_words}")

                probs = torch.softmax(logits, dim=-1)
                next_word = torch.multinomial(probs, num_samples=1)
                output_ids.append(next_word.item())

                if next_word.item() == word2idx['<EOS>']:
                    logger.info(f"Обнаружен токен <EOS> на шаге {step}, завершение генерации")
                    break

                next_word = next_word.unsqueeze(0)
                input_ids = torch.cat([input_ids, next_word], dim=1)

        logger.debug(f"Сгенерированные output_ids: {output_ids}")

        words = []
        for idx in output_ids:
            if idx in idx2word:
                word = idx2word[idx]
                if word not in ['<SOS>', '<EOS>', '<PAD>']:
                    words.append(word)

        # Форматирование
        generated_text = ' '.join(words)
        logger.info(f"Успешная генерация текста. Длина: {len(words)} слов. "
                    f"Результат: '{generated_text[:50]}...'")

        return generated_text.capitalize()

    except Exception as e:
        logger.error(f"Критическая ошибка генерации текста: {str(e)}", exc_info=True)
        logger.error(f"Параметры на момент ошибки: "
                     f"start_text='{start_text[:30]}...', "
                     f"author_id={author_id}, "
                     f"max_len={max_len}, "
                     f"temperature={temperature}")
        raise RuntimeError("Произошла ошибка при генерации текста. Попробуйте еще раз.")


@dp.message(Command('start'))
async def cmd_start(message: types.Message):
    """Обработка команды /start"""
    user_id = message.from_user.id
    user_states[user_id] = UserState()

    if model is None:
        await message.answer("⚠️ Модель еще не загружена. Пожалуйста, подождите...")
        if not load_model():
            await message.answer("❌ Не удалось загрузить модель. Попробуйте позже.")
            return

    state = user_states[user_id]

    text = (
        "👋 Добро пожаловать в генератор стихов!\n\n"
        "Отправьте мне тему стихотворения, и я напишу его в стиле выбранного автора.\n\n"
        "⚙️ Текущие настройки:\n"
        f"• Температура: {state.temperature}\n"
        f"• Длина текста: {state.max_len} токенов\n"
        f"• Автор: {authors[state.author_id]}\n\n"
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
            text=f"👨‍🎨 Автор",
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
    for i, author in enumerate(authors):
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
        text=f"Автор изменен"
    )


@dp.callback_query(lambda c: c.data == 'set_len')
async def set_length(callback_query: types.CallbackQuery):
    """Выбор длины текста"""
    builder = InlineKeyboardBuilder()
    builder.add(
        types.InlineKeyboardButton(
            text="10 символов",
            callback_data="len_10"
        ),
        types.InlineKeyboardButton(
            text="30 символов",
            callback_data="len_30"
        ),
        types.InlineKeyboardButton(
            text="50 символов",
            callback_data="len_50"
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

    if model is None or word2idx is None or idx2word is None:
        await message.answer("⚠️ Модель или словари еще не загружены. Пожалуйста, подождите...")
        if not load_model():
            await message.answer("❌ Не удалось загрузить модель. Попробуйте позже.")
            return

    try:
        msg = await message.answer("🖋️ Генерирую стихотворение...")
        input_text = message.text.strip()

        if not input_text:
            await message.answer("Пожалуйста, введите начало стихотворения.")
            return

        try:
            generated_text = generate_text(
                start_text=input_text,
                author_id=state.author_id,
                max_len=state.max_len,
                temperature=state.temperature
            )

            generated_text = generated_text.replace('<SOS>', '').replace('<EOS>', '').strip()
            if not generated_text:
                raise ValueError("Пустой результат генерации")

        except Exception as e:
            logger.error(f"Ошибка генерации: {str(e)}", exc_info=True)
            await msg.edit_text(
                "⚠️ Произошла ошибка при генерации. Попробуйте изменить параметры или начать с другой фразы.")
            return

        result = (
            f"🎭 Автор: {authors[state.author_id]}\n"
            f"🌡️ Температура: {state.temperature}\n"
            f"📏 Длина: {state.max_len} токенов\n\n"
            f"📜 Результат:\n\n{generated_text}\n\n"
            "✏️ Попробуйте изменить параметры через /settings"
        )

        await msg.edit_text(result)

    except Exception as e:
        logger.error(f"Ошибка в обработке сообщения: {str(e)}", exc_info=True)
        await message.answer("⚠️ Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")


async def on_startup():
    """Действия при запуске бота"""
    try:
        if not load_model():
            logger.error("Не удалось загрузить модель при старте")
        logger.info("Бот запущен")
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    import asyncio
    from nltk.tokenize import word_tokenize
    from nltk import download

    download('punkt')


    async def main():
        await on_startup()
        await dp.start_polling(bot, skip_updates=True)


    asyncio.run(main())
