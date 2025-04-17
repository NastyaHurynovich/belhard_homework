import chainlit as cl
from main import PoetTransformer

# Загружаем модель при старте приложения
poetry_model = PoetTransformer('poet_transformer.pth')


@cl.on_chat_start
async def start_chat():
    # Создаем элементы интерфейса
    settings = await setup_interface()
    cl.user_session.set("settings", settings)


async def setup_interface():
    # Выбор автора
    authors = list(poetry_model.author_to_id.keys())
    author = await cl.AskActionMessage(
        content="Выберите стиль автора:",
        actions=[cl.Action(name=a, value=a) for a in authors]
    ).send()

    # Настройки генерации
    settings = {
        "author": author.get("value") if author else None,
        "temperature": 0.7,
        "length": 100
    }

    # Слайдер для температуры
    temp = await cl.AskSliderMessage(
        content="Выберите креативность (температуру):",
        min=0.1,
        max=1.5,
        step=0.1,
        initial=0.7
    ).send()
    settings["temperature"] = temp.get("value")

    return settings


@cl.on_message
async def generate_poem(message: cl.Message):
    settings = cl.user_session.get("settings")

    # Получаем сообщение пользователя
    prompt = message.content

    # Генерация стиха
    poem = poetry_model.generate(
        prompt=prompt,
        author=settings["author"],
        max_len=settings["length"],
        temperature=settings["temperature"]
    )

    # Форматирование результата
    formatted_poem = format_poem(poem)

    # Отправка результата
    await cl.Message(
        content=f"**Сгенерированный стих ({settings['author'] or 'без стиля'}):**\n\n{formatted_poem}",
    ).send()


def format_poem(text):
    # Простое форматирование - добавление переносов строк
    return text.replace(". ", ".\n").replace(", ", ",\n")


if __name__ == "__main__":
    cl.run()