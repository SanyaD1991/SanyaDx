import json
import aiosqlite
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram import F

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Замените "YOUR_BOT_TOKEN" на токен, который вы получили от BotFather
API_TOKEN = '7643375571:AAGvGnelawtcrW3crMCHHZWVjl13LzQHCjs'
# Объект бота
bot = Bot(token=API_TOKEN)
# Диспетчер
dp = Dispatcher()
# URL вашей HTML-страницы
#WEB_APP_URL = 'https://3945-185-233-80-143.ngrok-free.app/quiz.html'
WEB_APP_URL = 'https://www.google.com/'

def create_url_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton("Открыть сайт", url=WEB_APP_URL))
    return keyboard


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    keyboard = create_url_keyboard()
    await message.answer("Добро пожаловать! Нажмите кнопку ниже, чтобы открыть сайт.", reply_markup=keyboard)


# Запуск процесса поллинга новых апдейтов
async def main():

    # Запускаем создание таблицы базы данных
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())