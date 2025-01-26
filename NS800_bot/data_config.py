import json
from aiogram import Bot, Dispatcher, types
# Замените "YOUR_BOT_TOKEN" на токен, который вы получили от BotFather
API_TOKEN = '7643375571:AAGvGnelawtcrW3crMCHHZWVjl13LzQHCjs'
# Зададим имя базы данных
DB_NAME = 'quiz_bot.db'

# Открываем файл и загружаем его содержимое в переменную quiz_data
with open('quiz_data.json', 'r', encoding='utf-8') as file:
    quiz_data = json.load(file)