import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram import F
from NS800_bot.database_controller import get_quiz_index, update_quiz_index, create_table
from NS800_bot.data_config import API_TOKEN, quiz_data, DB_NAME
from NS800_bot.game_controller import get_question, new_quiz, question_update

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=API_TOKEN)
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /верный ответ
@dp.callback_query(lambda c: c.data.startswith("right:"))
async def right_answer(callback: types.CallbackQuery):
    user_answer = callback.data.split(":", 1)[1]
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )
    await callback.message.answer(user_answer)
    await question_update(callback, True)

# Хэндлер на команду /не верный ответ
@dp.callback_query(lambda c: c.data.startswith("wrong:"))
async def wrong_answer(callback: types.CallbackQuery):
    user_answer = callback.data.split(":", 1)[1]
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )
    await callback.message.answer(user_answer)
    await question_update(callback, False)


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать игру"))
    await message.answer("Добро пожаловать в квиз!", reply_markup=builder.as_markup(resize_keyboard=True))


# Хэндлер на команду /quiz
@dp.message(F.text=="Начать игру")
@dp.message(Command("quiz"))
async def cmd_quiz(message: types.Message):

    await message.answer(f"Давайте начнем квиз!")
    await new_quiz(message)

# Запуск процесса поллинга новых апдейтов
async def main():

    # Запускаем создание таблицы базы данных
    await create_table()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())





