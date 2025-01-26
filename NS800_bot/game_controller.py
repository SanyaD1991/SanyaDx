from aiogram import types
from aiogram.utils.keyboard import InlineKeyboardBuilder
from NS800_bot.database_controller import get_quiz_index, update_quiz_index
from NS800_bot.data_config import quiz_data
correct_answer = 0


def generate_options_keyboard(answer_options, right_answer):
    builder = InlineKeyboardBuilder()

    for option in answer_options:
        data = f"right:{option}" if option == right_answer else f"wrong:{option}"
        builder.add(types.InlineKeyboardButton(
            text=option,
            callback_data = data) #"right_answer" if option == right_answer else "wrong_answer")
        )
    builder.adjust(1)
    return builder.as_markup()

async def get_question(message, user_id):

    # Получение текущего вопроса из словаря состояний пользователя
    current_question_index = await get_quiz_index(user_id)
    correct_index = quiz_data[current_question_index]['correct_option']
    opts = quiz_data[current_question_index]['options']
    kb = generate_options_keyboard(opts, opts[correct_index])
    await message.answer(f"{quiz_data[current_question_index]['question']}", reply_markup=kb)


async def new_quiz(message):
    user_id = message.from_user.id
    current_question_index = 0
    global correct_answer
    correct_answer = 0
    await update_quiz_index(user_id, current_question_index)
    await get_question(message, user_id)

async def question_update(callback: types.CallbackQuery, is_right: bool):

    user_id = callback.from_user.id
    # Получаем текущий индекс вопроса из базы данных
    current_question_index = await get_quiz_index(user_id)

    # Обновление номера текущего вопроса в базе данных
    current_question_index += 1
    await update_quiz_index(user_id, current_question_index)

    # Если параметр is_right равен True, увеличиваем счетчик
    global correct_answer
    if is_right:
        correct_answer += 1

    # Проверяем, есть ли еще вопросы
    if current_question_index < len(quiz_data):
        await get_question(callback.message, user_id)
    else:
        await callback.message.answer("Это был последний вопрос. Квиз завершен!")
        await callback.message.answer(f"Вы дали {correct_answer} правильных ответов из {current_question_index} вопросов.")