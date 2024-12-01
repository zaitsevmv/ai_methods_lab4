import logging
import os
import re

from aiogram import Bot, Dispatcher, html, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, KeyboardButton
from aiogram.utils.keyboard import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from model.gpt_model import generate_gpt
from model.llama_model import generate_llama


# Dictionary to track user state
user_state = {}

class TelegramBot:
    def __init__(self):
        token = os.getenv("TG_BOT_TOKEN")
        self.bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.dp = Dispatcher()

        # Register handlers
        self.register_handlers()

    def register_handlers(self):
        @self.dp.message(CommandStart())
        async def command_start_handler(message: Message) -> None:
            """
            This handler receives messages with `/start` command
                        
            Args:
                message: message recieved.
                
            Returns:
                None
            """
            user_state[message.from_user.id] = {"step": "model"}
            await ask_model(message)

        async def ask_model(message: Message) -> None:
            """
            Ask for model used for generation. Creates inline buttons. Buttons are handled in callback_query_handler
            
            Args:
                message: message recieved.
                
            Returns:
                None
            """
            buttons = [
                [
                    InlineKeyboardButton(text="ruGPT", callback_data="model_rugpt"),
                    InlineKeyboardButton(text="LLAMA", callback_data="model_llama"),
                ]
            ]
            keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
            await message.answer("Выберете модель ИИ:", reply_markup=keyboard)

        async def ask_type(callback: CallbackQuery) -> None:
            """
            Ask for type used of an anecdote. Writes it in user_state.
            
            Args:
                callback: callback recieved.
                
            Returns:
                None
            """
            user_id = callback.from_user.id
            user_state[user_id]["step"] = "type"

            keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="Короткий"), KeyboardButton(text="Смешной"), KeyboardButton(text="Грустный")],
                ],
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            await callback.message.answer("Выберете тип:", reply_markup=keyboard)

        async def ask_character(message: Message) -> None:
            """
            Ask for character used in an anecdote. Writes it in user_state.
            
            Args:
                message: message recieved.
                
            Returns:
                None
            """
            user_id = message.from_user.id
            user_state[user_id]["step"] = "character"

            keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="Вовочка"), KeyboardButton(text="Штирлиц"), KeyboardButton(text="Петька и Василий Иванович")],
                ],
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            await message.answer("Выберете главного героя:", reply_markup=keyboard)
            
        async def ask_location(message: Message) -> None:
            """
            Ask for location used in an anecdote. Writes it in user_state.
            
            Args:
                message: message recieved.
                
            Returns:
                None
            
            """
            user_id = message.from_user.id
            user_state[user_id]["step"] = "location"

            keyboard = ReplyKeyboardMarkup(
                keyboard=[
                    [KeyboardButton(text="Лес"), KeyboardButton(text="Дом"), KeyboardButton(text="Бар")],
                ],
                resize_keyboard=True,
                one_time_keyboard=True,
            )
            await message.answer("Выберете место:", reply_markup=keyboard)

        @self.dp.callback_query()
        async def callback_query_handler(callback: CallbackQuery) -> None:
            """
            Handles the model selection callback
            
            Args:
                callback: button callback.

            Returns:
                None
            """
            user_id = callback.from_user.id
            if callback.data.startswith("model_"):
                # Save the chosen model and move to the next step
                user_state[callback.from_user.id] = {"step": "model"}
                chosen_model = callback.data.split("_")[1]
                user_state[user_id]["model"] = chosen_model
                await callback.message.answer(f"Вы выбрали: {chosen_model}!")
                await ask_type(callback)

            # Acknowledge the callback
            await callback.answer()
        
        def input_valid(choice: str) -> bool:
            data = choice.replace(" ", "")
            return re.fullmatch(r"[А-Яа-я0-9]+", data)

        @self.dp.message()
        async def message_handler(message: Message) -> None:
            """
            Handles responses for all selections
            
            Args:
                message: message recieved.

            Returns:
                None
            """
            user_id = message.from_user.id

            # If the user state is not initialized, reset
            if user_id not in user_state or "step" not in user_state[user_id]:
                await command_start_handler(message)
                return

            step = user_state[user_id]["step"]

            if step == "type":
                # Save the type and move to the next step
                user_response = message.text.strip()
                if not input_valid(user_response):
                    user_response = ' '
                user_state[user_id]["type"] = user_response
                if user_response != ' ':
                    await message.answer(f"Вы выбрали: {user_response}!")
                await ask_character(message)
                
            elif step == "character":
                # Save the character and move to the next step
                user_response = message.text.strip()
                if not input_valid(user_response):
                    user_response = ' '
                user_state[user_id]["character"] = user_response
                if user_response != ' ':
                    await message.answer(f"Вы выбрали: {user_response}!")
                await ask_location(message)
                
            elif step == "location":
                # Save the location and conclude
                user_response = message.text.strip()
                if not input_valid(user_response):
                    user_response = ' '
                user_state[user_id]["location"] = user_response
                if user_response != ' ':
                    await message.answer(f"Вы выбрали: {message.text}!")
                    
                await message.answer("Генерация анекдота...")

                anecdote = await generate_anecdote(user_state[user_id])
                await message.answer(f"{anecdote}")
                # Reset the user state
                user_state[user_id] = {"step": "model"}
                await ask_model(message)
            else:
                await message.answer("Возникла ошибка, попробуйте позже")
                
        async def generate_anecdote(current_user_state: dict) -> str:
            """
            Generates an anecdote using the data collected from the user.

            Args:
                current_user_state: dict that contains collected data.

            Returns:
                The generated anecdote string.
            """

            # Data for prompt
            type_ = current_user_state["type"]
            character = current_user_state["character"]
            location = current_user_state["location"]

            # Generate results using selected model
            if current_user_state["model"] == 'rugpt':
                prompt = (
                    f"{type_} анекдот, "
                    f"с персонажем {character}. Действия анекдота происходят в {location}. "
                )
                return await generate_gpt(prompt)
            if current_user_state["model"] == 'llama':
                prompt = (
                    f"Придумай короткий {type_} анекдот, "
                    f"с персонажем {character}. Действия анекдота происходят в {location}. "
                    "Сделай его смешным и увлекательным и коротким."
                )
                return await generate_llama(prompt)
            return 'Модель не найдена'