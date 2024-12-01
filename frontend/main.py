import asyncio
import logging
import sys
from dotenv import load_dotenv

from frontend.tg_bot import TelegramBot


# Asynchronous function to start the Telegram bot
async def start_bot():
    bot = TelegramBot() 
    await bot.dp.start_polling(bot.bot)  # Starting the bot's polling to receive updates


# Main asynchronous function to set up and run the bot
async def main():
    load_dotenv()  # Load environment variables from a .env file
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)  # Configure logging to output to stdout
    await start_bot()  


if __name__ == "__main__":
    asyncio.run(main())