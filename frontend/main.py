import asyncio
import logging
import sys
from dotenv import load_dotenv

from frontend.tg_bot import TelegramBot


async def start_bot():
    bot = TelegramBot()
    await bot.dp.start_polling(bot.bot)


async def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    await start_bot()


if __name__ == "__main__":
    asyncio.run(main())