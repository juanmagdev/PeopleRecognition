import asyncio
from telegram import Bot

telegram_bot_token = '6934123684:AAEWK2BUPPVIphHmRvodqashjcjpdRPm2uE'
bot = Bot(token=telegram_bot_token)
chat_id = '1027514141'
message_text = 'Cuidao que te roban'

async def send_telegram_message():
    await bot.send_message(chat_id=chat_id, text=message_text)

async def main():
    await send_telegram_message()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
