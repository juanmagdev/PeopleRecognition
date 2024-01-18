from telegram import Bot, InputFile
import base64
from io import BytesIO
from PIL import Image

async def send_telegram_message(bot, chat_id, text, image_base64):
    try:
        # Decode base64 image data
        image_binary = base64.b64decode(image_base64)

        # Convert binary image data to a BytesIO stream
        image_stream = BytesIO(image_binary)

        # Open the image using PIL (Python Imaging Library)
        image = Image.open(image_stream)

        # Create a new BytesIO stream to save the image as JPEG
        jpeg_stream = BytesIO()
        image.save(jpeg_stream, format='JPEG')

        # Send the JPEG image as a document with the specified caption
        await bot.send_document(chat_id=chat_id, document=InputFile(jpeg_stream.getvalue(), filename='image.jpg'), caption=text)

    except Exception as e:
        print(f"Error sending message: {e}")
