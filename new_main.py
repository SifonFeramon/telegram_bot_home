# -*- coding: utf-8 -*-
import logging
import os.path

import asyncio
import schedule
import cv2
import praw
import time
import aiohttp
import urllib3
import random
import datetime
import threading
import numpy as np
import argparse
import io

#from telegram.ext import Updater, CommandHandler, MessageHandler#, Filters
from telegram import Update
from telegram.ext import (
    Application,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import requests  # Модуль для обработки URL
from bs4 import BeautifulSoup  # Модуль для работы с HTML
import pytz

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]


async def get_camera_image(server_url: str):
    """
    Fetch image from camera server via HTTP.
    
    Args:
        server_url: Base URL of the camera server (e.g., http://192.168.1.100:8080)
    
    Returns:
        Image bytes if successful, None otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/capture", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    logger.error(f"Camera server returned status {resp.status}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to camera server: {e}")
        return None
    except asyncio.TimeoutError:
        logger.error("Camera server request timed out")
        return None


async def days_birthdays(context):
    try:
        birthday_al = time_for_event("birthday_al")
        birthday_dm = time_for_event("birthday_dm")
        birthday_ta = time_for_event("birthday_ta")
        birthday_da = time_for_event("birthday_da")
        good_years_dm = time_for_event("good_years_dm")
        good_years_da = time_for_event("good_years_da")
        happy_year = time_for_event("happy_year")
        summer_time = time_for_event("summer_time")
        await context.bot.send_message(chat_id=home_chat_id, text=birthday_al)
        time.sleep(1)      
        await context.bot.send_message(chat_id=home_chat_id, text=birthday_dm)
        time.sleep(1)
        await context.bot.send_message(chat_id=home_chat_id, text=birthday_ta)
        time.sleep(1)        
        await context.bot.send_message(chat_id=home_chat_id, text=birthday_da)
        time.sleep(3)        
        await context.bot.send_message(chat_id=home_chat_id, text=good_years_da)
        time.sleep(3)        
        await context.bot.send_message(chat_id=home_chat_id, text=good_years_dm)
        time.sleep(3)        
        await context.bot.send_message(chat_id=home_chat_id, text=happy_year)
        pic_relevant = await find_relevant_picture("christmas")
        await context.bot.send_photo(chat_id=home_chat_id, photo=pic_relevant, caption="Happy New Year")
        time.sleep(5)        
        await context.bot.send_message(chat_id=home_chat_id, text=summer_time)        
    except:
        print("Unexpected error in days_birthdays\n")    
    
async def good_night_message(context):
    try:
        await context.bot.send_message(chat_id=home_chat_id, text='Time go to sleep!')
        pic_relevant = await find_relevant_picture("ExposurePorn")
        await context.bot.send_photo(chat_id=home_chat_id, photo=pic_relevant, caption="Время выключать свет")
    except:
        print("Unexpected error in good_night_message\n")


def time_for_event(key_word):

    events = ["birthday_ta", "birthday_al", "birthday_da", "birthday_dm", "happy_year", "summer_time", 'good_years_da', 'good_years_dm']
    if key_word in events:
        current_date = datetime.datetime.now()
        dict_days = {"birthday_ta": datetime.datetime(current_date.year, 10, 14, 0, 0, 0),
                     "birthday_al": datetime.datetime(current_date.year, 3, 16, 0, 0, 0),
                     "birthday_da": datetime.datetime(current_date.year, 12, 6, 0, 0, 0),
                     "birthday_dm": datetime.datetime(current_date.year, 8, 29, 0, 0, 0),
                     "happy_year": datetime.datetime(current_date.year, 1, 1, 0, 0, 0),
                     "summer_time": datetime.datetime(current_date.year, 6, 1, 0, 0, 0),
                     "good_years_da": datetime.datetime(2025, 12, 6, 0, 0, 0),
                     "good_years_dm": datetime.datetime(2027, 8, 29, 0, 0, 0)}
                     
        days_left = (dict_days[key_word] - current_date).days
        if days_left < 0:
            dict_days[key_word] = dict_days[key_word].replace(year=current_date.year+1)
            days_left = (dict_days[key_word] - current_date).days

        dict_text = {"birthday_ta": "До дня рождения Татьяны осталось ",
                     "birthday_al": "До дня рождения Алексея осталось ",
                     "birthday_da": "До дня рождения Дарьи осталось ",
                     "birthday_dm": "До дня рождения Дмитрия осталось ",
                     "happy_year": "До Нового года осталось ",
                     "summer_time": "До лета осталось ",
                     "good_years_da": "До совершеннолетия Дарьи осталось ",
                     "good_years_dm": "До совершеннолетия Дмитрия осталось "}

        return dict_text[key_word] + str(days_left) + "дней"


async def good_morning_message(context):

    try:
        await context.bot.send_message(chat_id=home_chat_id, text='Time give up!')
        pic_relevant = await find_relevant_picture("sunrise")
        await context.bot.send_photo(chat_id=home_chat_id, photo=pic_relevant, caption="Пора вставать!")
    except:
        print("Unexpected error in good_morning_message\n")


async def check_people(context):

    logger.info("Start check people in frame")
    
    # Fetch image from camera server
    image_bytes = await get_camera_image(camera_server_url)
    if image_bytes is None:
        logger.warning("Failed to get image from camera server")
        return
    
    # Decode image bytes to OpenCV frame
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        logger.warning("Failed to decode image from camera server")
        return
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == LABELS.index("person") and confidence > 0.4:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
    
    # ensure at least one detection exists
    results = []
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)

        if len(results) > 0:
            for i_res in results:
                cv2.rectangle(frame, (i_res[1][0], i_res[1][1]), (i_res[1][2], i_res[1][3]), (0, 255, 0), 2)
                cv2.imwrite("check_people.png", frame)
                await context.bot.send_message(chat_id=home_chat_id, text="Найден человек\n")
                await context.bot.send_photo(chat_id=home_chat_id, photo=open("check_people.png", 'rb'), caption="Это человек!")

    logger.info("End check people in frame")


def google_request(update, i_text):

    text_requests = {'morning': 'https://www.google.com/search?q=%D0%B2%D0%BE%D1%81%D1%85%D0%BE%D0%B4+%D1%81%D0%BE%D0%BB%D0%BD%D1%86%D0%B0+%D0%B2+%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%B2%D0%B5-%D0%BD%D0%B0-%D0%B4%D0%BE%D0%BD%D1%83&oq=%D0%B2%D0%BE%D1%81%D1%85%D0%BE%D0%B4+%D0%B2+%D0%A0%D0%BE%D1%81%D1%82%D0%BE%D0%B2%D0%B5&aqs=chrome.2.69i57j0i22i30l2.5436j0j7&sourceid=chrome&ie=UTF-8',
                     'night': 'https://www.google.com/search?q=%D0%B7%D0%B0%D1%85%D0%BE%D0%B4+%D1%81%D0%BE%D0%BB%D0%BD%D1%86%D0%B0+%D0%B2+%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%B2%D0%B5-%D0%BD%D0%B0-%D0%B4%D0%BE%D0%BD%D1%83&ei=B6aPY8zRJYLRrgTPk7OYBg&ved=0ahUKEwiM8aa56uX7AhWCqIsKHc_JDGMQ4dUDCA8&uact=5&oq=%D0%B7%D0%B0%D1%85%D0%BE%D0%B4+%D1%81%D0%BE%D0%BB%D0%BD%D1%86%D0%B0+%D0%B2+%D1%80%D0%BE%D1%81%D1%82%D0%BE%D0%B2%D0%B5-%D0%BD%D0%B0-%D0%B4%D0%BE%D0%BD%D1%83&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIKCAAQgAQQRhD-AToKCAAQRxDWBBCwAzoGCAAQBxAeOgcIABCABBANOgwIABCABBANEEYQ_gFKBAhBGABKBAhGGABQughY5xFg9BVoAnABeACAAU6IAbMDkgEBNpgBAKABAcgBCMABAQ&sclient=gws-wiz-serp'}
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}

    texts_mesages = {'morning': "Sunrise!", 'night': "Sunset!"}

    if i_text in text_requests.keys():
        full_page = requests.get(text_requests[i_text], headers=headers)
        soup = BeautifulSoup(full_page.content, 'html.parser')
        convert = soup.findAll("div", {"class": "MUxGbd"})
        now_date = datetime.datetime.now()
        morning_time = now_date.replace(hour=int(convert[0].text.split(':')[0]), minute=int(convert[0].text.split(':')[-1]))
        time_timer = (morning_time - now_date).total_seconds()
        print(f"Seconds waiting {time_timer} for {i_text}\n")
        timer = threading.Timer(time_timer, lambda: update.bot.send_message(chat_id=home_chat_id, text=texts_mesages[i_text]))
        timer.start()


def run_timer(context):
    schedule.run_pending()

async def find_relevant_picture(i_tag):
    random.seed(datetime.datetime.now())
    pic = None
    if i_tag in ["dog", "cat", "bears", "animal", "dance", "ExposurePorn", "sunrise", "CozyPlaces", "EarthPorn", "corgi", "christmas"]:
        top_post = reddit.subreddit(i_tag).top(time_filter="week", limit=1000)
        pictures = []
        try:
            for idx, submission in enumerate(top_post):
                posturl = submission.url
                if os.path.splitext(posturl)[-1] == '.jpg' or os.path.splitext(posturl)[-1] == '.png' or \
                        os.path.splitext(posturl)[-1] == '.gif':
                    #print(posturl)
                    pictures.append(posturl)
                    if len(pictures) > 100:
                        break
            if len(pictures) != 0:
                random.shuffle(pictures)
                pic = pictures[random.randint(0, len(pictures))]
        except:
            print("Unexpected error in find_relevant_picture\n")

    return pic

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
async def start(update, context):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hi!') 


async def help(update, context):
    """Send a message when the command /help is issued."""
    print("Help answer\n")
    await update.message.reply_text(" list of search pictures  " + "  ".join(["dog", "cat", "bears", "animal", "dance"]) + " command pic__{your subject}") 


async def echo(update, context):
    """Echo the user message."""
    user = update.message.from_user
    print(f"{context=}\n")
    if update.message.text == "test_home":
        await update.effective_message.reply_text("My home at Rostov!") 
        print(f"{user['username']=} and his user ID:{user['id']}\n")
    elif update.message.text == "view_home":
        if user['id'] not in [209255151, 978949705]:
            await update.effective_message.reply_text("Похоже вы не живёте в этом доме!\n") 
        else:
            # Fetch image from camera server
            image_bytes = await get_camera_image(camera_server_url)
            if image_bytes is not None:
                await update.effective_message.reply_photo(io.BytesIO(image_bytes), caption="Комната")
            else:
                await update.effective_message.reply_text("Не удалось получить изображение с камеры") 
    elif "pic__" in update.message.text:
        print(update.message.text, update.message.text.split('pic__')[-1])
        relevant_pic = await find_relevant_picture(update.message.text.split('pic__')[-1])
        await update.effective_message.reply_photo(relevant_pic, caption="Воть")
    else:
        ...


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def new_main(args):
    global is_alive
    
    application = Application.builder().token(args.token).build()
    job_queue =  application.job_queue

    job_start_day = job_queue.run_daily(good_morning_message, datetime.time(hour=9,minute=0, tzinfo=pytz.timezone("Europe/Moscow")), days=(0, 1, 2, 3, 4, 5, 6))
    job_birthdays = job_queue.run_daily(days_birthdays, datetime.time(hour=23,minute=40, tzinfo=pytz.timezone("Europe/Moscow")), days=(0, 1, 2, 3, 4, 5, 6))
    job_end_day = job_queue.run_daily(good_night_message, datetime.time(hour=23,minute=59, tzinfo=pytz.timezone("Europe/Moscow")), days=(0, 1, 2, 3, 4, 5, 6))
    job_check_people = job_queue.run_repeating(check_people, interval=180, first=10) 
    application.add_handler(CommandHandler(["start", "help"], start))
    
    application.add_handler(MessageHandler(filters.TEXT, echo))
    
    application.run_polling()

if __name__ == '__main__':
    is_alive = True    
    
    parser = argparse.ArgumentParser(description="Home telegram bot")
    parser.add_argument("-token", dest="token", type=str, required=True)
    parser.add_argument("-client_id", dest="client_id", type=str, required=True)    
    parser.add_argument("-client_secret", dest="client_secret", type=str, required=True)    
    parser.add_argument("-camera_server_url", dest="camera_server_url", type=str, required=True,
                        help="URL of the camera server (e.g., http://192.168.1.100:8080)")    
    parser.add_argument("-home_chat_id", dest="home_chat_id", type=str, required=True)    
    
    args = parser.parse_args()
    
    reddit = praw.Reddit(user_agent="Get top wallpaper from /r/{subreddit} by /u/ssimunic".format(subreddit="cats"),
                     client_id=args.client_id, client_secret=args.client_secret)

    reddit.read_only = True

    camera_server_url = args.camera_server_url
    home_chat_id = args.home_chat_id
    
    print(args)
    
    new_main(args)
