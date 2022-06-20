import random

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from numpy.core.numeric import True_

import logging

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


# takes the BOT_CONFIG from .json file
with open('BOT_CONFIG_22032022.json', 'r') as f:
    BOT_CONFIG = json.load(f)


"""Bot testing"""


X = []
y = []
for intent in BOT_CONFIG['intents']:
    for example in BOT_CONFIG['intents'][intent]['examples']:
        X.append(example)
        y.append(intent)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Creating a Vectorizer
vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(1, 3),
    binary=True,
    sublinear_tf=True
)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Creating a Classifier
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train)
clf.score(X_train_vectorized, y_train), clf.score(X_test_vectorized, y_test)

# TfidfVectorizer + LinearSVC bundle gives 0.8309368191721133 points on train
# and 0.30662020905923343 points on test


def get_intent_by_model(text):
    """
    Defines the intent by user's text, using
    prepared earlier Classifier and Vectorizer
    :param text: input text
    :return: str intent
    """
    vectorized_text = vectorizer.transform([text])
    return clf.predict(vectorized_text)[0]


def bot(text):
    """
    Gives a random answer from suitable responses to user's input
    :param text: input text
    :return: str response
    """
    intent = get_intent_by_model(text)

    return random.choice(BOT_CONFIG['intents'][intent]['responses'])


"""Bundle with Telegram bot"""


# Enabling logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)


def start(update: Update, context: CallbackContext) -> None:
    """
    Send a message when the command /start is issued.
    :param update: default - module Update
    :param context: CallbackContext
    """
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Привет, {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


def help_command(update: Update, context: CallbackContext) -> None:
    """
    Send a message when the command /help is issued.
    :param update: default - module Update
    :param context: CallbackContext
    """
    user = update.effective_user
    # Bot will ask to speak with him
    update.message.reply_markdown_v2(
        fr'{user.mention_markdown_v2()}\, поговори со мной\. Я знаю классные анекдоты'
    )


def answer(update: Update, context: CallbackContext) -> None:
    """Answer the user message."""
    update.message.reply_text(bot(clean(update.message.text)))


def main() -> None:
    """Start the bot."""
    # Creating the Updater and pass it bot's token.
    updater = Updater("1769968340:AAG3B_W0qdTSYh0MnUj2lf5ESthzWMO5rec")

    # Getting the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    # on non command i.e. message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, answer))

    # Starting the Bot
    updater.start_polling()

    updater.idle()


main()
