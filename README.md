# About 
A little example using RAG in a [Poe.com](Poe.com) bot. 

What's a poe bot you ask? [Poe.com](Poe.com) is a service that lets you converse with 
a wide variety of different LLMs (GPT-3, GPT-4, Llama, Mistral, Claude, etc). A poe bot acts as middleware and receives 
the chat messages a user has addressed to the bot on poe.com and then forwards these messages, after whatever logic is implemented,
to an LLM on poe.com. The bot then gets the reply from the LLM, does more logic, and sends it back to the user.
The beauty is that Poe.com hosts the LLMs and does inference for you - and they do it for free... for now. 

For the dataset I've chosen a set of 3824 news articles from 2016-2017 (See: https://doi.org/10.7910/DVN/GMFCTR)

## Local Setup
- create virtualenv as necessary and cd to project dir
- unzip db.zip
- pip install -r requirements.txt
- python NewsArticles.py
- start the reverse proxy [ngrok](https://ngrok.com/) in separate terminal to allow remote access to your local machine

## Poe Setup 
- create an account on poe.com
- create a bot via [poe.com/create_bot](poe.com/create_bot)
- make sure to select "user server" and note the access key.
- enter in the server url given by ngrok
- update bot settings the first time, and every time the settings change (such as when a new LLM is chosen for inference)

Bot settings are updated via a post as such 
`curl -X POST https://api.poe.com/bot/fetch_settings/<botname>/<access_key>`
See more about that [here](https://developer.poe.com/server-bots/updating-bot-settings)

## Conversing with the bot
Talk with the bot on poe.com. You should see info on your machine's terminal, specifically the prompt that is being sent to poe.com
after appropriate articles have been found in the vector database.

## Extra 
If you want to get a handle on how to create your own bot, take a look at `SimpleBot.py` as it will show you how to 
receive and send messages from Poe.com

## Example action on Poe.com
![capture](https://github.com/fjohnson/zotpoe/blob/master/example_output.PNG)
