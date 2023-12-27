# About 
A little example using RAG in a poe.com bot. 

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
  
