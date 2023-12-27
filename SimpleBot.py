"""

Sample bot that echoes back messages.

This is the simplest possible bot and a great place to start if you want to build your own bot.

https://pypi.org/project/fastapi-poe/ <--- See more here for other examples

Also see: https://developer.poe.com/server-bots/poe-protocol-specification
"""
from __future__ import annotations

from pprint import pprint
from typing import AsyncIterable

import fastapi_poe as fp
from fastapi_poe import ProtocolMessage, QueryRequest


async def get_responses():
    """
    This function allows you to message and receive a reply from a Poe bot without
    going through the web interface. It requires a developer API key and its usage counts towards
    your allotted quota.
    """
    api_key = ''
    messages = [fp.ProtocolMessage(role="user", content="Hello world")]
    responses = []

    async for partial in fp.get_bot_response(messages=messages, bot_name="Llama-2-70b", api_key=api_key):
        responses.append(partial.text)
        full_prompt = partial.full_prompt

    print(''.join(responses))
    print(full_prompt)


class SimpleBot(fp.PoeBot):
    def __init__(self):
        # The selected LLM we want to use for inference.
        self.com_bot = "GPT-4"

    def create_request(self, org_request, content):
        msg = ProtocolMessage(role="user", content=content)
        request = QueryRequest(
            query=[msg],
            version=org_request.version,
            type="query",
            user_id=org_request.user_id,
            conversation_id="", # Ok to be blank, not sure what this is for.
            message_id="", # Ok to be blank, not sure what this is for.
        )

        # Debug
        pprint({'version': org_request.version,
                'user_id': org_request.user_id,
                'conversation_id': org_request.conversation_id,
                'message_id': org_request.message_id})
        return request

    async def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        print(f"Incoming message with {len(request.query)} Messages")
        last_message = request.query[-1].content
        prompt = (f"Answer the following question in your best pirate voice. "
                  f"Be as humorous as possible, but keep it short."
                  f" Question:{last_message}")

        new_request = self.create_request(request, prompt)

        # This function allows us to call self.com_bot for inference for free.
        # It does not count towards your usage quota on poe.com.
        async_gen = fp.stream_request(new_request, self.com_bot, '<missing>')
        pprint(request.access_key)
        async for msg in async_gen:
            yield msg

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        # You _need_ to do `curl -X POST https://api.poe.com/bot/fetch_settings/{bot_name}/{access_key}`
        # every time you change the bot type used for inference (i.e. when self.com_bot changes)
        # See: https://developer.poe.com/server-bots/updating-bot-settings
        return fp.SettingsResponse(server_bot_dependencies={self.com_bot: 1})

def fastapi_app():
    bot = SimpleBot()
    fp.run(bot, allow_without_key=True)


if __name__ == '__main__':
    fastapi_app()

