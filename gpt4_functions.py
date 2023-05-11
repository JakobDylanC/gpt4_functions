import os
import openai
import tiktoken
import backoff
import asyncio

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(f"Required environment variable OPENAI_API_KEY is not set.")
openai.api_key = os.environ["OPENAI_API_KEY"]
GPT4_TOKENS_PER_MESSAGE = 3
GPT4_TOKENS_PER_NAME = 1
GPT4_TOKENS_PER_MESSAGES = 3
GPT4_MAX_TOTAL_TOKENS = 8192
MAX_COMPLETION_TOKENS = 1024 #customize here, must be less than GPT4_MAX_TOTAL_TOKENS
MAX_PROMPT_TOKENS = GPT4_MAX_TOTAL_TOKENS - MAX_COMPLETION_TOKENS - GPT4_TOKENS_PER_MESSAGES
TIMEOUT_SECONDS = 480
encoding = tiktoken.get_encoding("cl100k_base")

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
async def generate_response(system_prompt, msgs):
    print(f"Generating GPT response for prompt:\n{msgs[-1]['content']}")
    gpt_response = "Sorry, an error occurred. Please try again."
    try:
        response = await asyncio.wait_for(openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[system_prompt]+msgs,
                max_tokens=MAX_COMPLETION_TOKENS,
                n=1,
                temperature=0.7,
        ), timeout = TIMEOUT_SECONDS)
        gpt_response = response["choices"][0]["message"]["content"].strip()
        print(f"GPT response:\n{gpt_response}")
        print(f"Prompt tokens: {response['usage']['prompt_tokens']}  Completion tokens: {response['usage']['completion_tokens']}  Total tokens: {response['usage']['total_tokens']}")
    except asyncio.TimeoutError:
        print("Timeout exceeded.")
    return gpt_response


def count_tokens(msg):
    num_tokens = GPT4_TOKENS_PER_MESSAGE
    for key, value in msg.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += GPT4_TOKENS_PER_NAME
    return num_tokens


def split_response(response, max_length):
    if len(response) <= max_length:
        return [response]
    response_chunks = []
    start_index = 0
    while start_index < len(response):
        end_index = min(start_index + max_length, len(response))
        split_index = response.rfind(" ", start_index, end_index) if end_index < len(response) else end_index
        end_index = split_index if split_index != -1 else end_index
        response_chunks.append(response[start_index:end_index])
        start_index = end_index + 1 if split_index != -1 else end_index
    return response_chunks