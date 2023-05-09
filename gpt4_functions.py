import os
import openai
import tiktoken
import backoff
import asyncio

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(f"Required environment variable OPENAI_API_KEY is not set.")
openai.api_key = os.environ["OPENAI_API_KEY"]
GPT4_MAX_TOTAL_TOKENS = 8192
GPT4_TOKENS_PER_MESSAGE = 3
GPT4_TOKENS_PER_NAME = 1
GPT4_TOKENS_PER_MESSAGES = 3
MAX_COMPLETION_TOKENS = 1024 #customize here
MAX_PROMPT_TOKENS = GPT4_MAX_TOTAL_TOKENS - MAX_COMPLETION_TOKENS - GPT4_TOKENS_PER_MESSAGES
TIMEOUT_SECONDS = 300
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(msg):
    num_tokens = GPT4_TOKENS_PER_MESSAGE
    for key, value in msg.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
            num_tokens += GPT4_TOKENS_PER_NAME
    return num_tokens

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
async def generate_response(msgs):
    print(f"Generating GPT response for prompt:\n{msgs[-1]['content']}")
    gpt_response = "Sorry, an error occurred. Please try again."
    try:
        response = await asyncio.wait_for(openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=msgs,
                max_tokens=MAX_COMPLETION_TOKENS,
                n=1,
                temperature=0.7,
        ), timeout = TIMEOUT_SECONDS)
        gpt_response = response["choices"][0]["message"]["content"].strip()
        print(f"GPT response:\n{gpt_response}")
        print(f"Prompt tokens: {response['usage']['prompt_tokens']}  Completion tokens: {response['usage']['completion_tokens']}  Total tokens: {response['usage']['total_tokens']}")
        # if response["usage"]["prompt_tokens"] > MAX_PROMPT_TOKENS:
        #     try: msgs.pop(1)
        #     except IndexError: pass

    # except openai.error.InvalidRequestError:
    #     if len(msgs) > 1:
    #         print("Too many prompt tokens, trying again...")
    #         msgs.pop(0)
    #         gpt_response = await generate_response(msgs)
    except asyncio.TimeoutError:
        print("Timeout exceeded.")

    return gpt_response

def split_response(response, max_length):
    if len(response) <= max_length:
        return [response]
    
    response_chunks = []
    start_i = 0
    print('bruh2')
    while start_i < len(response):
        print('bruh3')
        end_i = min(start_i + max_length, len(response))
        print('bruh4')
        if end_i < len(response):
            print('bruh5')
            end_i = (found := response.rfind(" ", start_i, end_i)) if found != -1 else end_i
            print('bruh6')
        print(f"start_i: {start_i}, end_i: {end_i}, found: {found}")  # Debug print
        response_chunks.append(response[start_i:end_i])
        start_i = end_i + 1 if found != -1 else end_i

    return response_chunks