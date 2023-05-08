import os
import openai
import backoff
import asyncio

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(f"Required environment variable OPENAI_API_KEY is not set.")
openai.api_key = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4"
GPT4_MAX_TOTAL_TOKENS = 8192
MAX_COMPLETION_TOKENS = 1024 #customize here
MAX_PROMPT_TOKENS = GPT4_MAX_TOTAL_TOKENS - MAX_COMPLETION_TOKENS
TIMEOUT_SECONDS = 300

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
async def generate_response(system_prompt, message_history):
    print(f"Generating GPT response for prompt:\n{message_history[-1]['content']}")
    gpt_response = "Sorry, an error occurred. Please try again."
    try:
        response = await asyncio.wait_for(openai.ChatCompletion.acreate(
                model=MODEL,
                messages=system_prompt+message_history,
                max_tokens=MAX_COMPLETION_TOKENS,
                n=1,
                temperature=0.7,
        ), timeout = TIMEOUT_SECONDS)
        gpt_response = response["choices"][0]["message"]["content"].strip()
        print(f"GPT response:\n{gpt_response}")
        print(f"Prompt tokens: {response['usage']['prompt_tokens']}  Completion tokens: {response['usage']['completion_tokens']}  Total tokens: {response['usage']['total_tokens']}")
        if response["usage"]["prompt_tokens"] > MAX_PROMPT_TOKENS:
            message_history.pop(0)

    except openai.error.InvalidRequestError:
        if len(message_history) > 1:
            print("Too many prompt tokens, trying again...")
            message_history.pop(0)
            gpt_response = await generate_response(system_prompt, message_history)
    except asyncio.TimeoutError:
        print("Timeout exceeded.")

    return gpt_response

def split_response(response, max_length):
    if len(response) <= max_length:
        return [response]
    
    response_chunks = []
    start_i = 0
    while start_i < len(response):
        end_i = min(start_i + max_length, len(response))
        if end_i < len(response):
            end_i = (found := response.rfind(" ", start_i, end_i)) if found != -1 else end_i
        response_chunks.append(response[start_i:end_i])
        start_i = end_i + 1 if found != -1 else end_i

    return response_chunks