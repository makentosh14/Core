# async_utils.py

import asyncio
import sys

# Windows fix for asyncio event loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Async delay wrapper
async def async_sleep(seconds):
    await asyncio.sleep(seconds)

# Safe async retry handler
async def retry_async(func, retries=3, delay=2, *args, **kwargs):
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                await async_sleep(delay)
            else:
                raise e
