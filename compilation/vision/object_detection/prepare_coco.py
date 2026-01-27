import asyncio
import os
import random

import aiofiles  # Async file handler
import aiohttp
from datasets import load_dataset
from tqdm.asyncio import tqdm

# We use a Semaphore to limit the number of simultaneous downloads.
# 50 is a safe number. If you go too high (e.g., 500), servers might block you.
CONCURRENCY_LIMIT = 50


async def download_image(session, url, semaphore):
    # Acquire a "permit" from the semaphore
    async with semaphore:
        filename = os.path.basename(url.split("?")[0])
        filepath = os.path.join("coco-selected", filename)

        try:
            # We set a timeout for the connection
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()

                # Open file asynchronously and write chunks
                async with aiofiles.open(filepath, mode="wb") as f:
                    async for chunk in response.content.iter_chunked(1024):
                        await f.write(chunk)
                return 1
        except Exception:
            # Return 0 if download failed
            return 0


async def main():
    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split="train")[
        "URL"
    ]

    os.makedirs("coco-selected", exist_ok=True)

    # 2. Select URLs
    idx = random.sample(
        range(len(dataset)), 100
    )  # Change 100 to 10000 to see real speed!
    target_urls = [dataset[i] for i in idx]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    print(f"Starting async download of {len(target_urls)} images...")

    # 3. Create a single ClientSession (Connection Pooling)
    async with aiohttp.ClientSession() as session:
        tasks = [download_image(session, url, semaphore) for url in target_urls]

        # tqdm.gather runs the tasks and handles the progress bar
        results = await tqdm.gather(*tasks)

    print(
        f"Done! Successfully downloaded {sum(results)} out of {len(target_urls)} images."
    )


if __name__ == "__main__":
    # asyncio.run starts the event loop
    asyncio.run(main())
