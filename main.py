import asyncio
import aiohttp
import json
from typing import Optional

BASE_URL = "https://api.music.yandex.net/artists"
MAX_CONCURRENT = 100
CONSECUTIVE_FAIL_LIMIT = 100
START_ID = 4463796
MAX_RETRIES = 3

results = {}
lock = asyncio.Lock()


async def fetch_artist(session: aiohttp.ClientSession, artist_id: int) -> Optional[dict]:
    url = f"{BASE_URL}/{artist_id}/brief-info"
    retry_delay = 5
    try:
        for att in range(MAX_RETRIES):
            print(f'trying to fetch {url}')
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {"id": artist_id, "data": data}
                elif resp.status == 429:
                    wait = int(resp.headers.get("Retry-After", retry_delay))
                    print(f"[{artist_id}] Rate limited, waiting {wait}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait)
                    retry_delay *= 2
                elif resp.status == 404:
                    return None
                else:
                    print(f"[{artist_id}] Unexpected status: {resp.status}")
                    return None
    except Exception as e:
        print(f"[{artist_id}] Error: {e}")
        return None


async def main():
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    headers = {'User-Agent': 'PostmanRuntime/7.29.0'}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        artist_id = START_ID
        consecutive_failures = 0
        total_collected = 0

        with open("artists.jsonl", "a") as f:
            while consecutive_failures < CONSECUTIVE_FAIL_LIMIT:
                batch_ids = list(range(artist_id, artist_id + MAX_CONCURRENT))
                artist_id += MAX_CONCURRENT

                tasks = [fetch_artist(session, aid) for aid in batch_ids]
                batch_results = await asyncio.gather(*tasks)

                any_success = False
                for aid, result in zip(batch_ids, batch_results):
                    if result:
                        f.write(json.dumps({"id": aid, "data": result}) + "\n")
                        f.flush()
                        total_collected += 1
                        any_success = True
                        consecutive_failures = 0
                        print(f"[✓] ID {aid} collected")

                if not any_success:
                    consecutive_failures += MAX_CONCURRENT
                    print(
                        f"[~] Batch {batch_ids[0]}-{batch_ids[-1]} all empty ({consecutive_failures}/{CONSECUTIVE_FAIL_LIMIT} consecutive fails)")

        print(f"\nDone. Collected {total_collected} artists.")

if __name__ == "__main__":
    asyncio.run(main())