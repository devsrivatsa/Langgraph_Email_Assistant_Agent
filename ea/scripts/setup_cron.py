"""Setup a cron job that runs every 20 minutes to check for new emails."""
import argparse
import asyncio
from typing import Optional
from langgraph_sdk import get_client

async def main(
        url: Optional[str] = None,
        minutes_since: int = 60,
):
    local_url = "http://localhost:8000"
    if url is None:
        client = get_client(url=local_url)
    else:
        client = get_client(url=url)
    
    await client.cron.create(
        "cron",
        schedule="*/20 * * * *",
        input={"minutes_since": minutes_since},   
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to run against",
    )
    parser.add_argument(
        "--minutes-since",
        type=int,
        default=60,
        help="Only process emails that are less than this many minutes old.",
    )

    args = parser.parse_args()
    asyncio.run(
        main(
            url=args.url,
            minutes_since=args.minutes_since,
        )
    )